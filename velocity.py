# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 20:45:44 2022

@author: Tangui Aladjidi
"""
from scipy import spatial
from numba import cuda
import numba
import matplotlib.pyplot as plt
import time
import math
import numpy as np
import pyfftw
import pickle
import cupy as cp
import networkx as nx
import multiprocessing
# from cupyx.scipy import spatial as spatial_cp
pyfftw.interfaces.cache.enable()

# simple timing decorator


def timer(func):
    """This function shows the execution time of
    the function object passed"""
    def wrap_func(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print(f'Function {func.__name__!r} executed in {(t2-t1)*1e3:.4f} ms')
        return result
    return wrap_func


def timer_repeat(func, *args, N_repeat=1000):
    t = np.zeros(N_repeat, dtype=np.float32)
    for i in range(N_repeat):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        t[i] = t1-t0
    print(f"{N_repeat} executions of {func.__name__!r} {np.mean(t)*1e3:.3f} +/- {np.std(t)*1e3:.3f} ms per loop (min : {np.min(t)*1e3:.3f} / max : {np.max(t)*1e3:.3f} ms / med: {np.median(t)*1e3:.3f})")
    return np.mean(t), np.std(t)


def timer_repeat_cp(func, *args, N_repeat=1000):
    t = np.zeros(N_repeat, dtype=np.float32)
    for i in range(N_repeat):
        start_gpu = cp.cuda.Event()
        end_gpu = cp.cuda.Event()
        start_gpu.record()
        func(*args)
        end_gpu.record()
        end_gpu.synchronize()
        t[i] = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    print(f"{N_repeat} executions of {func.__name__!r} {np.mean(t)*1e3:.3f} +/- {np.std(t)*1e3:.3f} ms per loop (min : {np.min(t)*1e3:.3f} / max : {np.max(t)*1e3:.3f} ms / med: {np.median(t)*1e3:.3f})")
    return np.mean(t), np.std(t)


@numba.njit(numba.float32[:, :](numba.float32[:, :], numba.int64), fastmath=True, cache=True, parallel=True)
def phase_jumps(phase, axis):
    dphi = np.zeros(phase.shape, dtype=np.float32)
    if axis == 0:
        for i in numba.prange(1, phase.shape[0]):
            for j in range(phase.shape[1]):
                dphi[i, j] = phase[i, j] - phase[(i-1) % phase.shape[0], j]
    elif axis == 1:
        for i in numba.prange(phase.shape[0]):
            for j in range(1, phase.shape[1]):
                dphi[i, j] = phase[i, j] - phase[i, (j-1) % phase.shape[1]]
    return dphi


@numba.njit(numba.float32[:, :](numba.float32[:, :, :]), fastmath=True, cache=True, parallel=True)
def phase_sum(velo):
    """Computes the phase gradient winding

    Args:
        velo (np.ndarray): Phase differences. velo[0, :, :] is d/dy phi (derivative along rows).

    Returns:
        np.ndarray: The winding array.
    """
    grad0 = velo[0]
    grad1 = velo[1]
    cont = np.empty(grad0.shape, dtype=np.float32)
    for i in numba.prange(cont.shape[0]):
        for j in range(cont.shape[1]):
            cont[i, j] = grad0[i % cont.shape[0], j % cont.shape[1]]
            cont[i, j] += grad0[i % cont.shape[0], (j+1) % cont.shape[1]]
            cont[i, j] -= grad0[(i+1) % cont.shape[0], (j+1) % cont.shape[1]]
            cont[i, j] -= grad0[(i+1) % cont.shape[0], j % cont.shape[1]]
            cont[i, j] -= grad1[i % cont.shape[0], j % cont.shape[1]]
            cont[i, j] += grad1[i % cont.shape[0], (j + 1) % cont.shape[1]]
            cont[i, j] += grad1[(i+1) % cont.shape[0], (j+1) % cont.shape[1]]
            cont[i, j] -= grad1[(i+1) % cont.shape[0], j % cont.shape[1]]
    return cont


@cuda.jit((numba.float32[:, :, :], numba.float32[:, :]), fastmath=True)
def phase_sum_cp(grad, cont):
    """Computes the phase gradient winding

    Args:
        grad (cp.ndarray): Phase differences. grad[0, :, :] is d/dy phi (derivative along rows).
        cont (cp.ndarray): output array
    Returns:
        None
    """
    i, j = numba.cuda.grid(2)
    if i < cont.shape[0] and j < cont.shape[1]:
        cont[i, j] = grad[0, i % cont.shape[0], j % cont.shape[1]]
        cont[i, j] += grad[0, i % cont.shape[0], (j+1) % cont.shape[1]]
        cont[i, j] -= grad[0, (i+1) % cont.shape[0], (j+1) % cont.shape[1]]
        cont[i, j] -= grad[0, (i+1) % cont.shape[0], j % cont.shape[1]]
        cont[i, j] -= grad[1, i % cont.shape[0], j % cont.shape[1]]
        cont[i, j] += grad[1, i % cont.shape[0], (j + 1) % cont.shape[1]]
        cont[i, j] += grad[1, (i+1) % cont.shape[0], (j+1) % cont.shape[1]]
        cont[i, j] -= grad[1, (i+1) % cont.shape[0], j % cont.shape[1]]


def velocity(phase: np.ndarray, dx: float = 1) -> np.ndarray:
    """Returns the velocity from the phase

    Args:
        phase (np.ndarray): The field phase
        dx (float, optional): the pixel size in m. Defaults to 1 (adimensional).

    Returns:
        np.ndarray: The velocity field [vx, vy]
    """
    # 1D unwrap
    phase_unwrap = np.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=np.float32)
    phase_unwrap[0, :, :] = np.unwrap(phase, axis=1)
    phase_unwrap[1, :, :] = np.unwrap(phase, axis=0)
    # gradient reconstruction
    velo = np.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=np.float32)
    velo[0, :, :] = np.gradient(phase_unwrap[0, :, :], dx, axis=1)
    velo[1, :, :] = np.gradient(phase_unwrap[1, :, :], dx, axis=0)
    return velo


def velocity_cp(phase: np.ndarray, dx: float = 1) -> np.ndarray:
    """Returns the velocity from the phase

    Args:
        phase (np.ndarray): The field phase
        dx (float, optional): the pixel size in m. Defaults to 1 (adimensional).

    Returns:
        np.ndarray: The velocity field [vx, vy]
    """
    # 1D unwrap
    phase_unwrap = cp.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=np.float32)
    phase_unwrap[0, :, :] = cp.unwrap(phase, axis=1)
    phase_unwrap[1, :, :] = cp.unwrap(phase, axis=0)
    # gradient reconstruction
    velo = cp.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=np.float32)
    velo[0, :, :] = cp.gradient(phase_unwrap[0, :, :], dx, axis=1)
    velo[1, :, :] = cp.gradient(phase_unwrap[1, :, :], dx, axis=0)
    return velo


def helmholtz_decomp(phase: np.ndarray, plot=True, dx: float = 1) -> tuple:
    """Decomposes a phase picture into compressible and incompressible velocities

    Args:
        phase (np.ndarray): 2D array of the field's phase
        plot (bool, optional): Final plots. Defaults to True.
        dx (float, optional): Spatial sampling size in m. Defaults to 1.
    Returns:
        tuple: (velo, v_incc, v_comp) a tuple containing the velocity field,
        the incompressible velocity and compressible velocity.
    """
    # try to load previous fftw wisdom
    try:
        with open("fft.wisdom", "rb") as file:
            wisdom = pickle.load(file)
            pyfftw.import_wisdom(wisdom)
    except FileNotFoundError:
        print("No FFT wisdom found, starting over ...")
    sy, sx = phase.shape
    # meshgrid in k space
    kx = 2*np.pi*np.fft.fftfreq(sx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(sy, d=dx)
    K = np.array(np.meshgrid(kx, ky))

    velo = velocity(phase, dx)

    v_tot = np.hypot(velo[0], velo[1])
    V_k = pyfftw.interfaces.numpy_fft.fft2(velo)

    # Helmholtz decomposition fot the compressible part
    V_comp = -1j*np.sum(V_k*K, axis=0)/((np.sum(K**2, axis=0))+1e-15)
    v_comp = np.real(pyfftw.interfaces.numpy_fft.ifft2(1j*V_comp*K))

    # Helmholtz decomposition fot the incompressible part
    v_inc = velo - v_comp
    # save FFT wisdom
    with open("fft.wisdom", "wb") as file:
        wisdom = pyfftw.export_wisdom()
        pickle.dump(wisdom, file)
    if plot == True:
        flow = np.hypot(v_inc[0], v_inc[1])
        YY, XX = np.indices(flow.shape)
        plt.figure(1, [12, 9])
        plt.imshow(v_tot, vmax=1)
        plt.title(r'$|v^{tot}|$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()

        plt.figure(2, [12, 9])
        plt.imshow(flow, vmax=1)
        plt.title(r'$|v^{inc}_{math}|$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()

        plt.figure(3, [12, 9])
        plt.imshow(np.hypot(v_comp[0], v_comp[1]), vmax=1)
        plt.title(r'$|v^{comp}_{math}|$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()

        # flows are calculated by streamplot
        plt.figure(4, [12, 9])
        plt.imshow(flow, vmax=0.5, cmap='viridis')
        plt.streamplot(XX, YY, v_inc[0], v_inc[1],
                       density=5, color='white', linewidth=1)
        plt.title(r'$v^{flow}$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.show()

    return velo, v_inc, v_comp


def helmholtz_decomp_cp(phase: np.ndarray, plot=True, dx: float = 1) -> tuple:
    """Decomposes a phase picture into compressible and incompressible velocities

    Args:
        phase (np.ndarray): 2D array of the field's phase
        plot (bool, optional): Final plots. Defaults to True.
        dx (float, optional): Spatial sampling size in m. Defaults to 1.
    Returns:
        tuple: (velo, v_incc, v_comp) a tuple containing the velocity field,
        the incompressible velocity and compressible velocity.
    """
    sy, sx = phase.shape
    # meshgrid in k space
    kx = 2*np.pi*cp.fft.fftfreq(sx, d=dx)
    ky = 2*np.pi*cp.fft.fftfreq(sy, d=dx)
    K = cp.array(cp.meshgrid(kx, ky))
    velo = velocity_cp(phase)
    v_tot = cp.hypot(velo[0], velo[1])
    V_k = cp.fft.fft2(velo)
    # Helmohltz decomposition fot the compressible part
    V_comp = -1j*cp.sum(V_k*K, axis=0)/((cp.sum(K**2, axis=0))+1e-15)
    v_comp = cp.real(cp.fft.ifft2(1j*V_comp*K))
    # Helmohltz decomposition fot the incompressible part
    v_inc = velo - v_comp
    if plot == True:
        flow = cp.hypot(v_inc[0], v_inc[1])
        YY, XX = np.indices(flow.shape)
        plt.figure(1, [12, 9])
        plt.imshow(cp.asnumpy(v_tot), vmax=1)
        plt.title(r'$|v^{tot}|$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()

        plt.figure(2, [12, 9])
        plt.imshow(cp.asnumpy(flow), vmax=1)
        plt.title(r'$|v^{inc}_{math}|$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()

        plt.figure(3, [12, 9])
        plt.imshow(cp.asnumpy(cp.hypot(v_comp[0], v_comp[1])), vmax=1)
        plt.title(r'$|v^{comp}_{math}|$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()

        # flows are calculated by streamplot
        plt.figure(4, [12, 9])
        plt.imshow(cp.asnumpy(flow), vmax=0.5, cmap='viridis')
        plt.streamplot(XX, YY, cp.asnumpy(v_inc[0]), cp.asnumpy(v_inc[1]),
                       density=5, color='white', linewidth=1)
        plt.title(r'$v^{flow}$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.show()

    return velo, v_inc, v_comp


def vortex_detection(phase: np.ndarray, plot: bool = False) -> np.ndarray:
    """Detects the vortex positions using circulation calculation

    Args:
        phase (np.ndarray): Phase field.
        plot (bool, optional): Whether to plot the result or not. Defaults to True.

    Returns:
        np.ndarray: A list of the vortices position and charge
    """
    velo = velocity(phase)
    windings = phase_sum(velo)
    plus_y, plus_x = np.where(windings > 2*np.pi)
    minus_y, minus_x = np.where(windings < -2*np.pi)
    vortices = np.zeros((len(plus_x)+len(minus_x), 3), dtype=np.float32)
    vortices[0:len(plus_x), 0] = plus_x
    vortices[0:len(plus_x), 1] = plus_y
    vortices[0:len(plus_x), 2] = 1
    vortices[len(plus_x):, 0] = minus_x
    vortices[len(plus_x):, 1] = minus_y
    vortices[len(plus_x):, 2] = -1
    if plot:
        velo = velocity(phase)
        fig, ax = plt.subplots(figsize=[12, 9])
        im = plt.imshow(phase, cmap='twilight_shifted')
        ax.scatter(vortices[:, 0], vortices[:, 1],
                   c=vortices[:, 2], cmap='bwr')
        fig.colorbar(im, ax=ax, label="Phase")
        plt.show()
    return vortices


def vortex_detection_cp(phase: cp.ndarray, plot: bool = False) -> cp.ndarray:
    """Detects the vortex positions using circulation calculation

    Args:
        phase (np.ndarray): Phase field.
        plot (bool, optional): Whether to plot the result or not. Defaults to True.

    Returns:
        np.ndarray: A list of the vortices position and charge
    """
    velo = velocity_cp(phase)
    windings = cp.empty_like(velo[0])
    tpb = 16
    bpgx = math.ceil(windings.shape[0]/tpb)
    bpgy = math.ceil(windings.shape[1]/tpb)
    phase_sum_cp[(bpgx, bpgy), (tpb, tpb)](velo, windings)
    plus_y, plus_x = cp.where(windings > 2*np.pi)
    minus_y, minus_x = cp.where(windings < -2*np.pi)
    vortices = cp.zeros((len(plus_x)+len(minus_x), 3), dtype=np.float32)
    vortices[0:len(plus_x), 0] = plus_x
    vortices[0:len(plus_x), 1] = plus_y
    vortices[0:len(plus_x), 2] = 1
    vortices[len(plus_x):, 0] = minus_x
    vortices[len(plus_x):, 1] = minus_y
    vortices[len(plus_x):, 2] = -1
    if plot:
        plt.figure(1, figsize=[12, 9])
        plt.imshow(cp.hypot(velo[0], velo[1]).get(), vmin=0, vmax=1)
        plt.scatter(vortices[:, 0].get(), vortices[:, 1].get(),
                    c=vortices[:, 2].get(), cmap='bwr')
        plt.colorbar(label="Vorticity")
        plt.show()
    return vortices


@numba.njit(numba.bool_[:](numba.int64[:]), cache=True, fastmath=True)
def mutual_nearest_neighbors(nn) -> np.ndarray:
    """Returns a list of pairs of mutual nearest neighbors and
    the product of their charges

    Args:
        nn (np.ndarray): array of nearest neighbors

    Returns:
        np.ndarray: A list of booleans telling if vortex i is a mutual NN pair without
        double counting.
    """
    mutu = np.zeros(nn.shape[0], dtype=np.bool_)
    for k in range(nn.shape[0]):
        next_closest = nn[k]
        if nn[next_closest] == k and not mutu[next_closest]:
            mutu[k] = True
    return mutu


def build_pairs(vortices: np.ndarray, nn: np.ndarray, mutu: np.ndarray, queue: np.ndarray):
    """Builds the dipoles and the pairs of same sign

    Args:
        vortices (np.ndarray): Vortices
        ranking (np.ndarray): Ranking matrix
        queue (np.ndarray): Vortices still under consideration
        mutu (np.ndarray): Mutual nearest neighbors
    Returns:
        dipoles, pairs, queue : np.ndarray dipoles, clusters and updated queue
    """
    closest = nn[mutu]
    ll = vortices[:, 2]*vortices[nn, 2]
    dipoles_ = mutu[ll[mutu] == -1]
    dipoles = np.empty((len(dipoles_), 2), dtype=np.int64)
    dipoles[:, 0] = dipoles_
    dipoles[:, 1] = closest[ll[mutu] == -1]
    # remove them from queue
    queue[dipoles[:, 0]] = -1
    queue[dipoles[:, 1]] = -1
    # check pairs
    pairs_ = mutu[ll[mutu] == 1]
    pairs = np.empty((len(pairs_), 2), dtype=np.int64)
    pairs[:, 0] = pairs_
    pairs[:, 1] = closest[ll[mutu] == 1]
    queue[pairs[:, 0]] = -1
    queue[pairs[:, 1]] = -1
    # update queue
    queue = queue[queue >= 0]
    return dipoles, pairs, queue


def clustering(vortices: np.ndarray, nn: np.ndarray, queue: np.ndarray,
               dipoles: np.ndarray, cluster_tree: nx.Graph) -> list:
    """Sorts the vortices into clusters after the dipoles have been removed

    Args:
        vortices (np.ndarray): Vortices
        ranking (np.ndarray): Vortices ranking matrix
        queue (np.ndarray): Vortices to sort
        dipoles (list): List of dipoles
        cluster_tree (nx.Graph): List of clusters in graph form

    Returns:
        None
    """
    def check_vortex(q):
        closest = nn[q]
        sgn = vortices[q, 2]
        sgn_closest = vortices[closest, 2]
        # if it's a different sign, we don't consider it
        if sgn_closest == sgn:
            # we check it's not a dipole
            is_in_dipoles = np.any(dipoles == closest)
            # if it's not, we connect them
            if not (is_in_dipoles):
                cluster_tree.add_edge(q, closest)
    map(check_vortex, queue)


def cluster_vortices(vortices: np.ndarray) -> list:
    """Clusters the vortices into dipoles, clusters and single vortices

    Args:
        vortices (np.ndarray): Array of vortices [[x, y, l], ...]

    Returns:
        list: dipoles, clusters, singles
    """
    queue = np.arange(0, vortices.shape[0], 1, dtype=np.int64)
    # store vortices in tree
    tree = spatial.KDTree(vortices[:, 0:2])
    # find nearest neighbors
    nn = tree.query(vortices[:, 0:2], k=2, workers=-1, p=2.0)[1]
    # nn[i] is vortex i nearest neighbor
    nn = nn[:, 1]
    mutu = mutual_nearest_neighbors(nn)
    mutu = queue[mutu]
    # RULE 1
    # instantiate graph for representation
    dipoles, pairs, queue = build_pairs(
        vortices, nn, mutu, queue)
    assert 2*len(dipoles) + 2*pairs.shape[0] + \
        len(queue) == vortices.shape[0], "PROBLEM count"
    # build graph
    cluster_graph = nx.Graph()
    for c in range(pairs.shape[0]):
        cluster_graph.add_node(pairs[c, 0], pos=vortices[pairs[c, 0], 0:2])
        cluster_graph.add_node(pairs[c, 1], pos=vortices[pairs[c, 1], 0:2])
        cluster_graph.add_edge(pairs[c, 0], pairs[c, 1])
    for q in queue:
        cluster_graph.add_node(q, pos=vortices[q, 0:2])
    # RULE 2
    clustering(
        vortices, nn, queue, dipoles, cluster_graph)
    connected = nx.connected_components(cluster_graph)
    clusters = []
    singles = []
    for c in connected:
        if len(list(c)) == 1:
            singles.append(list(c)[0])
        else:
            clusters.append(list(c))
    return dipoles, clusters, singles


def main():
    phase = np.load("v500_1_phase.npy")
    phase_cp = cp.asarray(phase)
    timer_repeat_cp(cp.asnumpy, phase_cp, N_repeat=25)
    # Vortex detection step
    vortices_cp = vortex_detection_cp(phase_cp, plot=False)
    vortices = vortex_detection(phase, plot=False)
    timer_repeat(vortex_detection, phase, N_repeat=25)
    timer_repeat(vortex_detection_cp, phase_cp, N_repeat=25)
    # Velocity decomposition in incompressible and compressible
    velo, v_inc, v_comp = helmholtz_decomp(phase, plot=False)
    # Clustering benchmarks
    dipoles, clusters, singles = cluster_vortices(vortices)
    timer_repeat(cluster_vortices, vortices, N_repeat=100)
    # Plot results
    flow = np.hypot(v_inc[O], v_inc[1])
    fig, ax = plt.subplots()
    YY, XX = np.indices(flow.shape)
    im = ax.imshow(flow)
    ax.streamplot(XX, YY, v_inc[0], v_inc[1],
                  density = 5, color = 'white', linewidth = 1)
    for dip in range(dipoles.shape[0]):
        ax.plot(vortices[dipoles[dip, :], 0],
                vortices[dipoles[dip, :], 1], color='g', marker='o')
    for sing in singles:
        ax.scatter(vortices[sing, 0], vortices[sing, 1],
                   c=vortices[sing, 2], cmap='bwr', vmin=-1, vmax=1)
    for cluster in clusters:
        if vortices[cluster[0], 2] == 1:
            c = 'r'
        else:
            c = 'b'
        ax.plot(vortices[cluster, 0], vortices[cluster,
                                               1], marker='o', color=c)
    ax.set_title(r'$|v^{inc}|$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(0, phase.shape[1])
    ax.set_ylim(0, phase.shape[0])
    plt.colorbar(im, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
