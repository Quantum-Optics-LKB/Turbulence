# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 20:45:44 2022

@author: Tangui Aladjidi
"""
import numpy as np
import pyfftw
import pickle
import cupy as cp
import matplotlib.pyplot as plt
from skimage import feature, restoration
from sklearn import cluster
import numba

pyfftw.interfaces.cache.enable()

# kernel used to measure vorticity


@numba.njit((numba.float32[:, :], numba.float32[:, :], numba.float32[:, :]),
            fastmath=True, cache=True)
def nbcorr(im, filter, output):
    n_rows, n_cols = im.shape
    height, width = filter.shape
    for rr in range(n_rows - height + 1):
        for cc in range(n_cols - width + 1):
            for hh in range(height):
                for ww in range(width):
                    imgval = im[rr + hh, cc + ww]
                    filterval = filter[hh, ww]
                    output[rr, cc] += imgval * filterval


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def velocity(phase: np.ndarray, plot=True, dx: float = 1) -> tuple:
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

    # 1D unwrap
    phase_unwrap = np.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=np.float32)
    for k in range(phase_unwrap.shape[1]):
        phase_unwrap[0, k, :] = restoration.unwrap_phase(phase[k, :])
        phase_unwrap[1, :, k] = restoration.unwrap_phase(phase[:, k])

    # gradient reconstruction
    velo = np.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=np.float32)
    velo[0, :, :] = np.gradient(phase_unwrap[0, :, :], axis=1)
    velo[1, :, :] = np.gradient(phase_unwrap[1, :, :], axis=0)

    v_tot = np.hypot(velo[0], velo[1])
    V_k = pyfftw.interfaces.numpy_fft.fft2(velo)

    # Helmohltz decomposition fot the compressible part
    V_comp = -1j*np.sum(V_k*K, axis=0)/((np.sum(K**2, axis=0))+1e-15)
    v_comp = np.real(pyfftw.interfaces.numpy_fft.ifft2(1j*V_comp*K))

    # Helmohltz decomposition fot the incompressible part
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


def velocity_cp(phase: np.ndarray, plot=True, dx: float = 1) -> tuple:
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
    kx = 2*cp.pi*cp.fft.fftfreq(sx, d=dx)
    ky = 2*cp.pi*cp.fft.fftfreq(sy, d=dx)
    K = cp.array(cp.meshgrid(kx, ky))

    # 1D unwrap
    phase_unwrap = cp.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=cp.float32)
    phase_unwrap[0, :, :] = cp.unwrap(phase, axis=1)
    phase_unwrap[1, :, :] = cp.unwrap(phase, axis=0)

    # gradient reconstruction
    velo = cp.empty(
        (2, phase.shape[0], phase.shape[1]), dtype=cp.float32)
    velo[0, :, :] = cp.gradient(phase_unwrap[0, :, :], axis=1)
    velo[1, :, :] = cp.gradient(phase_unwrap[1, :, :], axis=0)

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


def vortex_detection(v_tot: np.ndarray, plot: bool = True) -> np.ndarray:
    """Detects the vortex positions using circulation calculation

    Args:
        v_tot (np.ndarray): Total velocity field
        plot (bool, optional): Whether to plot the result or not. Defaults to True.

    Returns:
        np.ndarray: A list of the vortices position and charge
    """
    w = np.gradient(v_tot[1])[0] - np.gradient(v_tot[0])[1]
    kernel = np.array([[1, -1], [-1, 1]], dtype=np.float32)
    coordinates = feature.peak_local_max(
        w**2, threshold_abs=0.5, min_distance=15, exclude_border=True)
    # detect vortices and their sign
    vortices = []
    o = np.empty((4, 4), dtype=np.float32)
    for i in range(len(coordinates)):
        x, y = coordinates[i]
        o[:, :] = w[x-2:x+2, y-2:y+2]
        nbcorr(o, kernel, o)
        if np.max(o) < 2:
            vortices.append([y, x, -1])
        elif np.min(o) > -2:
            vortices.append([y, x, 1])
    vortices = np.array(vortices, dtype=np.float32)
    if plot:
        plt.figure(1, figsize=[12, 9])
        plt.imshow(np.hypot(v_tot[0], v_tot[1]), cmap='gray')
        plt.scatter(vortices[:, 0], vortices[:, 1],
                    c=vortices[:, 2], cmap='bwr')
        plt.colorbar(label="Vorticity")
        plt.show()
    return vortices


@numba.njit(numba.int64[:, :](numba.float32[:, :]), fastmath=True, cache=True)
def rank_neighbors(vortices: np.ndarray) -> np.ndarray:
    """Calculates the distance to all the other vortices and
    returns the sorted list of neighbors

    Args:
        vortices (np.ndarray): Array of vortices [[x, y, l], ...]

    Returns:
        list: Ranked list of vortices indices sorted by distance
        to the starting vortex
    """
    ranking = np.zeros((vortices.shape[0], vortices.shape[0]), dtype=np.int64)
    distances = np.empty(vortices.shape[0], dtype=np.float32)
    for i in range(vortices.shape[0]):
        x0 = vortices[i, 0]
        y0 = vortices[i, 1]
        for j in range(vortices.shape[0]):
            x = vortices[j, 0]
            y = vortices[j, 1]
            dist = np.hypot(x-x0, y-y0)
            distances[j] = dist
        ranking[i, :] = np.argsort(distances)
    return ranking


def mutual_nearest_neighbors(ranking) -> np.ndarray:
    """Returns a list of pairs of mutual nearest neighbors and
    the product of their charges

    Args:
        ranking (np.ndarray): Distance matrix where R_ij is the j th nearest neighbor from i

    Returns:
        np.ndarray: A list of mutual nearest neighbors [[i, j, ll]] where
        ll is the product of their charges (dipole or pairs).
    """
    mutu = []
    for k in range(ranking.shape[0]):
        if not any(k in x for x in mutu):
            if ranking[ranking[k, 1], 1] == k:
                mutu.append([k, ranking[k, 1]])
    return mutu


def cluster_vortices(vortices: np.ndarray) -> list:
    """Clusters the vortices into dipoles, clusters and single vortices

    Args:
        vortices (np.ndarray): Array of vortices [[x, y, l], ...]

    Returns:
        list: dipoles, clusters, singles
    """
    queue = [k for k in range(vortices.shape[0])]
    singles = []
    dipoles = []
    clusters = []
    # compute mutual NN with ranking
    ranking = rank_neighbors(vortices)
    mutu = mutual_nearest_neighbors(ranking)
    # RULE 1
    for mut in mutu:
        # if it's a dipole
        ll = vortices[mut[0], 2]*vortices[mut[1], 2]
        if ll == -1:
            dipoles.append([mut[0], mut[1]])
            # remove the dipoles from the queue
            queue.remove(mut[0])
            queue.remove(mut[1])
    assert len(queue) + 2*(len(dipoles)) == len(
        vortices), "Something went wrong, the vortices numbers don't add up"
    # RULE 2
    for q in queue:
        cluster = [q]
        sgn = vortices[q, 2]
        next_closest = ranking[q, 1]
        sgn_next_closest = vortices[next_closest, 2]
        if sgn_next_closest != sgn:
            clusters.append(cluster)
            continue
        else:
            # SMORT
            # where we are on the "chain"
            pos = q
            # as long as the sign is OK, we keep going from closest to closest neighbors
            is_in_cluster = False
            is_in_dipoles = False
            while sgn_next_closest == sgn:
                # if it is in a cluster, append current element to cluster
                for x in clusters:
                    if next_closest in x:
                        is_in_cluster = True
                        x.append(pos)
                        break
                # if I stumble over a dipole, break while loop
                for x in dipoles:
                    if next_closest in x:
                        is_in_dipoles = True
                        break
                if is_in_cluster or is_in_dipoles:
                    break
                else:
                    cluster.append(next_closest)
                # this loop is infinite if we have a mutual NN pair, we need to check
                if next_closest == ranking[ranking[next_closest, 1], 1]:
                    break
                else:
                    pos = next_closest
                    next_closest = ranking[next_closest, 1]
                    sgn_next_closest = vortices[next_closest, 2]
            if not(is_in_cluster):
                clusters.append(cluster)
    # because we initialize the clusters with the singleton [q], we have a lot of clusters that are [q, q, r]
    # so we filter
    true_clusters = []
    for cluster in clusters:
        cluster = unique(cluster)
        if len(cluster) == 1:
            singles.append(cluster[0])
        else:
            true_clusters.append(cluster)
    return dipoles, true_clusters, singles


def main():
    phase = np.load("tur_density/v500_1_phase.npy")
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    velocity_cp(cp.asarray(phase), plot=False)
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    print(f"Elapsed time : {t_gpu*1e-3} s (GPU)")
    t0 = time.perf_counter()
    velo, v_inc, v_comp = velocity(phase, plot=False)
    t1 = time.perf_counter()-t0
    print(f"Helmholtz decomposition : {t1} s (CPU)")
    t0 = time.perf_counter()
    vortices = vortex_detection(velo, plot=False)
    t1 = time.perf_counter()-t0
    print(f"Vortex detection : {t1} s (CPU)")
    t0 = time.perf_counter()
    dipoles, clusters, singles = cluster_vortices(vortices)
    t1 = time.perf_counter()-t0
    print(f"Vortex clustering : {t1} s (CPU)")
    fig, ax = plt.subplots()
    flow = np.hypot(v_inc[0], v_inc[1])
    YY, XX = np.indices(flow.shape)
    im = ax.imshow(flow, vmax=0.5, cmap='viridis')
    ax.streamplot(XX, YY, v_inc[0], v_inc[1],
                  density=5, color='white', linewidth=1)
    for dip in dipoles:
        ax.plot(vortices[dip, 0], vortices[dip, 1], color='g', marker='o')
    for sing in singles:
        ax.scatter(vortices[sing, 0], vortices[sing, 1],
                   c=vortices[sing, 2], cmap='bwr')
    for cluster in clusters:
        if vortices[cluster[0], 2] == 1:
            c = 'r'
        else:
            c = 'b'
        ax.plot(vortices[cluster, 0], vortices[cluster,
                                               1], marker='o', color=c)
    ax.set_title(r'$v^{flow}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(0, 2047)
    ax.set_ylim(0, 2047)
    plt.colorbar(im, ax=ax)
    plt.show()


if __name__ == "__main__":
    import time
    main()
