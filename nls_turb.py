#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: sskupin
Solves NLS equation with spectral operator splitting scheme

dA/dZ = i d^2A/dX^2 + i V(X,Z) A - i |A|^2 A

for given A(Z=0) for 0<Z<L
"""
import sys
import time

# import cupy as cp
import julia
import matplotlib.pyplot as plt
import numpy as np
import pyfftw
from julia import Main
from scipy.constants import c, epsilon_0, hbar, mu_0
from scipy.ndimage import zoom

# sys.path.append('/home/guillaume/Documents/cours/M2/stage/simulation')
# from azim_avg import azimuthalAverage as az_avg
# import contrast
# import tools


# @cp.fuse(kernel_name="nl_prop")
# def nl_prop(a: float, b: float, c: complex):
#     return c*a*cp.exp(1j*b*cp.abs(c)**2)


# def divide_where(a, b, th):
#     c = cp.empty_like(a)
#     c[b >= th] = a[b >= th]/b[b >= th]
#     c[b < th] = 0
#     return c


class NLSE:
    """Non linear Schrödinger Equation simulation class
    """

    def __init__(self, trans: float, puiss: float, waist: float, window: float, n2: float, L: float, NX: int = 1024, NY: int = 1024) -> None:
        """Instantiates the simulation

        Args:
            trans (float): Transmission
            puiss (float): Power in W
            waist (float): Waist size in m
            n2 (float): Non linear coeff in m^2/W
        """
        # liste des paramètres physiques
        self.n2 = n2
        self.waist = waist
        self.wl = 780e-9
        self.z_r = self.waist**2 * np.pi/self.wl
        self.k = 2 * np.pi / self.wl
        m = hbar * self.k/c
        self.L = L
        self.alpha = -np.log(trans)/self.L
        intens = 2*puiss/(np.pi*waist**2)
        self.E_00 = np.sqrt(2*intens/(c*epsilon_0))

        # number of grid points in X (even, best is power of 2 or low prime factors)
        self.NX = NX
        self.NY = NY
        self.window = window
        # transverse coordinate in units of W_0
        self.X, self.delta_X = np.linspace(-self.window/2, self.window/2, num=NX,
                                           endpoint=False, retstep=True, dtype=np.float32)
        self.Y, self.delta_Y = np.linspace(-self.window/2, self.window/2, num=NY,
                                           endpoint=False, retstep=True, dtype=np.float32)

        self.XX, self.YY = np.meshgrid(self.X, self.Y)

    # plot 2D amplitude on equidistant ZxX grid
    def plot_2D(self, ax, Z, X, AMP, title, cmap='viridis', label=r'$X$ (mm)', vmax=1):
        im = ax.imshow(AMP, aspect='equal', origin='lower', extent=(
            Z[0], Z[-1], X[0], X[-1]), cmap=cmap, vmax=vmax)
        ax.set_xlabel(label)
        ax.set_ylabel(r'$Y$ (mm)')
        ax.set_title(title)
        plt.colorbar(im)

        return

    # plot 1D amplitude and phase
    def plot_1D(self, ax, T, labelT, AMP, labelAMP, PHASE, labelPHASE, Tmin, Tmax):
        ax.plot(T, AMP, 'b')
        ax.set_xlim([Tmin, Tmax])
        ax.set_xlabel(labelT)
        ax.set_ylabel(labelAMP, color='b')
        ax.tick_params(axis='y', labelcolor='b')
        axbis = ax.twinx()
        axbis.plot(T, PHASE, 'r:')
        axbis.set_ylabel(labelPHASE, color='r')
        axbis.tick_params(axis='y', labelcolor='r')

        return

    # plot 1D amplitude and phase

    def plot_1D_amp(self, ax, T, labelT, AMP, labelAMP, Tmin, Tmax, color='b', label=''):
        ax.plot(T, AMP, color)
        ax.set_xlim([Tmin, Tmax])
        ax.set_xlabel(labelT)
        ax.set_ylabel(labelAMP, color=color)
        ax.tick_params(axis='y', labelcolor='b')

        return

    def slm(self, pattern, d_slm):
        phase = np.zeros((self.NY, self.NX))
        zoom_x = d_slm/self.delta_X
        zoom_y = d_slm/self.delta_Y
        phase_zoomed = zoom(pattern, (zoom_y, zoom_x))
        # compute center offset
        x_center = (self.NX - phase_zoomed.shape[1]) // 2
        y_center = (self.NY - phase_zoomed.shape[0]) // 2

        # copy img image into center of result image
        phase[y_center:y_center+phase_zoomed.shape[0],
              x_center:x_center+phase_zoomed.shape[1]] = phase_zoomed
        return phase

    def E_out(self, E_in: np.ndarray, z: float, plot=False) -> np.ndarray:
        """Propagates the field at a distance z

        Args:
            E_in (np.ndarray): Normalized input field (between 0 and 1)
            z (float): propagation distance in m
            plot (bool, optional): _description_. Defaults to False.

        Returns:
            np.ndarray: Propagated field in proper units V/m
        """

        # normalized longitudinal coordinate
        delta_Z = 1e-5*self.z_r
        Z = np.arange(0, z, step=delta_Z, dtype=np.float32)
        NZ = len(Z)
        # A = np.zeros([NX, NY]) + 0j
        A = pyfftw.empty_aligned((self.NX, self.NY), dtype=np.complex64)
        A[:, :] = self.E_00*E_in

        # definition of the Fourier frequencies for the linear step
        Kx = 2 * np.pi * np.fft.fftfreq(self.NX, d=self.delta_X)
        Ky = 2 * np.pi * np.fft.fftfreq(self.NY, d=self.delta_Y)

        Kxx, Kyy = np.meshgrid(Kx, Ky)
        propagator = np.exp(-1j * 0.5 * (Kxx**2 + Kyy**2)/self.k * delta_Z)
        # propagator_cp = cp.asarray(propagator)

        # def split_step_cp(A):
        #     """computes one propagation step"""
        #     # A = A * np.exp(-self.alpha*delta_Z)*cp.exp(1j * self.k * self.n2*c*epsilon_0 *
        #     #                                            cp.abs(A)**2 * delta_Z)
        #     A = nl_prop(np.exp(-self.alpha*delta_Z), self.k *
        #                 self.n2*c*epsilon_0 * delta_Z, A)
        #     A = cp.fft.fft2(A)
        #     A *= propagator_cp  # linear step in Fourier domain (shifted)
        #     A = cp.fft.ifft2(A)
        # return A

        # A = cp.asarray(A)
        # n2_old = self.n2
        # for i, z in enumerate(Z):
        #     if z > self.L:
        #         self.n2 = 0
        #     sys.stdout.write(f"\rIteration {i+1}/{len(Z)}")
        #     A[:, :] = split_step_cp(A)
        # print()
        # self.n2 = n2_old
        # A = cp.asnumpy(A)

        # for i in range(NZ-1):
        #     sys.stdout.write(f"\rIteration {i+1}/{NZ-1}")
        #     A = split_step_fftw(A)
        Main.eval(f"k = {self.k}")
        Main.eval(f"n2 = {self.n2}")
        Main.eval(f"c = {c}")
        Main.eval(f"epsilon_0 = {epsilon_0}")
        Main.eval(f"alpha = {self.alpha}")
        Main.eval(f"delta_X = {self.delta_X}")
        Main.eval(f"delta_Y = {self.delta_Y}")
        prop = Main.include("propagate.jl")
        t0 = time.perf_counter()
        A[:, :] = np.array(prop(A, z))[0, :, :]
        print(f"Time spent to solve : {time.perf_counter()-t0} s")
        if plot == True:
            fig = plt.figure(3, [9, 8])

            # plot amplitudes and phases
            a1 = fig.add_subplot(221)
            self.plot_2D(a1, self.X*1e3, self.Y*1e3, np.abs(A),
                         r'$|\psi|$', vmax=np.max(np.abs(A)))

            a2 = fig.add_subplot(222)
            self.plot_2D(a2, self.X*1e3, self.Y*1e3,
                         np.angle(A), r'arg$(\psi)$', cmap='twilight', vmax=np.pi)

            a3 = fig.add_subplot(223)
            lim = int(0.4*self.NX)
            im_fft = np.abs(np.fft.fftshift(
                np.fft.fft2(np.abs(A[lim:-lim, lim:-lim]))))
            Kx_2 = 2 * np.pi * np.fft.fftfreq(self.NX-2*lim, d=self.delta_X)
            len_fft = len(im_fft[0, :])
            self.plot_2D(a3, np.fft.fftshift(Kx_2), np.fft.fftshift(Kx_2), np.log10(im_fft),
                         r'$\mathcal{TF}(E_{out})$', cmap='viridis', label=r'$K_y$', vmax=np.max(np.log10(im_fft)))

            a4 = fig.add_subplot(224)
            self.plot_1D_amp(a4, Kx_2[1:-len_fft//2], r'$K_y$', np.mean(im_fft[len_fft//2-10:len_fft//2+10, len_fft//2+1:], axis=0),
                             r'$\mathcal{TF}(E_{out})$', np.fft.fftshift(Kx_2)[len_fft//2+1], np.fft.fftshift(Kx_2)[-1], color='b')
            a4.set_yscale('log')
            # a4.set_xscale('log')

            plt.tight_layout()
            # plt.savefig('turbulence.pdf', bbox_inches='tight', dpi=300, transparent=True)
            plt.show()

        return A

    def E_out_FWM(self, E_in_0: np.ndarray, E_in_1: np.ndarray, E_in_2: np.ndarray, z: float, plot=False):
        """
        n2 : non linear index (m²/W)
        waist : waist of the gaussian beam (m)
        trans : pourcentage of the beam going through the cell
        puiss : power of the beam at the entrance of the cell (W)
        """

        # normalized longitudinal coordinate
        delta_Z = 2e-5*self.z_r
        Z = np.arange(0, z, step=delta_Z, dtype=np.float32)
        NZ = len(Z)
        # A = np.zeros([NX, NY]) + 0j
        A = pyfftw.empty_aligned((3, self.NX, self.NY), dtype=np.complex64)
        A[:, :, :] = self.E_00*np.array([E_in_0, E_in_1, E_in_2])

        # definition of the Fourier frequencies for the linear step
        Kx = 2 * np.pi * np.fft.fftfreq(self.NX, d=self.delta_X)
        Ky = 2 * np.pi * np.fft.fftfreq(self.NY, d=self.delta_Y)

        Kxx, Kyy = np.meshgrid(Kx, Ky)
        propagator = np.exp(-1j * 0.5 * (Kxx**2 + Kyy**2)/self.k * delta_Z)
        # apply hyper gaussian filtering to kill parasitic high frequencies
        propagator *= np.exp(-((Kxx**2 + Kyy**2)/(2*0.4e6**2))**8)
        # plt.imshow(np.fft.fftshift(np.abs(propagator)))
        # plt.show()
        # propagator_cp = cp.asarray(propagator)
        threshold = 1e-1

        # def split_step_cp(A):
        #     """computes one propagation step"""
        #     # toto = cp.abs(2*A[2, :, :]*A[3, :, :]*cp.conj(A[0, :, :])/(A[1, :, :] *
        #     #                                                            (cp.abs(A[1, :, :]) > 1e-9)+1e-9*(cp.abs(A[1, :, :]) < 1e-9)))
        #     # plt.imshow(cp.asnumpy(
        #     #     toto))
        #     # plt.show()
        #     A[0, :, :] = A[0, :, :] * cp.exp(1j * self.k * self.n2*c*epsilon_0 * (
        #         cp.abs(A[0, :, :])**2 + 2*cp.abs(A[1, :, :])**2 + 2*cp.abs(A[2, :, :])**2 +
        #         divide_where(2*A[2, :, :]*A[1, :, :]*cp.conj(A[0, :, :]), A[0, :, :], threshold)) * delta_Z)
        #     A[1, :, :] = A[1, :, :] * cp.exp(1j * self.k * self.n2*c*epsilon_0 * (
        #         cp.abs(A[1, :, :])**2 + 2*cp.abs(A[0, :, :])**2 + 2*cp.abs(A[2, :, :])**2 +
        #         divide_where(A[0, :, :]**2 * cp.conj(A[2, :, :]), A[1, :, :], threshold)) * delta_Z)
        #     A[2, :, :] = A[2, :, :] * cp.exp(1j * self.k * self.n2*c*epsilon_0 * (
        #         cp.abs(A[2, :, :])**2 + 2*cp.abs(A[0, :, :])**2 + 2*cp.abs(A[1, :, :])**2 +
        #         divide_where(A[0, :, :]**2 * cp.conj(A[1, :, :]), A[2, :, :], threshold)) * delta_Z)
        #     A *= np.exp(-self.alpha*delta_Z)
        #     # plt.imshow(np.abs(cp.asnumpy(A[2, :, :])))
        #     # plt.show()
        #     A = cp.fft.fft2(A, axes=(1, 2))
        #     A *= propagator_cp  # linear step in Fourier domain (shifted)
        #     A = cp.fft.ifft2(A, axes=(1, 2))

        #     return A

        # A = cp.asarray(A)
        # n2_old = self.n2
        # for i, z in enumerate(Z):
        #     if z > self.L:
        #         self.n2 = 0
        #     sys.stdout.write(f"\rIteration {i+1}/{len(Z)}")
        #     A[:, :, :] = split_step_cp(A)
        # print()
        # self.n2 = n2_old
        # A = cp.asnumpy(A)
        # import various stuff to the Main Julia scope
        Main.eval(f"k = {self.k}")
        Main.eval(f"n2 = {self.n2}")
        Main.eval(f"c = {c}")
        Main.eval(f"epsilon_0 = {epsilon_0}")
        Main.eval(f"alpha = {self.alpha}")
        Main.eval(f"delta_X = {self.delta_X}")
        Main.eval(f"delta_Y = {self.delta_Y}")
        prop = Main.include('propagate_fwm.jl')
        # with open("propagate_fwm.jl") as f:
        #     A = Main.eval(f.read())
        t0 = time.perf_counter()
        A[:, :, :] = np.array(prop(A, z))[0, :, :, :]
        print(f"Time spent to solve : {time.perf_counter()-t0} s")

        if plot == True:
            for i in range(A.shape[0]):
                fig = plt.figure(i, [9, 8])
                fig.suptitle("$E_{}$ at z= {:.2e} m".format(i, z))
                # plot amplitudes and phases
                a1 = fig.add_subplot(221)
                self.plot_2D(a1, self.X*1e3, self.Y*1e3, np.abs(A[i, :, :]),
                             r'$|\psi|$', vmax=np.max(np.abs(A[i, :, :])))

                a2 = fig.add_subplot(222)
                self.plot_2D(a2, self.X*1e3, self.Y*1e3,
                             np.angle(A[i, :, :]), r'arg$(\psi)$', cmap='twilight', vmax=np.pi)

                a3 = fig.add_subplot(223)
                lim = int(0.4*self.NX)
                im_fft = np.abs(np.fft.fftshift(
                    np.fft.fft2(np.abs(A[i, lim:-lim, lim:-lim]))))
                Kx_2 = 2 * np.pi * \
                    np.fft.fftfreq(self.NX-2*lim, d=self.delta_X)
                len_fft = len(im_fft[0, :])
                self.plot_2D(a3, np.fft.fftshift(Kx_2)*1e-6, np.fft.fftshift(Kx_2)*1e-6, np.log10(im_fft),
                             r'$\mathcal{TF}(E_{out})$', cmap='viridis', label=r'$K_y$ in $\mu m^{-1}$', vmax=np.max(np.log10(im_fft)))

                a4 = fig.add_subplot(224)
                self.plot_1D_amp(a4, Kx_2[1:-len_fft//2]*1e-6, r'$K_y$ in $\mu m^{-1}$', np.mean(im_fft[len_fft//2-10:len_fft//2+10, len_fft//2+1:], axis=0),
                                 r'$\mathcal{TF}(E_{out})$', np.fft.fftshift(Kx_2)[len_fft//2+1]*1e-6, np.fft.fftshift(Kx_2)[-1]*1e-6, color='b')
                a4.set_yscale('log')
                # a4.set_xscale('log')

                plt.tight_layout()
                # plt.savefig('turbulence.pdf', bbox_inches='tight', dpi=300, transparent=True)
            plt.show()

        return A


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def flatTop_tur(sx, sy, length=150, width=60, k_counter=81, N_steps=81):
    output = np.zeros((sy, sx))
    Y, X = np.indices(output.shape)
    output[abs(X-output.shape[1]//2) < length/2] = 1
    output[abs(Y-output.shape[0]//2) > width/2] = 0

    grating_axe = X
    grating_axe = grating_axe % (sx/k_counter)
    grating_axe += abs(np.amin(grating_axe))
    grating_axe /= np.amax(grating_axe)

    grating_axe[X > output.shape[1]//2] *= -1
    grating_axe[X > output.shape[1]//2] += 1

    grating_axe_vert = Y
    grating_axe_vert = grating_axe_vert % (sy/N_steps)
    grating_axe_vert = normalize(grating_axe_vert)

    grating_axe = ((grating_axe+grating_axe_vert) % 1)*output
    return grating_axe


def flatTop_super(sx, sy, length=150, width=60, k_counter=81, N_steps=81):
    output = np.zeros((sy, sx))
    Y, X = np.indices(output.shape)
    output[abs(X-output.shape[1]//2) < length/2] = 1
    output[abs(Y-output.shape[0]//2) > width/2] = 0

    grating_axe = X
    grating_axe = grating_axe % (sx/k_counter)
    grating_axe += abs(np.amin(grating_axe))
    grating_axe /= np.amax(grating_axe)

    grating_axe[Y > output.shape[0]//2] *= -1
    grating_axe[Y > output.shape[0]//2] += 1

    grating_axe_vert = Y
    grating_axe_vert = grating_axe_vert % (sy/N_steps)
    grating_axe_vert = normalize(grating_axe_vert)

    grating_axe = ((grating_axe+grating_axe_vert) % 1)*output
    return grating_axe


if __name__ == "__main__":
    trans = 0.5
    n2 = -4e-10
    waist = 1e-3
    window = 2048*5.5e-6
    puiss = 1
    probe = 60e-3
    L = 5e-2
    simu = NLSE(trans, puiss, waist, window, n2, L, NX=256, NY=256)
    phase_slm = 2*np.pi*flatTop_tur(1272, 1024, length=1000, width=600)
    # phase_slm = 2*np.pi*flatTop_super(1272, 1024, length=1272, width=1024)
    phase_slm = simu.slm(phase_slm, 6.25e-6)
    # plt.imshow(phase_slm, cmap='twilight', vmin=-np.pi, vmSax=np.pi)
    # plt.show()
    E_in_0 = np.ones((simu.NY, simu.NX), dtype=np.complex64) * \
        np.exp(-(simu.XX**2 + simu.YY**2)/(2*simu.waist**2))
    E_in_1 = np.sqrt(probe/puiss)*np.ones((simu.NY, simu.NX), dtype=np.complex64) * \
        np.exp(-(simu.XX**2 + simu.YY**2)/(2*(0.33*simu.waist)**2))
    # E_in_0 += 5e-2*np.random.random(E_in_0.shape)
    # E_in_0 /= np.nanmax(E_in_0)
    E_in_0 *= np.exp(1j*phase_slm)  # + 0.1*np.random.random(E_in_0.shape)
    E_in_0 = np.fft.fftshift(np.fft.fft2(E_in_0))
    E_in_0[0:E_in_0.shape[0]//2+20, :] = 1e-10
    E_in_0[E_in_0.shape[0]//2+225:, :] = 1e-10
    E_in_0 = np.fft.ifft2(np.fft.ifftshift(E_in_0))
    # plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(E_in_0)))))
    # plt.imshow(np.abs(E_in_0))
    # plt.show()
    E_in_1 = 1e-6 * np.ones((simu.NY, simu.NX), dtype=np.complex64)
    E_in_2 = 1e-6 * np.ones((simu.NY, simu.NX), dtype=np.complex64)
    E_in_1 *= np.exp(-1j*1e4*simu.YY)
    E_in_2 *= np.exp(1j*1e4*simu.YY)
    # plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(E_in_1)))))
    # plt.imshow(np.abs(E_in_0))
    # plt.show()
    # E_in_3 *= np.exp(-1j*10e3*simu.YY)
    # for i in range(NZ-1):
    #     sys.stdout.write(f"\rIteration {i+1}/{NZ-1}")
    #     A = split_step_fftw(A)
    # fig, ax = plt.subplots(2, 2)
    # ax[0, 0].imshow(np.abs(E_in_0), origin="lower")
    # ax[0, 0].set_title("$E_0$")
    # ax[0, 1].imshow(np.abs(E_in_1), origin="lower")
    # ax[0, 1].set_title("$E_1$")plt.imshow(np.abs(cp.asnumpy(A[2, :, :])))
    # plt.show()
    # ax1[0, 0].imshow(np.angle(E_in_0), cmap="twilight",
    #                  vmin=-np.pi, vmax=np.pi, origin="lower")
    # ax1[0, 0].set_title("arg($E_0$)")
    # ax1[0, 1].imshow(np.angle(E_in_1), cmap="twilight",
    #                  vmin=-np.pi, vmax=np.pi, origin="lower")
    # ax1[0, 1].set_title("arg($E_1$)")
    # ax1[1, 0].imshow(np.angle(E_in_2), cmap="twilight",
    #                  vmin=-np.pi, vmax=np.pi, origin="lower")
    # ax1[1, 0].set_title("arg($E_2$)")
    # ax1[1, 1].imshow(np.angle(E_in_3), cmap="twilight",
    #                  vmin=-np.pi, vmax=np.pi, origin="lower")
    # ax1[1, 1].set_title("arg($E_3$)")
    # plt.show()
    A = simu.E_out(E_in_0, L, plot=True)
    # A = simu.E_out_FWM(E_in_0, E_in_1, E_in_2, L, plot=True)
