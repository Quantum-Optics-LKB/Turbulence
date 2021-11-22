# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:09:50 2021

@author: Myrann
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
from scipy import fftpack
from scipy.optimize import curve_fit
import sys

imag = 50
n0=30
avg = 30
M = 6.7/2 #grandissement du faisceau sur la camera
# ROI=[600:1400,640:1440] TURBULENCE
# ROI=[400:800,440:840] PAS TURBULENCE
dx = 6.5e-3/M #mm on divise par le grandissement
# kx = np.fft.fftshift(np.fft.fftfreq(400, d=dx))
kx = np.fft.fftshift(np.fft.fftfreq(2048, d=dx)) #POUR ROI
# kx = np.fft.fftshift(np.fft.fftfreq(2048, d=dx)) #POUR PAS ROI
h,w = (2048,2048)
h1,w1 = (800,800)
delta_n = 1.16043e-4
hl = 1/((2*np.pi*384e12/(3e8))*np.sqrt(2*np.abs(delta_n))) #healing lenght
hl= hl*1e3 #en mm

o=0
# folder_nbrs = [15,20,25,30]
folder_nbrs = [15,20,25,30,35]

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    ext = [np.min(kx/2), np.max(kx/2), np.min(kx/2), np.max(kx/2)]
    im = plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5), extent=ext)
    return im

def f(x,a,b):
    return a*x+b

def kolv(X,a):
    return X**(-a)

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

# FIN PLOT

#Plot lois de puissance

def power_law(x, a, b):
    """ Calcul la loi de puissance avec les constantes a et b """
    return a*np.power(x, b)

t = ['15','20','25','30']
t = ['30']
nn=5
avg=30
ims_fft_av = np.zeros((h,w))
for k in t:
    print(f'\nTreating dataset : {k}')
    ims = np.empty((avg//nn, h, w))
    oms = np.empty((avg//nn, h1, w1))
    # oms = np.empty((avg,200,200)) ici je prend un petit ROI où y a pas turbulence
    for l in range(nn):
        sys.stdout.write(f'\rBatch {l+1}/{nn}')
        for i in range(avg//nn):   
            ims[i, :, :] = plt.imread("v{}/v{}_{:05d}.tif".format(k,k, l*(avg//nn)+i+1)).astype(float)
        ims_fft = np.fft.fftshift(np.fft.fft2(ims))
        ims_fft_av += np.mean(np.abs(ims_fft), axis=0)
    ims_fft_av /= nn
    
    
    ky_0_av = ims_fft_av[len(ims_fft_av)//2-5:len(ims_fft_av)//2+5,:]
    ky_0_av = np.sum(ky_0_av,0)
    ky_0_av = ky_0_av[1026:2048] #ROI 
    kx_0 = kx[1026:2048] #ROI

    
    # kymax = np.max(ky_0_av) #pour faire le fit lineaire calibration
    # kymax_pos = np.where(ky_0_av == kymax)[0][0] #prend position dans le tableau du max
    # kmes[j] = kymax_pos
    # j = j+1
    
        
    # popt,pcov = curve_fit(kolv, kx_0[40:60], np.abs(ky_0_av[40:60]))
    pars, cov = curve_fit(power_law, kx_0[40:60], np.abs(ky_0_av[40:60]), p0=[0, 0], bounds=(-np.inf, np.inf))
    # pars1, cov1 = curve_fit(power_law, kx_0[70:90], np.abs(ky_0_av[70:90]), p0=[0, 0], bounds=(-np.inf, np.inf))

    
    plt.figure(1,[16,9])
    plt.plot(kx_0, np.abs(ky_0_av),'-',label=f'{k}')
    plt.plot(kx_0[40:60], power_law(kx_0[40:60], *pars),'--',label=f'{pars[1]}')
    # plt.plot(kx_0[70:90], power_law(kx_0[70:90], *pars1),'--',label=f'{pars1[1]}')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlabel('$k_x$ mm-1',fontsize=15)
    plt.ylabel('Density',fontsize=15)
    # plt.title(f'Density profile (v={n}, avg={avg})')
    plt.title(f'Density profile (power law)',fontsize=15)
    # plt.ylim(1e4,10e8)
    plt.xlim(1,40)
    plt.savefig('Profile_powerlaw.pdf')
    ims_fft_av = 0
    
#Fin plot powerlaw

#DEBUT TEST
# n0 =30
# plt.figure(0)
# im = plt.imread(f'Tur2/v{n0}/v{n0}_00001.tif').astype(float)
# im = im[900:1700,670:1470]
# # im[0:600,0:640] = 0
# # im[1400:2048,1440:2048] = 0
# plt.imshow(im, plt.cm.gray)
# plt.savefig(f'Original image (v={n0})')

# plt.figure(1)
# im_fft = fftpack.fft2(im)
# im_fft_shift = fftpack.fftshift(im_fft)
# im_fft_shift = np.log(np.abs(im_fft_shift))
# plot_spectrum(im_fft_shift)
# plt.xlabel('kx (1/mm)')
# plt.ylabel('ky (1/mm)')
# plt.title(f'FT of turbulence (v={n0})')
# # plt.savefig(f'FT of turbulence(v={n0}).pdf')

# ky_0= im_fft_shift[len(im_fft_shift)//2,:]
# ky_0 = ky_0[400:800] #ROI AVEC TURB
# # ky_0_av = ky_0_av[200:400] #POUR ROI SANS TURB
# kx_0 = kx[400:800]*hl
# # kx_0 = kx[200:400] #ROI SANS TURB

# plt.figure(2,[17,8])
# plt.plot(kx_0, np.abs(ky_0),'-',label=f'v{n0}')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('kx*healing_lenght')
# plt.ylabel('Density')
# # plt.title(f'Density profile (v={n}, avg={avg})')
# plt.title(f'Density profile (avg={avg})')
# # plt.ylim(0,1e8)
# plt.xlim(0,15)

# plt.figure(3)
# im_fft2 = fftpack.fft2(im_fft_shift)
# im_fft_shift2 = fftpack.fftshift(im_fft2)
# im_fft_shift2 = np.abs(im_fft_shift2)
# plot_spectrum(im_fft_shift2)
# plt.xlabel('kx (1/mm)')
# plt.ylabel('ky (1/mm)')
# plt.title(f'FT of FT of turbulence (v={n0})')

#FIN TEST

# DEBUT PLOT DE TOUS LES PORIFLE OVERLAP
# j = 0
# kwei = [15,20,25,30]
# kmes = np.ones(len(kwei))
# for n in kwei:
#     folder = f"v{n}"
#     # k = 0
#     # filename = folder+f'/v{n}_0000{k+1}.tif'
#     # while os.path.isfile(filename):
#     #     filename = folder+f'/v{n}_0000{k+1}.tif'
#     #     k+=1
#     # print(f"J'ai trouvé {k} images")
#     # im = plt.imread(f'v{n}/v{n}_00001.tif').astype(float)
#     # im = im[600:1400,640:1440]

#     # plt.imshow(im, plt.cm.gray)
#     # plt.title(f'Original image (v={n})')
#     # plt.savefig(f'Original image (v={n})')
    
#     ims = np.empty((avg, h, w))
#     oms = np.empty((avg,h1,w1))
#     # oms = np.empty((avg,200,200)) ici je prend un petit ROI où y a pas turbulence
#     for i in range(avg):
#         ims[i, :, :] = plt.imread("v{}/v{}_{:05d}.tif".format(n,n, i+1)).astype(float)
#         # ims[i, :, :][0:600, 0:640] = 0
#         # ims[i, :, :][1400:2048,1440:2048] = 0
#         oms[i, :, :] = ims[i,:,:][600:1400,640:1440]
#     ims_fft = np.fft.fftshift(np.fft.fft2(oms)) 
#     ims_fft_av = np.mean(np.abs(ims_fft), axis=0)
    
    
#     ky_0_av = ims_fft_av[len(ims_fft_av)//2,:]
#     ky_0_av = ky_0_av[404:600] #ROI 
#     # ky_0_av = ky_0_av[1050:2048] # PAS ROI AVEC TURB
#     # ky_0_av = ky_0_av[200:400] #POUR ROI SANS TURB
#     kx_0 = kx[404:600]*hl #ROI
#     # kx_0 = kx[1050:2048] #PAS ROI
#     # kx_0 = kx[200:400] #ROI SANS TURB
    
#     # kymax = np.max(ky_0_av) #pour faire le fit lineaire calibration
#     # kymax_pos = np.where(ky_0_av == kymax)[0][0] #prend position dans le tableau du max
#     # kmes[j] = kymax_pos
#     # j = j+1
    
    
#     # mean = kx_0[kymax_pos]
#     # sigma = 2.5
#     # popt1,pcov1 = curve_fit(gaus, kx_0, ky_0_av,p0=[kymax,mean,sigma]) #pour fit les porifles
#     # kymax = popt1[1] #pour prendre le max de la gaussienne 
#     # kymax_pos = np.where(gaus(kx_0,*popt1) == kymax)[0][0] #prend position dans le tableau du max
#     # kmes[j] = kymax_pos
    
#     j = j+1
#     plt.figure(1,[19,13])
#     plt.plot(kx_0, np.abs(ky_0_av),'-',label=f'v{n}')
#     # plt.plot(kx_0, gaus(kx_0,*popt1),'--',label=f'fit v{n} :-/')
#     # plt.yscale('log')
#     plt.legend()
#     plt.xlabel('kx*healing_lenght')
#     plt.ylabel('Density')
#     # plt.title(f'Density profile (v={n}, avg={avg})')
#     plt.title(f'Density profile (avg={avg})')
#     # plt.ylim(0,1e8)
#     plt.xlim(0,3e-4)
#     plt.savefig(f'Profile {imag} images.pdf')
    
# popt,pcov = curve_fit(f, kwei, kmes)


# plt.figure(5)
# plt.plot(kwei,kmes,'ro')
# plt.title('Calibration k(slm) vs k(mesure)')
# plt.xlabel('k slm (1/mm)')
# plt.ylabel('k mes (1/mm)')
# # plt.plot(kmes,f(kmes,*popt),'g--',label=f'fit: a={popt[0]},b={popt[1]}')
# plt.legend()
# plt.savefig('kmes vs kmes.pdf')

##FIN PLOT DE TOUS LES PORIFLE OVERLAP

# k calibration
# n=30
# avg=10
# ims = np.empty((avg, h, w))
# oms = np.empty((avg,h1,w1))
# # oms = np.empty((avg,200,200)) ici je prend un petit ROI où y a pas turbulence
# for i in range(avg):
#     ims[i, :, :] = plt.imread("v{}_{:05d}.tif".format(n, i+1)).astype(float)
#     oms[i, :, :] = ims[i,:,:][600:1400,640:1440]
# ims_fft = np.fft.fftshift(np.fft.fft2(oms)) 
# ims_fft_av = np.mean(np.abs(ims_fft), axis=0)


# ky_0_av = ims_fft_av[len(ims_fft_av)//2,:]
# ky_0_av = ky_0_av[400:800] #ROI AVEC TURB
# # ky_0_av = ky_0_av[200:400] #POUR ROI SANS TURB
# kx_0 = kx[400:800]
# # kx_0 = kx[200:400] #ROI SANS TURB

# plt.figure(1,[20,14])
# plt.plot(kx_0, np.abs(ky_0_av),'-',label=f'v{n}')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('kx (1/mm)')
# plt.ylabel('Density')
# # plt.title(f'Density profile (v={n}, avg={avg})')
# plt.title(f'Density profile (avg={avg})')
# # plt.ylim(0,1e8)
# plt.xlim(1,10)
# plt.savefig('Profile no turbulence')


# Plot de turb vs no turb 1 graphes pour chaque k

# n = 30
# avg = 10

# plt.figure(0)
# im = plt.imread(f'v{n}_00001.tif').astype(float)
# # im = im[850:1650,660:1460]
# # im[0:600,0:640] = 0
# # im[1400:2048,1440:2048] = 0
# plt.imshow(im, plt.cm.gray)
# plt.savefig(f'Original image (v={n})')

# t = ['KTUR','KNOTUR']
# nn=2
# ims_fft_av = np.zeros((h,w))
# for k in t:
#     print(f'\nTreating dataset : {k}')
#     ims = np.empty((avg//nn, h, w))
#     oms = np.empty((avg//nn, h1, w1))
#     # oms = np.empty((avg,200,200)) ici je prend un petit ROI où y a pas turbulence
#     for l in range(nn):
#         sys.stdout.write(f'\rBatch {l+1}/{nn}')
#         for i in range(avg//nn):   
#             ims[i, :, :] = plt.imread("{}/v{}/v{}_{:05d}.tif".format(k,n,n, l*(avg//nn)+i+1)).astype(float)
#         ims_fft = np.fft.fftshift(np.fft.fft2(ims))
#         ims_fft_av += np.mean(np.abs(ims_fft), axis=0)
#     ims_fft_av /= nn
    
    
#     # ky_0_av = ims_fft_av[len(ims_fft_av)//2,:]
#     ky_0_av = ims_fft_av[len(ims_fft_av)//2-5:len(ims_fft_av)//2+5,:]
#     ky_0_av = np.sum(ky_0_av,0)
#     ky_0_av = ky_0_av[1026:2048] #ROI 
#     # ky_0_av = ky_0_av[1050:2048] # PAS ROI AVEC TURB
#     # ky_0_av = ky_0_av[200:400] #POUR ROI SANS TURB
#     kx_0 = kx[1026:2048]*hl #ROI
#     # kx_0 = kx[1050:2048] #PAS ROI
#     # kx_0 = kx[200:400] #ROI SANS TURB
    
#     # kymax = np.max(ky_0_av) #pour faire le fit lineaire calibration
#     # kymax_pos = np.where(ky_0_av == kymax)[0][0] #prend position dans le tableau du max
#     # kmes[j] = kymax_pos
#     # j = j+1
    
    
#     # mean = kx_0[kymax_pos]
#     # sigma = 2.5
#     # popt1,pcov1 = curve_fit(gaus, kx_0, ky_0_av,p0=[kymax,mean,sigma]) #pour fit les porifles
#     # kymax = popt1[1] #pour prendre le max de la gaussienne 
#     # kymax_pos = np.where(gaus(kx_0,*popt1) == kymax)[0][0] #prend position dans le tableau du max
#     # kmes[j] = kymax_pos
    
#     plt.figure(1,[16,9])
#     plt.plot(kx_0, np.abs(ky_0_av),'-',label=f'{k}')
#     # plt.plot(kx_0, gaus(kx_0,*popt1),'--',label=f'fit v{n} :-/')
#     plt.yscale('log')
#     plt.legend()
#     plt.xlabel('$k_x\\xi$',fontsize=15)
#     plt.ylabel('Density',fontsize=15)
#     # plt.title(f'Density profile (v={n}, avg={avg})')
#     plt.title(f'Density profile (v={n}, avg={avg})',fontsize=15)
#     # plt.ylim(1e4,10e8)
#     plt.xlim(0,0.5)
#     plt.savefig(f'Profile (v={n}).pdf')
#     ims_fft_av = 0
    
    
    
# PLOT CSV


# import csv
# nl = open("Temp.csv")
# def getColonne(file, n, sep=" "):
#     f = open(file, 'r')
#     r = csv.reader(f, delimiter=sep, quoting=csv.QUOTE_NONNUMERIC)
#     liste = list(r)
#     f.close()
#     res = []
#     if (n < len(liste[0])) and (n >= -len(liste[0])):
#         for ligne in liste:
#             res.append(ligne[n])
#     return res

# time = getColonne("Temp.csv", 0, sep=",")
# sat = getColonne("Temp.csv", 1, sep=",")
# temp = getColonne("Temp.csv", 3, sep=",")
# plt.figure(4444,[16,9])
# plt.plot(time,sat,label='Saturated abs')
# plt.plot(time,temp,label='Température')
# plt.xlabel('Time (s)')
# plt.ylabel('Transmission')
# plt.legend()





