#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
# import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from skimage.restoration import unwrap_phase

import sys
sys.path.append('/home/guillaume/Documents/cours/M2/stage/simulation')
import tools
import contrast
from azim_avg import azimuthalAverage as az_avg
from scipy.signal import savgol_filter

# grand = 2.67/2
grand = 400/75/2
pix   = 6.5e-6

def get_absc(path, name_bg):
    im_bg = tools.open_tif(path, name_bg)
    im_bg = im_bg - np.mean(im_bg[0:100,0:100])
    im_bg = im_bg/np.max(im_bg)
    centre_x, centre_y = tools.centre(im_bg)

    absc   = np.linspace(0, 1024, 1024)
    profil = az_avg(im_bg, center=[centre_x, centre_y])[:1024]

    ptot, pcov = curve_fit(tools.gauss_fit, absc, profil, p0=[1, 400, 0])
    
    plt.plot(absc, profil)
    plt.plot(absc, tools.gauss_fit(absc, *ptot), label='waist = {:.3} mm'.format(ptot[1]*pix/grand*np.sqrt(2)*1e3))
    plt.legend()
    plt.show()

    # attention ! on fit ici l'intensité, donc il faut un sqrt(2) pour revenir au waist du champ électrique !
    waist = ptot[1]*pix/grand*np.sqrt(2)

    absc_phys = np.linspace(-centre_x/ptot[1], (2048-centre_x)/ptot[1] , 2048)

    #plt.plot(absc_phys, im_bg[:, centre_x], label='{}'.format(waist))
    #plt.show()

    return centre_x, centre_y, waist, absc_phys

def phase_dem(phase_im):
    roi_phase = phase_im[600:-600, 600:-600]
    grad = np.gradient(roi_phase)
    grad_y, grad_x = grad[0], grad[1]

    x = np.linspace(0, 2048, 2048)
    y = np.linspace(0, 2048, 2048)
    xx, yy = np.meshgrid(x, y)
    phase_demod = phase_im - np.mean(grad_x)*xx - np.mean(grad_y)*yy

    return phase_demod 


def r2(im):

    im = im - np.mean(im[0:100,0:100])
    im= im/np.max(im)

    centre_x, centre_y = tools.centre(im)
    x = (np.linspace(0,2048,2048) - centre_x)*pix/grand
    y = (np.linspace(0,2048,2048) - centre_y)*pix/grand

    XX, YY = np.meshgrid(x, y)
    R2 = XX**2 + YY**2

    #plt.plot(az_avg(im, center=[centre_y, centre_x]))
    #plt.show()

    #mask = contrast.cache(rad, center=[centre_x, centre_y], out=False)
   
    im[im<0.004] = 0
    
    #plt.imshow(R2*im)
    #plt.colorbar()
    #plt.show()

    return np.sum(R2*im)/np.sum(im)

def data_r2():

    #path = '/home/guillaume/Documents/cours/M2/stage/mesures/n2'
    
    #r2_tot = []
    #for j in range(1, 11):
    #    im = tools.open_tif(path, 'no_cell0_{:05}'.format(j))
    #    r2_tot.append(r2(im))

    #r0 = np.mean(r2_tot)

    path = '/home/guillaume/Documents/cours/M2/stage/mesures/n2'

    #im    = tools.open_tif(path, 'no_cell') 
    im    = tools.open_tif(path+'/detun_defoc/77_10', 'cell') 
    r2_0 = r2(im)
    centre_x, centre_y, waist, absc_phys = get_absc(path+'/detun_defoc/77_10', 'cell')
    print(waist)

    path  = path+'/detun_defoc'
    ang_tot = np.loadtxt(path+'/ang_tot.txt')

    r2_tot = []
    for j in range(0, len(ang_tot)-1):
        ang_j = str(ang_tot[j]).split('.')
        im    = tools.open_tif(path+'/{}_{:.2}'.format(ang_j[0], ang_j[1]), 'cell') 
        r2_im = r2(im)
        r2_tot.append(r2_im)
        sys.stdout.write("\rVariance du Rayon : {:.3} pour l'angle {}".format(r2_im, j))

    r2_tot = np.array(r2_tot[::-1])*1e6
    absc = np.linspace(0,1,len(r2_tot))
    plt.plot(absc, r2_tot - r2_0*1e6)
    plt.ylim(0, 1.2*(r2_tot[-1] - r2_0*1e6))
    plt.xlabel('Intensité (u.a.)')
    plt.ylabel(r'$R^2 - R^2_0$ (mm$^2$)')
    plt.savefig(path+'/r2.png')
    plt.show()
    
    np.savetxt(path+'/r2_tot.txt', np.column_stack((r2_tot)))

    return 

def ref_1():

    # path = '/home/guillaume/Documents/cours/M2/stage/mesures/n2/detun_ref'

    # ang_tot = np.loadtxt(path+'/ang_tot.txt')
    # ang_tot = 11
    
    # ang_j = str(ang_tot[0]).split('.')
    im    = tools.open_tif('delta_n',f'R{100}') 
    # im    = tools.open_tif('PHASE','REF') 
    im    = im - np.mean(im[0:100,0:100])
    im    = im/np.max(im)
    # maxi  = contrast.maximum(im)
    
    
    # m = np.array([30,50,70,100,120,140,160,180,200,300,400,500,600])
    m = np.array([5,20,100,200,420])
    #m = np.array([420])
    # m = ['']
        
    phase_tot = []
    for j in m:
        sys.stdout.write(f"\rIteration Power={j}")
    
        # ang_j = str(ang_tot[j]).split('.')
        im    = tools.open_tif('delta_n',f'F{j}')
        # im    = tools.open_tif('PHASE',f'FORK') 
        im    = im - np.mean(im[0:100,0:100])
        im    = im/np.max(im)
        
        centre_x, centre_y = tools.centre(im)
        
        im_cont, im_fringe = contrast.im_osc(im, cont=True)
        
        phase    = np.angle(im_fringe)
        analytic = np.abs(im_fringe)
        contr    = 2*analytic/im_cont 
        mask    = np.where(contr>0.1, 1, 0)*np.where(im_cont>0.01, 1, 0)  
        
        im_avg  = az_avg(np.abs(im_cont), center=[centre_x, centre_y])
        
        plt.figure(4839)
        plt.plot(im_avg)

        rad_max = np.argmin(np.where(im_avg<0.04, -1, 0))
        # print(rad_max)
        
        plt.figure(13)
        plt.imshow(contr, vmin=0, vmax=1)
        plt.imshow(np.where(mask==1, contr, np.nan), vmin=0, vmax=1)
        plt.colorbar()

        if j == 600:
            plt.figure(35)
            plt.imshow(phase)
            plt.imshow(np.where(mask==1, phase, np.nan))
            plt.colorbar()
        
        phase = contrast.phase_dem(unwrap_phase(phase))
        phase = np.where(mask==1, phase, np.nan)
        
        y, x = np.indices(im.shape)
        r = np.hypot(x - centre_x, y - centre_y)
        
        im_avg = az_avg(phase[np.where(mask)], center=[centre_x, centre_y], r=r[np.where(mask)])
        im_avg = savgol_filter(im_avg, 11, 3)
        
        if len(m)==1:
            plt.figure(22)        
            plt.plot(-im_avg+ im_avg[0])
            plt.plot(-im_avg[:rad_max]+ im_avg[0])
            plt.xlabel('r $\mu$m')
            plt.ylabel('Non-linear phase (rad)')
            plt.title('Phase profile')
            plt.savefig('Phase profile.pdf')
            plt.show()
            
            velocity = (3e8)*np.gradient(im_avg[:rad_max]- im_avg[0])/((2*np.pi*384.230e12)/3e8)
            plt.figure(222)
            plt.plot(-velocity)
            plt.xlabel('r $\mu$m')
            plt.ylabel('Fluid velocity (1e5 m/s)')
            plt.title('Velocity profile')
            plt.savefig('Velocity_profile.pdf')
            plt.show()
        
        # if j==0:
        #     np.savetxt(path+'/phase_test.txt', np.column_stack((im_avg[:rad_max])))
                    
        phase_tot.append(np.max(im_avg[:rad_max]) - np.min(im_avg[:rad_max]))
        
        
        
    absc = m*1e-3 
    np.savetxt('phase_2.txt', np.column_stack((phase_tot)))  
    plt.figure(999)
    plt.plot(absc, phase_tot)
    plt.xlabel('Power (W)')
    plt.ylabel('Phase non linéaire (rad)')
    # plt.savefig('phase_2.pdf')
    plt.show()

    return


    
def fit(intens, n2, I_sat):
    return n2*intens/(1+intens/I_sat)

if __name__=="__main__":
        
    #data_r2()

    ref_1()

    phase  = np.loadtxt('phase_2.txt')
    dn     = phase/(2*np.pi/780e-9 * 0.075)
    # puiss  = np.array([30,50,70,100,120,140,160,180,200,300,400,500,600])*1e-3
    puiss  = np.array([5,20,100,200,420])*1e-3
    waist  = 5e-4
    intens = 2*puiss/(np.pi*waist**2)
    

    
    ptot, pcov = curve_fit(fit, intens, dn, p0=[2.2e-10, 1e5], bounds=([1e-11, 1e4], [1e-9, 1e7]))
    print(np.sqrt(np.diag(pcov)))
    

    plt.figure(3200)
    plt.plot(intens/1e4, dn, 'o', label='Mesures')
    plt.plot(intens/1e4, fit(intens, *ptot), label=r'$I_s$ = {:.3} W/cm$^2$ et $n_2$ = {:.3} m$^2$/W'.format(ptot[1]/1e4, ptot[0]))
    plt.xlabel(r'Intensité (W/cm$^2$)')
    plt.ylabel(r'$\Delta$ n')
    plt.legend()
    plt.savefig('Delta_n.pdf')
    
    # plt.figure(6666)
    # plt.plot(intens/1e4, velocity)
    # plt.title('Velocity profile')
    # plt.xlabel(r'Intensité (W/cm$^2$)')
    # plt.ylabel('Velocity')
    # plt.savefig('Velocity profile.pdf')