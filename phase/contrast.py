# -*-coding:utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from skimage.restoration import unwrap_phase

import sys
sys.path.append('/home/guillaume/Documents/cours/M2/stage/simulation')
import tools

    
def cache(radius, center=np.array([1024, 1024]), out=True, nb_pix=2048):
    
    Y, X = np.ogrid[:nb_pix, :nb_pix]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
 
    if out:
        mask = dist_from_center <= radius
    else:
        mask = dist_from_center > radius 
    
    return mask
    

def cache_2(x, y, center=np.array([1024, 1024]), out=True, nb_pix=2048):
    
    Y, X = np.ogrid[:nb_pix, :nb_pix]
    dist_from_center_x = np.abs(X - center[0])
    dist_from_center_y = np.abs(Y - center[1])
    
    if out:
        mask_x = dist_from_center_x <= x
        mask_y = dist_from_center_y <= y
        
        return mask_x & mask_y
        
    else:
        mask_x = np.where(dist_from_center_x > x, 1, 0)
        mask_y = np.where(dist_from_center_y > y, 1, 0)
        
        return np.where(mask_x + mask_y>0, True, False)
    
    
    

def maximum(im):

    im_fft = np.fft.fftshift(np.fft.fft2(im))

    # freq = np.sum(np.abs(im_fft), axis=1)
    freq = np.sum(np.abs(im_fft), axis=0)
    freq = freq/np.max(freq)

    maxi = []
    """ prom = 0.2
    while len(maxi) != 3:
        maxi = find_peaks(freq, distance=10, prominence=prom)[0]
        prom *=0.9
        if prom <= 0.01 :
            raise Exception("No fringe detected") 
    print(maxi)"""
    return list(maxi)
    

def im_osc(im, maxi=[957, 1024, 1100],  cont=True):

     
    nb_pix = len(im[0,:])
    axis   = 1
    im_fft = np.fft.fftshift(np.fft.fft2(im))
    
    freq = np.sum(np.abs(im_fft), axis=axis)
    freq = freq/np.max(freq)
    
    if not maxi:
        
        maxi=[]
        prominence = 0.2
        while len(maxi) != 3 :
            maxi = find_peaks(freq, distance=10, prominence=prominence)[0]
            prominence *= 0.9
            if prominence < 0.01:
                raise Exception("No fringe detected")
  

        
    if axis == 0:
        center = np.array([nb_pix//2, maxi[2]])
    else :
        center = np.array([maxi[2], nb_pix//2])
    radius = (maxi[2]-maxi[1])*0.85

    if cont:
        cont_size = int(1*radius/3)
        im_fft_cont = im_fft.copy()
        mask = cache(cont_size, out=False,center=[nb_pix//2, nb_pix//2],  nb_pix=nb_pix)
        im_fft_cont[mask] = 0
        
        im_cont = np.real(np.fft.ifft2(np.fft.fftshift(im_fft_cont)))    
    
    im_fft_fringe = im_fft.copy()    
    mask = cache(radius, center=center, out=False, nb_pix=nb_pix)
    im_fft_fringe[mask] = 0 
    
    
    #uncomment this to check the k space
    
    plt.figure(1)
    plt.imshow(np.log(np.abs(im_fft)))
    plt.colorbar()
    
    plt.figure(2)
    plt.imshow(np.log(np.abs(im_fft_cont)))
    plt.colorbar()
    
    plt.figure(3)
    plt.imshow(np.log(np.abs(im_fft_fringe)))
    plt.colorbar()
    plt.show()
    
    
 
    im_fringe = np.fft.ifft2(np.fft.fftshift(im_fft_fringe))
    
    if cont:
       return im_cont, im_fringe
       
    return im_fringe
    
    
def contr(im, maxi=None):

    im_cont, im_fringe = im_osc(im, maxi=maxi)

    """
    plt.figure(1)
    plt.imshow(im_cont)
    plt.colorbar()
    
    plt.figure(2)
    plt.imshow(np.abs(im_fringe))
    plt.colorbar()
    
    plt.figure(4)
    plt.imshow(2*np.abs(im_fringe)/im_cont, vmin=0, vmax=1)
    plt.colorbar()

    plt.show()
    """
    
    analytic = np.abs(im_fringe)
    
    return 2*analytic/im_cont
    
    
def phase(im, maxi=None):

    im_fringe = im_osc(im, maxi=maxi, cont=False)
    im_phase = unwrap_phase(np.angle(im_fringe))
    #im_phase = im_phase - np.linspace(im_phase[1024, 0], im_phase[1024,-1], 2048)
    
    """
    plt.figure(1)
    plt.imshow(np.angle(im_fringe))
    plt.colorbar()
    
    plt.figure(2) 
    plt.imshow(im_phase)
    plt.colorbar()
    plt.show()
    """

    return im_phase

def phase_dem(phase_im):
    grad = np.gradient(phase_im[700:-700, 700:-700])

    len_im =  len(phase_im[0,:])
    x = np.linspace(0, len_im, len_im)
    y = np.linspace(0, len_im, len_im)
    xx, yy = np.meshgrid(x, y)
    phase_demod = phase_im - np.mean(grad[1])*xx - np.mean(grad[0])*yy
    
    return phase_demod

if __name__=="__main__":

    path = "delta_n"

    im_j = tools.open_tif(path, 'F420')
    im_j = im_j/np.max(im_j)
    
    maxi = maximum(im_j)
    phase(im_j, maxi)

