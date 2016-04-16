from scipy import fftpack
from numpy import conj

def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = conj(fftpack.fftshift(fftpack.fftn(psf)))
    a = star_fft/psf_fft
    a[abs(psf_fft)==0] = 0
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(a)))
