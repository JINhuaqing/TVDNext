import numpy as np
from scipy.io import loadmat
from easydict import EasyDict as edict
import torch
from numpy.fft import rfft2, rfft, fft
from scipy import signal


def mat2Tensor(dat, h, stepsize=1):
    """
    To do FFT on the intial data which would yeild a 3-mode tensor with shape d x dF x dT
    dF: the length along the frequeny mode
    dT: the length along the time mode
    args
    dat: the array data to do FFT, d x dTinit
    h: the window size for h
    stepsize: the step size when screening
    """
    h = 10
    d, dTinit = dat.shape
    dT = dTinit - 2*h
    fftDatList = []
    for idx in range(h, dTinit-h, stepsize):
        low, up = idx-h, idx+h+1
        subDat = dat[:, low:up]
        fftSubDat = rfft2(subDat)
        fftDatList.append(fftSubDat)
    fftDat = np.abs(np.array(fftDatList))
    fftDat = np.transpose(fftDat, (1, 2, 0))
    d, dF, dT = fftDat.shape
    X = fftDat[:, :, :-1]
    Y = fftDat[:, :, 1:]
    
    return edict({"X":X, "Y":Y})