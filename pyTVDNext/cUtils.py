import numpy as np
from scipy.io import loadmat
from easydict import EasyDict as edict
import torch
from numpy.fft import rfft2, rfft, fft
from scipy import signal


    
    
def GetAmatArr(Y, X, times, freqs, downrates=1, hs=None):
    """
    Input: 
        Y: A tensor with shape, d x dF x (dT-1)
        X: A tensor with shape, d x dF x (dT-1)
        times: A list of time points with length dT-1
        freqs: A list of frequency points with length dF
        downrates: The downrate factors for freq and time, determine how many A(s_i, t_i) matrix to be summed
        hs: the bandwidths for the kernel regression.
    Return:
        A d x d matrix, it is sum of dF x (dT-1)/downrates  A(s_i, t_i) matrix
    """
    d, dF, dT1 = X.shape
    hF, hT = hs
    if len(downrates) == 1:
        DRF = DRT = downrates
    else:
        DRF, DRT = downrates
        
    Ytrans = np.transpose(Y, (1, 2, 0)) # dF x (dT-1) * d
    Xtrans = np.transpose(X, (1, 2, 0))
        
    Amat = np.zeros((d, d))
    for idxf, fs in enumerate(freqs[::DRF]):
        for idxt, ts in enumerate(times[::DRT]):
            t_diff = times - ts
            freqs_diff = freqs- fs
            
            kernelst = 1/np.sqrt(2*np.pi) * np.exp(-t_diff**2/2/hT**2) # normal_pdf(x/h)
            kernelsf = 1/np.sqrt(2*np.pi) * np.exp(-freqs_diff**2/2/hF**2) # normal_pdf(x/h)
            
            kernelroot = (kernelsf[:, np.newaxis] * kernelst)  ** (1/2) #  dF x dT
            
            kerY = kernelroot[:, :, np.newaxis] * (Ytrans) # dF x (dT-1) x d
            kerX = kernelroot[:, :, np.newaxis] * (Xtrans) # dF x (dT-1) x d
            kerYmat = kerY.reshape(-1, d)
            kerXmat = kerX.reshape(-1, d)
            
            M = kerXmat.T.dot(kerXmat)/dF/dT1
            XY = kerYmat.T.dot(kerXmat)/dF/dT1
            
            invM = np.linalg.inv(M)
            Amat = Amat + XY.dot(invM)
    return Amat


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

def GetAmatTorch(Y, X, times, freqs, downrates=1, hs=None):
    """
    Input: 
        Y: A tensor with shape, d x dF x (dT-1)
        X: A tensor with shape, d x dF x (dT-1)
        times: A list of time points with length dT-1
        freqs: A list of frequency points with length dF
        downrates: The downrate factors for freq and time, determine how many A(s_i, t_i) matrix to be summed
        hs: the bandwidths for the kernel regression.
    Return:
        A d x d matrix, it is sum of dF x (dT-1)/downrates  A(s_i, t_i) matrix
    """
    d, dF, dT1 = X.shape
    hF, hT = hs
    if len(downrates) == 1:
        DRF = DRT = downrates
    else:
        DRF, DRT = downrates
        
    Ytrans = torch.tensor(np.transpose(Y, (1, 2, 0))) # dF x (dT-1) * d
    Xtrans = torch.tensor(np.transpose(X, (1, 2, 0)))
        
    Amat = torch.zeros(d, d)
    for idxf, fs in enumerate(freqs[::DRF]):
        for idxt, ts in enumerate(times[::DRT]):
            t_diff = times - ts
            freqs_diff = freqs- fs
            
            kernelst = 1/np.sqrt(2*np.pi) * np.exp(-t_diff**2/2/hT**2) # normal_pdf(x/h)
            kernelsf = 1/np.sqrt(2*np.pi) * np.exp(-freqs_diff**2/2/hF**2) # normal_pdf(x/h)
            kernelst = torch.tensor(kernelst)
            kernelsf = torch.tensor(kernelsf)
            
            kernelroot = (kernelsf.unsqueeze(-1) * kernelst)  ** (1/2) #  dF x dT
            
            kerY = kernelroot.unsqueeze(-1)* (Ytrans) # dF x (dT-1) x d
            kerX = kernelroot.unsqueeze(-1) * (Xtrans) # dF x (dT-1) x d
            kerYmat = kerY.reshape(-1, d)
            kerXmat = kerX.reshape(-1, d)
            
            M = kerXmat.T.mm(kerXmat)/dF/dT1
            XY = kerYmat.T.mm(kerXmat)/dF/dT1
            
            invM = torch.inverse(M)
            Amat = Amat + XY.mm(invM)
    return np.array(Amat.cpu())