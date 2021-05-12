import numpy as np
from scipy.io import loadmat
from easydict import EasyDict as edict
import torch
from numpy.fft import rfft2, rfft, fft
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm
from pathlib import Path
from collections import defaultdict as ddict
import torch
import time
from Rfuns import bw_nrd0_R


def mat2Tensor(dat, fs, bandsCuts=None, Nord=None, q=None):
    """
    To do filter on the intial data which would yeild a 3-mode tensor with shape d x dF x dT
    dF: the length along the frequeny mode
    dT: the length along the time mode
    args
    dat: the array data to do filter, d x dT
    fs: the sampling freq of the dat
    bandsCuts: the cirtical freqs to use
    Nord: The order of the filter
    """
    if Nord is None:
        Nord = 10
    if bandsCuts is None:
        bandsCuts = [[2, 3.5], [4, 7], [8, 12], [13, 30], [30, 80]]
    filDats = []
    for bandsCut in bandsCuts:
        sos = signal.butter(Nord, bandsCut, btype="bp", fs=fs, output="sos")
        filDat = signal.sosfilt(sos, dat, axis=-1)
        filDats.append(filDat)
    filDats = np.array(filDats)
    filDats = np.transpose(filDats, (1, 0, 2))
    if q is not None:
        filDats = signal.decimate(filDats, q=q)
    
    X = filDats[:, :, :-1]
    Y = filDats[:, :, 1:]
    
    return edict({"X":X, "Y":Y})


def GetAmatArr(Y, X, times, freqs, downrates=1, hs=None):
    """
    Input: 
        Y: A tensor with shape, d x dF x dT
        X: A tensor with shape, d x dF x dT
        times: A list of time points with length dT
        freqs: A list of frequency points with length dF
        downrates: The downrate factors for freq and time, determine how many A(s_i, t_i) matrix to be summed
        hs: the bandwidths for the kernel regression.
    Return:
        A d x d matrix, it is sum of dF x dT/downrates  A(s_i, t_i) matrix
    """
    d, dF, dT = X.shape
    if hs is None:
        hs = [bw_nrd0_R(freqs), bw_nrd0_R(times)]
    hF, hT = hs
    if len(downrates) == 1:
        DRF = DRT = downrates
    else:
        DRF, DRT = downrates
        
    Ytrans = np.transpose(Y, (1, 2, 0)) # dF x dT * d
    Xtrans = np.transpose(X, (1, 2, 0))
        
    Amat = np.zeros((d, d))
    for idxf, fs in enumerate(freqs[::DRF]):
        for idxt, ts in enumerate(times[::DRT]):
            t_diff = times - ts
            freqs_diff = freqs- fs
            
            kernelst = 1/np.sqrt(2*np.pi) * np.exp(-t_diff**2/2/hT**2) # normal_pdf(x/h)
            kernelsf = 1/np.sqrt(2*np.pi) * np.exp(-freqs_diff**2/2/hF**2) # normal_pdf(x/h)
            
            kernelroot = (kernelsf[:, np.newaxis] * kernelst)  ** (1/2) #  dF x dT
            
            kerY = kernelroot[:, :, np.newaxis] * (Ytrans) # dF x dT x d
            kerX = kernelroot[:, :, np.newaxis] * (Xtrans) # dF x dT x d
            kerYmat = kerY.reshape(-1, d)
            kerXmat = kerX.reshape(-1, d)
            
            M = kerXmat.T.dot(kerXmat)/dF/dT
            XY = kerYmat.T.dot(kerXmat)/dF/dT
            
            invM = np.linalg.inv(M)
            Amat = Amat + XY.dot(invM)
    return Amat


# +
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

def GetAmatTorch(Y, X, times, freqs, downrates=1, hs=None):
    """
    Input: 
        Y: A tensor with shape, d x dF x dT
        X: A tensor with shape, d x dF x dT
        times: A list of time points with length dT
        freqs: A list of frequency points with length dF
        downrates: The downrate factors for freq and time, determine how many A(s_i, t_i) matrix to be summed
        hs: the bandwidths for the kernel regression.
    Return:
        A d x d matrix, it is sum of dF x dT/downrates  A(s_i, t_i) matrix
    """
    d, dF, dT = X.shape
    if hs is None:
        hs = [bw_nrd0_R(freqs), bw_nrd0_R(times)]
    hF, hT = hs
    if len(downrates) == 1:
        DRF = DRT = downrates
    else:
        DRF, DRT = downrates
        
    Ytrans = Y.permute((1, 2, 0))
    # Xtrans = torch.tensor(np.transpose(X, (1, 2, 0)))
    Xtrans = X.permute((1, 2, 0))
        
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
            
            kerY = kernelroot.unsqueeze(-1)* (Ytrans) # dF x dT x d
            kerX = kernelroot.unsqueeze(-1) * (Xtrans) # dF x dT x d
            kerYmat = kerY.reshape(-1, d)
            kerXmat = kerX.reshape(-1, d)
            
            M = kerXmat.T.mm(kerXmat)/dF/dT
            XY = kerYmat.T.mm(kerXmat)/dF/dT
            
            invM = torch.inverse(M)
            Amat = Amat + XY.mm(invM)
    return np.array(Amat.cpu())


# -

class OneStepOpt():
    """
        I concatenate the real and image part into one vector.
    """
    def __init__(self, X, Y, pUinv, fixedParas, lastTheta, penalty="SCAD", is_ConGrad=False, is_std=None, **paras):
        """
         Input: 
             Y: A tensor with shape, d x dF x dT
             X: A tensor with shape, d x dF x dT
             pUinv, the first R row of the inverse of the eigen vector matrix, R x d, complex data
             fixedParas: The fixed parameters when optimizing, real data 
                 when update \mu, 2R x dT
                 when update \nu, 2R x dF
             lastTheta: The parameters for optimizing at the last time step, initial parameters, vector of 2R(D-1), real data
             penalty: The penalty type, "SCAD" or "GroupLasso"
             is_ConGrad: Whether to use conjugate gradient method to update gamma or not. 
                        When data are large, it is recommended to use it
             is_std: Whether to std the gam and theta or not
             paras:
                 beta: tuning parameter for iteration
                 alp: tuning parameter for iteration
                 rho: a vector of length (D-1)2R, real data
                 lam: the parameter for SCAD
                 a: the parameter for SCAD, > 1+1/beta
                 iterNum:  integer, number of iterations
                 iterC: decimal, stopping rule
                 eps: decimal, stopping rule for conjugate gradient method
        """
        
        parasDefVs = {"a": 2.7,  "beta": 1, "alp": 0.9,  "rho": None,  "lam": 1e2, 
                      "iterNum": 100, "iterC": 1e-4, "eps": 1e-6}
        
        self.paras = edict(parasDefVs)
        for key in paras.keys():
            self.paras[key] = paras[key]
        
            
            
        self.d, self.dF, self.dT = X.shape
        self.R2, _ = fixedParas.shape
        self.D = int(lastTheta.shape[0]/self.R2+1)
        self.nD = self.dF if self.D == self.dT else self.dT
        
        self.pUinv = pUinv
        self.X = X.type_as(self.pUinv)
        self.Y = Y.type_as(self.pUinv) # Make them complex
        
        R = int(self.R2/2)
        if self.paras.rho is None:
            self.paras.rho = torch.ones(self.R2*(self.D-1))
            
            
        self.fixedParas = torch.complex(fixedParas[:R, :], fixedParas[R:, :]).type_as(self.pUinv) # R x nD
        
        self.lastTheta= lastTheta
        
        
        self.newVecGam = None
        self.newVecGamStd = None
        self.halfRho = None
        self.rho = self.paras.rho
        self.lam = self.paras.lam
        self.a = self.paras.a 
        self.iterNum = self.paras.iterNum
        self.iterC = self.paras.iterC
        self.penalty = penalty.lower()
        if is_ConGrad is None:
            is_ConGrad = self.D >= 1e3
        self.is_ConGrad = is_ConGrad
        if is_std is None:
            is_std = self.D == self.dF
        self.is_std = is_std
            
        self.leftMat = None
        self.leftMatVec = None
        self.NewXYR2Sum = None
        self.lastThetaStd = None
        
        self.NewXr = None
        self.NewYr = None
        self.NewXi = None
        self.NewYi = None
        
        self.GamMat = None
        self.ThetaMat = None
        self.GamMatStd = None
        self.ThetaMatStd = None
        
    def obtainNewData(self):
        pY = self.Y.permute(1, 2, 0) # dF x dT x d
        pX = self.X.permute(1, 2, 0)
        cNewX = pX.matmul(self.pUinv.T)  # dF x dT x R
        if self.D == self.dF:
            cNewY = pY.matmul(self.pUinv.T) * (1/self.fixedParas.T) # dF x dT x R
        else:
            cNewY = pY.matmul(self.pUinv.T) * (1/self.fixedParas.T).unsqueeze(1) # dF x dT x R
        self.NewXr = cNewX.real
        self.NewYr = cNewY.real
        self.NewXi = cNewX.imag
        self.NewYi = cNewY.imag
        
    def _AmatOpt(self, vec):
        rVec1 = self.leftMatVecP1 * vec
        rVec2 = self.paras.beta * DiffMatTOpt(DiffMatOpt(vec, self.R2), self.R2)
        return rVec1 + rVec2
    
    def _ConjuGrad(self, vec, maxIter=1000):
        """ 
        Ax = vec
        """
        eps = self.paras.eps
        
        xk = torch.zeros_like(vec)
        rk = vec - self._AmatOpt(xk)
        pk = rk
        if torch.norm(rk) <= eps:
            return xk
        
        for k in range(maxIter):
            alpk = torch.sum(rk**2) / torch.sum(pk * self._AmatOpt(pk))
            xk = xk + alpk * pk 
            
            rk_1 = rk
            rk = rk - alpk * self._AmatOpt(pk)
            
            if torch.norm(rk) <= eps:
                break 
                
            betk = torch.sum(rk**2)/torch.sum(rk_1**2)
            pk = rk + betk * pk
            
        return xk
        
    
    def updateVecGam(self):
        """
            I use conjugate gradient to solve it. 
            Update the Gamma matrix, first step 
        """
        self.DiffMatSq = genDiffMatSqfn(self.R2, self.D) # R2D x R2D
        
        optAxis = 1 if self.D == self.dF else 0
        
        if self.leftMat is None:
            NewXSq = self.NewXr**2 + self.NewXi**2 # dF x dT x R
            NewXSqR2 = torch.cat((NewXSq, NewXSq), dim=2) # dF x dT x 2R
            NewXSqR2Sum = NewXSqR2.sum(axis=optAxis) # dF x 2R or dT x 2R
            self.leftMat = torch.diag(NewXSqR2Sum.flatten()).to_sparse() +  \
                    self.paras.beta * self.DiffMatSq
        
        if self.NewXYR2Sum is None:
            NewXY1 = self.NewXr * self.NewYr + self.NewXi * self.NewYi
            NewXY2 = - self.NewXi * self.NewYr + self.NewXr * self.NewYi
            NewXYR2 = torch.cat((NewXY1, NewXY2), dim=2) # dF x dT x 2R
            self.NewXYR2Sum = NewXYR2.sum(axis=optAxis) # dF x 2R or dT x 2R
        rightVec = self.NewXYR2Sum.flatten() + \
                    DiffMatTOpt(self.rho + self.paras.beta * self.lastTheta, self.R2)
        
        # self.newVecGam, = torch.inverse(self.leftMat).matmul(rightVec)
        # Better way to do so
        self.newVecGam, _  = torch.solve(rightVec.reshape(-1, 1), self.leftMat.to_dense()) 
        self.newVecGam = self.newVecGam.reshape(-1)
        
    def updateVecGamConGra(self):
        """
            Update the Gamma matrix, first step, wth Conjugate Gradient Method
        """
        optAxis = 1 if self.D == self.dF else 0
        
        if self.leftMat is None:
            NewXSq = self.NewXr**2 + self.NewXi**2
            NewXSqR2 = torch.cat((NewXSq, NewXSq), dim=2) # dF x dT x 2R
            NewXSqR2Sum = NewXSqR2.sum(axis=optAxis) # dF x 2R or dT x 2R
            self.leftMatVecP1 = NewXSqR2Sum.flatten()
        
        if self.NewXYR2Sum is None:
            NewXY1 = self.NewXr * self.NewYr + self.NewXi * self.NewYi
            NewXY2 = - self.NewXi * self.NewYr + self.NewXr * self.NewYi
            NewXYR2 = torch.cat((NewXY1, NewXY2), dim=2) # dF x dT x 2R
            self.NewXYR2Sum = NewXYR2.sum(axis=optAxis) # dF x 2R or dT x 2R
        rightVec = self.NewXYR2Sum.flatten() + \
                    DiffMatTOpt(self.rho + self.paras.beta * self.lastTheta, self.R2)
        
        self.newVecGam = self._ConjuGrad(rightVec)
        
    def updateHRho(self):
        """
            Update the vector rho at 1/2 step, second step
        """
        halfRho = self.rho - self.paras.alp * self.paras.beta * (DiffMatOpt(self.newVecGam, self.R2) - self.lastTheta)
        self.halfRho = halfRho
       
    
    def updateTheta(self):
        """
            Update the vector Theta, third step
        """
        halfTheta = DiffMatOpt(self.newVecGam, self.R2) - self.halfRho/self.paras.beta
        tranHTheta = halfTheta.reshape(-1, self.R2) # D-1 x 2R
        hThetaL2Norm = tranHTheta.abs().square().sum(axis=1).sqrt() # D-1
        normCs = torch.zeros_like(hThetaL2Norm) - 1
        
        normC1 = hThetaL2Norm - self.lam/self.paras.beta
        normC1[normC1<0] = 0
        
        normC2 = (self.paras.beta * (self.a - 1) * hThetaL2Norm - self.a * self.lam)/(self.paras.beta * self.a - self.paras.beta -1)
        
        c1 = (1+1/self.paras.beta)* self.lam
        c2 = self.a * self.lam
        
        normCs[hThetaL2Norm<=c1] = normC1[hThetaL2Norm<=c1]
        normCs[hThetaL2Norm>c2] = hThetaL2Norm[hThetaL2Norm>c2]
        normCs[normCs==-1] = normC2[normCs==-1]
        
        normCs[normCs!=0] = normCs[normCs!=0]/hThetaL2Norm[normCs!=0]
        
        self.lastTheta = (tranHTheta*normCs.reshape(-1, 1)).flatten()
        
    def updateThetaGL(self):
        """
            Update the vector Theta, third step with group lasso penalty
        """
        halfTheta = DiffMatOpt(self.newVecGam, self.R2) - self.halfRho/self.paras.beta
        tranHTheta = halfTheta.reshape(-1, self.R2) # D-1 x 2R
        hThetaL2Norm = tranHTheta.abs().square().sum(axis=1).sqrt() # D-1
        
        normC1 = hThetaL2Norm - self.lam
        normC1[normC1<0] = 0
        
        normCs = normC1
        
        normCs[normC1!=0] = normC1[normC1!=0]/hThetaL2Norm[normC1!=0]
        self.lastTheta = (tranHTheta*normCs.reshape(-1, 1)).flatten()
        
    
    def updateRho(self):
        """
            Update the vector rho, fourth step
        """
        newRho = self.halfRho - self.paras.alp * self.paras.beta * (DiffMatOpt(self.newVecGam, self.R2) - self.lastTheta)
        self.rho = newRho
        
    
    def __call__(self, is_showProg=False, leave=False):
        if self.NewXr is None:
            self.obtainNewData()
        if self.iterC is None:
            self.iterC = 0
        
        chDiff = torch.tensor(1e10)
        
        if self.is_ConGrad:
            self.updateVecGamConGra()
        else:
            self.updateVecGam()
            
        self.updateHRho()
        
        if self.penalty.startswith("scad"):
            self.updateTheta()
        elif self.penalty.startswith("group"):
            self.updateThetaGL()
            
        self.updateRho()
        
        lastVecGam = self.newVecGam
        if is_showProg:
            with tqdm(total=self.iterNum, leave=leave) as pbar:
                for i in range(self.iterNum):
                    pbar.set_description(f"Inner Loop: The chdiff is {chDiff.item():.3e}.")
                    pbar.update(1)
                    
                    if self.is_ConGrad:
                        self.updateVecGamConGra()
                    else:
                        self.updateVecGam()
                        
                    self.updateHRho()
                    
                    if self.penalty.startswith("scad"):
                        self.updateTheta()
                    elif self.penalty.startswith("group"):
                        self.updateThetaGL()
                        
                    self.updateRho()
                    chDiff = torch.norm(self.newVecGam-lastVecGam)/torch.norm(lastVecGam)
                    lastVecGam = self.newVecGam
                    if chDiff < self.iterC:
                        pbar.update(self.iterNum)
                        break
        else:
            for i in range(self.iterNum):
                if self.is_ConGrad:
                    self.updateVecGamConGra()
                else:
                    self.updateVecGam()
                    
                self.updateHRho()
                
                if self.penalty.startswith("scad"):
                    self.updateTheta()
                elif self.penalty.startswith("group"):
                    self.updateThetaGL()
                
                self.updateRho()
                chDiff = torch.norm(self.newVecGam-lastVecGam)/torch.norm(lastVecGam)
                lastVecGam = self.newVecGam
                if chDiff < self.iterC:
                    break
            
        self._post()
            
    def _post(self):
        if self.is_std:
            R = int(self.R2/2)
            newGam = self.newVecGam.reshape(-1, self.R2) # D x 2R
            theta = self.lastTheta.reshape(-1, self.R2)# (D-1) x 2R
            
            newGamNorm2 = newGam.square().sum(axis=0) # 2R
            newGamNorm = torch.sqrt(newGamNorm2[:R] + newGamNorm2[R:])
            newGamNorm = torch.cat([newGamNorm, newGamNorm])
            newGam = newGam/newGamNorm
            theta = theta/newGamNorm
            self.newVecGamStd = newGam.flatten()
            self.lastThetaStd = theta.flatten()
            
            self.GamMat = colStackFn(self.newVecGam, self.R2)
            self.GamMatStd = colStackFn(self.newVecGamStd, self.R2)
            self.ThetaMat = colStackFn(self.lastTheta, self.R2)
            self.ThetaMatStd = colStackFn(self.lastThetaStd, self.R2)
        else:
            self.GamMat = colStackFn(self.newVecGam, self.R2)
            self.ThetaMat = colStackFn(self.lastTheta, self.R2)


class TVDNextOpt():
    """
        The class to implement the full procedure of TVDNext method
    """
    def __init__(self, rawDat, fs, T, Rn, hs, **paras):
        """
         Input: 
             rawDat: The raw dataset, tensor of d x dT+1
             fs: The sampling freq of the raw dataset
             T: Time course of the data
             Rn: The nominal rank of A mat, Rn << d to reduce the computational burden 
             hs: the bandwidths for the kernel regression whe estimating A matrix
             paras:
               For Preprocess:
                 is_detrend: Whether detrend the raw data or not
                 bandsCuts: the cirtical freqs to use
                 Nord: The order of the filter
                 q: The decimate rate
                 
               For A matrix:
                 downrates: The downrate factors for freq and time, determine how many A(s_i, t_i) matrix to be summed
                 
               For one-step Opt:
                 betas: list of two tuning parameter for iteration
                 alps: list of two tuning parameter for iteration
                 rhos: list of two vectors of length (dF-1)2R and (dT-1)2R, real data
                 lams: list of two parameters for SCAD, for mu and nu
                 As: list of two parameters for SCAD for mu and nu, > 1+1/beta
                 iterNums:  integer or list of two integers, number of iterations for one-step-opt
                 iterCs: decimal or list of two decimate, stopping rule for one-step-opt
               
               For the outer optimization procedure:
                 paraMuInit: The initial value of mu parameters, along the freq axis
                 paraNuInit: The initial value of nu parameters, along the time axis
                 maxIter: Integer, the maximal times of iteration for the outer loop
                 outIterC:  decimal, stopping rule for the outer loop
        """
        parasDefVs = {
                      "is_detrend": True, "bandsCuts": [[2, 3.5], [4, 7], [8, 12], [13, 30], [30, 80]], 
                      "Nord": None, "q": 10, 
                      "downrates": [1, 10],  "betas":[1, 1], "alps": [0.9, 0.9],  "rhos": None,  "lams": None, 
                      "As": [2.7, 2.7],  "iterNums": [5, 1], "iterCs": None, "paraMuInit": None,
                      "paraNuInit": None, "maxIter": 100, "outIterC": None
                    }
        self.paras = edict(parasDefVs)
        for key in paras.keys():
            self.paras[key] = paras[key]
            
        if self.paras.iterCs is None:
            self.paras.iterCs = [None, None]
        
            
        self.rawDat = rawDat
        self.fs, self.T = fs, T
        self.Rn = Rn
        self.hs = hs 
        
        # Some none definitions
        self.X = self.Y = self.pUinv = None
        self.dF = self.dT = self.D = self.nD = None
        self.lastThetaMu = self.lastTheteNu = None # Vector of 2R(dF-1)/2R(dT-1)
        self.paraMu = self.paraNu = None # matrix of 2R x dF/dT
        self.kpidx = None
        self.R = self.R2 = None
        self.U = None
        
    def _PreProcess(self):
        """
        To preprocess the raw dataset, including 
            1. Detrend, 
            2. Filter under bands
            3. Decimate
        """
        dat = signal.detrend(self.rawDat)
        cDat = mat2Tensor(dat, fs=self.fs, q=self.paras.q)
        # Avoid stride problem when convert numpy to tensor
        self.X = torch.tensor(cDat.X.copy())
        self.Y = torch.tensor(cDat.Y.copy())
        
    def _estAmat(self):
        _, self.dF, self.dT = self.Y.shape
        times = np.linspace(0, self.T, self.dT)
        freqs = np.array([np.mean(bandCut) for bandCut in self.paras.bandsCuts])
        if self.hs is None:
            self.hs = [bw_nrd0_R(freqs), bw_nrd0_R(times)]
        
        self.Amat = GetAmatTorch(self.Y, self.X, times, freqs, self.paras.downrates, self.hs)
        
        res = np.linalg.eig(self.Amat)
        absEigVs = np.abs(res[0])
        # Sort the eigvs and vectors such that the vals is in order
        sortIdx = np.argsort(absEigVs)[::-1]
        lams = res[0][sortIdx]
        U = res[1][:, sortIdx]
        self.U = U
        
        eigF = np.concatenate([[np.inf], lams])
        # To remove conjugate eigvector
        # self.kpidx = np.where(np.diff(np.abs(eigF))[:self.Rn] != 0)[0] 
        # Only for test
        self.kpidx = np.arange(self.Rn) 
        self.R = len(self.kpidx)
        self.R2 = 2 * self.R
        
        Uinv = np.linalg.inv(U)
        pUinv = Uinv[self.kpidx, :]
        self.pUinv = torch.tensor(pUinv)
        
    def __call__(self, showSubProg=False, **runParas):
        for key in runParas:
            self.paras[key] = runParas[key]
            
        if self.X is None:
            self._PreProcess()
        if self.pUinv is None:
            self._estAmat()
        
        _, self.dF, self.dT = self.X.shape
        
        self.D = self.dF
        self.nD = int(self.dF*self.dT/self.D)
        if self.paras.paraMuInit is None:
            self.paras.paraMuInit = torch.rand(self.R2, self.dF)
        if self.paras.paraNuInit is None:
            self.paras.paraNuInit = torch.rand(self.R2, self.dT)
        if self.paras.rhos is None:
            rho1 = torch.ones(self.R2*(self.dF-1))
            rho2 = torch.ones(self.R2*(self.dT-1))
            self.paras.rhos = [rho1, rho2]
            
        
        
        chDiffBoth = torch.tensor(1e10) # Stopping rule
        
        lastMuTheta = DiffMatOpt(colStackFn(self.paras.paraMuInit), self.R2)
        fixedNuMat = self.paras.paraNuInit
        
        stopLastMuMat = self.paras.paraMuInit
        stopLastNuMat = self.paras.paraNuInit
        
        pbar = tqdm(range(self.paras.maxIter))
        for i in pbar:
            if i == 0:
                pbar.set_description(f"Outer Loop: The chdiff is {chDiffBoth.item():.3e}.")
            else:
                pbar.set_description(f"Outer Loop:"
                                     f"{chDiffMu.item():.3e}, "
                                     f"{chDiffNu.item():.3e}, "
                                     f"{chDiffBoth.item():.3e}.")
            optMu = OneStepOpt(X=self.X, Y=self.Y, pUinv=self.pUinv, fixedParas=fixedNuMat, lastTheta=lastMuTheta, 
                               alp=self.paras.alps[0], beta=self.paras.betas[0], lam=self.paras.lams[0], 
                               a=self.paras.As[0], iterNum=self.paras.iterNums[0], rho=self.paras.rhos[0], iterC=self.paras.iterCs[0])
            optMu(showSubProg)
            
            fixedMuMat = optMu.GamMatStd
            if i == 0:
                lastNuTheta = DiffMatOpt(colStackFn(fixedNuMat), self.R2)
            else:
                lastNuTheta = optNu.lastTheta
            
            optNu = OneStepOpt(X=self.X, Y=self.Y, pUinv=self.pUinv, fixedParas=fixedMuMat, lastTheta=lastNuTheta, 
                               alp=self.paras.alps[1], beta=self.paras.betas[1], lam=self.paras.lams[1], 
                               a=self.paras.As[1], iterNum=self.paras.iterNums[1], rho=self.paras.rhos[1], iterC=self.paras.iterCs[1])
            optNu(showSubProg)
            
            fixedNuMat = optNu.GamMat
            # lastMuTheta = DiffMatOpt(colStackFn(fixedMuMat), self.R2)
            lastMuTheta = optMu.lastThetaStd
            
            chDiffMu = torch.norm(stopLastMuMat-fixedMuMat)/torch.norm(stopLastMuMat)
            chDiffNu = torch.norm(stopLastNuMat-fixedNuMat)/torch.norm(stopLastNuMat)
            chDiffBoth = torch.max(chDiffMu,chDiffNu)
            
            stopLastMuMat = fixedMuMat
            stopLastNuMat = fixedNuMat
            # if show_prog:
            #     if (i+1) % thin == 1:
            #         print(f"Current iteration is {i+1}/{self.paras.maxIter}, the change of diff is {chDiffBoth}")
            if chDiffBoth <= self.paras.outIterC:
                break
            
        self.paraMu = stopLastMuMat
        self.paraNu = stopLastNuMat
        self.lastOptMu = optMu
        self.lastOptNu = optNu


# ### Misc

# +
def genDiffMatfn(R, D):
    """
    Generate the Matrix for different of parameters, D_\mu and D_\nu, RD-R x RD
    """
    mat = torch.zeros(R*D-R, R*D, dtype=torch.int8)
    mat[:, R:] += torch.eye(R*D-R, dtype=torch.int8)
    mat[:, :-R] += -torch.eye(R*D-R, dtype=torch.int8)
    return mat.to_sparse()

def DiffMatOpt(Vec, R):
    """
    D_\mu operator, i.e. output is D_\mu Vec
    """ 
    VecDiff = Vec[R:] - Vec[:-R]
    return VecDiff

def DiffMatTOpt(Vec, R):
    """
    D_\mu\trans operator, i.e. output is D_\mu\trans Vec
    """ 
    outVec = torch.zeros(Vec.shape[0]+R)
    outVec[R:-R] = Vec[:-R] - Vec[R:]
    outVec[:R] = -Vec[:R]
    outVec[-R:] = Vec[-R:]
    return outVec
    

def genDiffMatSqfn(R, D):
    """
    Generate the squared Matrix for different of parameters, D_\mu \trans D_\nm, RD x RD
    """
    mVec = torch.ones(R*D, dtype=torch.int8)
    mVec[R:-R] += torch.ones(R*(D-2), dtype=torch.int8)
    mat = torch.diag(mVec)
    mat[R:, :-R] += -torch.eye(R*(D-1), dtype=torch.int8)
    mat[:-R, R:] += -torch.eye(R*(D-1), dtype=torch.int8)
    return mat.to_sparse()

def genIdenMatfn(s, R, D):
    """
    Generate the (R x RD) Matrix with only s^th (from 1) block is identity matrix 
    s: The index is from 1 not 0
    """
    assert s <= D, "Parameter s is to large!"
    s = s - 1
    mat = torch.zeros(R, R*D, dtype=torch.int8)
    mat[:, R*s:(R*s+R)] += torch.eye(R, dtype=torch.int8)
    return mat.to_sparse().double()

def colStackFn(a, R=None):
    """
    transform between R x D matrix and RD vector where the elements are stacked by columns
    
    args:
        a: The target to do transfrom
        R: num of rows
    """
    
    if a.dim() == 2:
        out = a.permute((1, 0)).reshape(-1)
    elif a.dim() == 1:
        out = a.reshape(-1, R).permute((1, 0))
    return out
# -


