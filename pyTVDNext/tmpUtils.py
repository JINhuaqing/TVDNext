import numpy as np
from easydict import EasyDict as edict

timeLims = edict()
timeLims.st02 = [35, 95]
timeLims.st03 = [20, 80]


# +
def supInfDist(set1, set2):
    if len(set2) == 0:
        dist = 0
    elif len(set1) == 0:
        dist = np.max(set2)
    else:
        set1 = np.array(set1)
        set2 = np.array(set2)
        dist = np.abs(set1 - set2.reshape(-1, 1)).min(axis=1).max()
    return dist

# Compute the Hausdorff distance between two change point sets
def hdist(set1, set2):
    dist1 = supInfDist(set1, set2)
    dist2 = supInfDist(set2, set1)
    return np.max((dist1, dist2))

# load the gt for MEG--Eye data
def txt2Time(txtF):   
    with open(txtF, "r") as f:
        data = f.readlines() 
    data = data[1:]
    data = [i.strip().split("(") for i in data]
    data = [float(i[0]) for i in data if len(i)>1]
    return data

# Time to change points
def time2pts(ts, lims, Up=7200):
    ts = np.array(ts)
    timeC = 60
    ts = ts[ts>=lims[0]]
    ts = ts[ts<=lims[1]]
    ts = ts - lims[0]
    cpts = ts*Up/timeC
    cpts = cpts.astype(np.int)
    
    res = edict()
    res.ts = ts
    res.cpts = cpts
    return res


# +
# Make sure the generated seq is in decreasing order w.r.t mode 
def GenCVec(nR, eigInd):
    eigInd = np.array(eigInd)
    imgPart = np.sort(np.random.randint(1, 10, nR))[::-1]
    imgPart[eigInd==False] = 0
    imgPartSub = imgPart[eigInd]
    imgPartSub[1::2] = - imgPartSub[0::2]
    imgPart[eigInd] = imgPartSub
    
    realPart = np.sort(np.random.randint(1, 10, nR))[::-1]
    realPart[eigInd==False] += 10
    realPartSub = realPart[eigInd]
    realPartSub[1::2] = realPartSub[0::2]
    realPart[eigInd] = realPartSub
    
    cVec = realPart.astype(np.complex)
    cVec.imag = imgPart
    return cVec

# def GenCVec(nR, eigInd):
#     eigInd = np.array(eigInd)
#     imgPart = np.random.randint(1, 10, nR)
#     imgPart[eigInd==False] = 0
#     imgPartSub = imgPart[eigInd]
#     imgPartSub[1::2] = - imgPartSub[0::2]
#     imgPart[eigInd] = imgPartSub
#     
#     realPart = np.random.randint(1, 10, nR)
#     realPartSub = realPart[eigInd]
#     realPartSub[1::2] = realPartSub[0::2]
#     realPart[eigInd] = realPartSub
#     
#     cVec = realPart.astype(np.complex)
#     cVec.imag = imgPart
#     return cVec

def GenCVecs(nR, eigInd, num):
    Vecs = []
    for i in range(num):
        Vecs.append(GenCVec(nR, eigInd))
    return np.array(Vecs).T


def GenFVecs(nR, eigInd, ChgsF):
    num = len(ChgsF) - 1
    cVecs = GenCVecs(nR, eigInd, num)
    numSegs = np.diff(ChgsF)
    
    fVecsList = []
    for idx, numSeg in enumerate(numSegs):
        fVecsList = fVecsList + list(cVecs[:, idx]) * numSeg
    fVecs = np.array(fVecsList)
    return fVecs.reshape(-1, nR).T
    


# -
def matchU(eU, U):
    nrow, p = eU.shape
    absU = np.abs(U)
    ordList = []
    kpIdxs = list(range(nrow))
    
    for i in range(nrow):
        cVec = np.abs(eU[i, :])
        cDiff = cVec.reshape(-1, p) - absU[np.array(kpIdxs), :]
        cIdx = np.array(kpIdxs)[np.argmin((cDiff**2).sum(axis=1))]
        ordList.append(cIdx)
        kpIdxs.remove(cIdx)
    return np.array(ordList)

