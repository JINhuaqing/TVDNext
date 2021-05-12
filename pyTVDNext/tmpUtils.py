import numpy as np


# +
def GenCVec(nR, eigInd):
    eigInd = np.array(eigInd)
    imgPart = np.random.randint(1, 10, nR)
    imgPart[eigInd==False] = 0
    imgPartSub = imgPart[eigInd]
    imgPartSub[1::2] = - imgPartSub[0::2]
    imgPart[eigInd] = imgPartSub
    
    realPart = np.random.randint(1, 10, nR)
    realPartSub = realPart[eigInd]
    realPartSub[1::2] = realPartSub[0::2]
    realPart[eigInd] = realPartSub
    
    cVec = realPart.astype(np.complex)
    cVec.imag = imgPart
    return cVec

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


