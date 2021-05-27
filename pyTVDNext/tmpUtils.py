import numpy as np


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

