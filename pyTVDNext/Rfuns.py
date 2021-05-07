# functions  call from  R language

import rpy2.robjects as robj
import numpy as np

def bw_nrd0_R(time, fct=1):
    bw_nrd0 = robj.r["bw.nrd0"]
    time_r = robj.FloatVector(time)
    return np.array(bw_nrd0(time_r))[0]*fct

