#! /usr/bin/env python
#################################################################
###  This program is part of vceSAR  v2.0                     ### 
###  Copy Right (c): 2019-2026, Yunmeng Cao                   ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com & ym.cao@csu.edu.cn             ###
###  Univ. : KAUST & GNS & CSU                                ###   
#################################################################

import numpy as np
import os
import sys  
import subprocess
import getopt
import time
import glob
import argparse
from scipy import linalg 
import h5py
from icams import _utils as ut
###### mintPy modelu ###############################
from mintpy.objects import (stack,
                            stackDict)
from mintpy.objects import gnss
from mintpy.utils import (readfile,
                          plot as pp)
from mintpy.multilook import multilook_data
from mintpy import subset, version
#######################################################

def estimate_timeseries(A, ifgram, weight_sqrt=None, rcond=1e-5):

    ifgram = ifgram.reshape(A.shape[0], -1)
    if weight_sqrt is not None:
        weight_sqrt = weight_sqrt.reshape(A.shape[0], -1)
    else:
        weight_sqrt = np.ones((A.shape[0],1))
        weight_sqrt = weight_sqrt.reshape(A.shape[0], -1)
    
    num_date = A.shape[1]
    num_pixel = ifgram.shape[1]

    # Initial output value
    ts = np.zeros((num_date, num_pixel), np.float32)
    temp_coh = 0.
    num_inv_ifg = 0

    #if weight_sqrt is not None:
    
    A_w = np.multiply(A, weight_sqrt)
    ifgram_w = np.multiply(ifgram, weight_sqrt)
    X = linalg.lstsq(A_w, ifgram_w, cond=rcond)[0]
    
    #else:
    #    X = linalg.lstsq(A, ifgram, cond=rcond)[0]
    
    ts = X
    ifgram_diff = ifgram - np.dot(A, X)
    
    # calculate temporal coherence
    #num_inv_ifg = A.shape[0]
    #temp_coh = np.abs(np.sum(np.exp(1j*ifgram_diff), axis=0)) / num_inv_ifg
    sigma0 = np.sum(ifgram_w*(ifgram_diff**2))/(num_date-1)
    
    P = np.diag(weight_sqrt.reshape(A.shape[0],))
    A0 = np.dot(np.transpose(A),P)
    QQ = np.linalg.inv(np.dot(A0,A))
    #print(QQ)
    var_ts = np.diag(sigma0*QQ)
    var_ts = var_ts.reshape(A.shape[1], -1)

    return ts, var_ts

def date2list(date12):
    row,col = date12.shape
    date12_list = []
    for i in range(row):
        CC=str(date12[i,0])
        M1 = CC[2:10]
        CC=str(date12[i,1])
        S1 = CC[2:10]
        SS = M1 + '_' + S1
        date12_list.append(SS)
    return date12_list

def get_design_matrix4timeseries(date12):
    """Return design matrix of the input ifgramStack for timeseries estimation
    Parameters: date12_list : list of string in YYYYMMDD_YYYYMMDD format
                refDate : str, date in YYYYMMDD format
    Returns:    A : 2D array of float32 in size of (num_ifgram, num_date-1)
                B : 2D array of float32 in size of (num_ifgram, num_date-1)
    Examples:   obj = ifgramStack('./inputs/ifgramStack.h5')
                A, B = obj.get_design_matrix4timeseries(obj.get_date12_list(dropIfgram=True))
                A = ifgramStack.get_design_matrix4timeseries(date12_list, refDate='20101022')[0]
                A = ifgramStack.get_design_matrix4timeseries(date12_list, refDate=0)[0] #do not omit the 1st column
    """
    # Date info
    date12_list = date2list(date12)
    mDates = [i.split('_')[0] for i in date12_list]
    sDates = [i.split('_')[1] for i in date12_list]
    dateList = sorted(list(set(mDates + sDates)))
    #dates = [dt(*time.strptime(i, "%Y%m%d")[0:5]) for i in dateList]
    #tbase = np.array([(i - dates[0]).days for i in dates], np.float32) / 365.25
    numIfgram = len(date12_list)
    numDate = len(dateList)
    
    # calculate design matrix
    A = np.zeros((numIfgram, numDate), np.float32)
    B = np.zeros(A.shape, np.float32)
    for i in range(numIfgram):
        m_idx, s_idx = [dateList.index(j) for j in date12_list[i].split('_')]
        A[i, m_idx] = -1
        A[i, s_idx] = 1
        
        B[i, m_idx] = 1
        B[i, s_idx] = 1

    # Remove reference date as it can not be resolved
    return A, B, numDate
 
def write_variogram_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):
    #output = 'variogramStack.h5'
    'lags                  1 x N '
    'semivariance          M x N '
    'sills                 M x 1 '
    'ranges                M x 1 '
    'nuggets               M x 1 '
    
    if os.path.isfile(out_file):
        print('delete exsited file: {}'.format(out_file))
        os.remove(out_file)

    print('create HDF5 file: {} with w mode'.format(out_file))
    with h5py.File(out_file, 'w') as f:
        for dsName in datasetDict.keys():
            data = datasetDict[dsName]
            ds = f.create_dataset(dsName,
                              data=data,
                              compression=compression)
        
        for key, value in metadata.items():
            f.attrs[key] = str(value)
            #print(key + ': ' +  value)
    print('finished writing to {}'.format(out_file))
        
    return out_file 

####################################################

INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019-2026, Yunmeng Cao   @vceSAR v2.0
   
   Estimte the variance components of the tropospheric delay in SAR images
'''

EXAMPLE = '''
    Usage:
            vce_sar.py variogramStack.h5 --weight
            
    Examples:
            vce_sar.py variogramStack.h5 
##############################################################################
'''


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Estimate the non-differential atmospheric variance.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('input_file',help='input file name (ifgramStack.h5 or timeseires.h5).')
    parser.add_argument('-o','--out_file', dest='out_file', metavar='FILE',
                      help='name of the output file')

    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 
    FILE = inps.input_file
    
    if inps.out_file:
        OUT = inps.out_file
    else:
        OUT = 'variogramTs.h5'
    
    date_list = ut.read_hdf5(FILE, datasetName='date_list')[0]
    date12 = ut.read_hdf5(FILE, datasetName='date')[0]
    bperp = ut.read_hdf5(FILE, datasetName='bperp')[0]
    meta = readfile.read_attribute(FILE, datasetName=None) 
    lags = ut.read_hdf5(FILE, datasetName='Lags')[0]
    
    semivariance = ut.read_hdf5(FILE, datasetName='semivariance')[0]
    semivariance_std = ut.read_hdf5(FILE, datasetName='semivariance_std')[0]
    A,B, numbDate = get_design_matrix4timeseries(date12)
    row,col = semivariance.shape
    
    semivarianceTs = np.zeros((numbDate,col))
    semivarianceTs_std = np.zeros((numbDate,col))
    
    semivarianceTs0 = np.zeros((numbDate,col))
    semivarianceTs_std0 = np.zeros((numbDate,col))
    
    for i in range(col):
        y0 = semivariance[:,i]
        weight0 = semivariance_std[:,i]
        weight1 = 1/weight0
        ts, var_ts = estimate_timeseries(B, y0, weight_sqrt=weight1, rcond=1e-5)
        ts0, var_ts0 = estimate_timeseries(B, y0, weight_sqrt=None, rcond=1e-5)
        ts = ts.reshape(numbDate,)
        var_ts =var_ts.reshape(numbDate,)
        semivarianceTs[:,i] = ts
        semivarianceTs_std[:,i] =np.sqrt(var_ts)
        
        ts0 = ts0.reshape(numbDate,)
        var_ts0 =var_ts0.reshape(numbDate,)
        semivarianceTs0[:,i] = ts0
        semivarianceTs_std0[:,i] =np.sqrt(var_ts0)
        

    
    datasetDict = dict()
    datasetDict['Lags']=np.float32(lags)
    datasetDict['semivariance']=np.float32(semivariance)
    datasetDict['semivariance_std']=np.float32(semivariance_std)
    datasetDict['semivarianceTs_weight']=np.float32(semivarianceTs)    
    datasetDict['semivarianceTs_weight_std']=np.float32(semivarianceTs_std)
    datasetDict['semivarianceTs']=np.float32(semivarianceTs0)
    datasetDict['semivarianceTs_std']=np.float32(semivarianceTs_std0)
    datasetDict['date']=date12
    datasetDict['date_list']=date_list
    datasetDict['bperp']=bperp
    
    write_variogram_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None)  
    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[:])
