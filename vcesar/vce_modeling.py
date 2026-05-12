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

from pykrige import OrdinaryKriging
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

from icams import _utils as ut

from scipy.optimize import least_squares
from scipy.optimize import leastsq
from scipy.stats.stats import pearsonr

from pykrige import variogram_models

#import matlab.engine

###### mintPy modelu ###############################


variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model,
                      'hole-effect': variogram_models.hole_effect_variogram_model}

def calculate_variogram_model(lags, semivariance, variogram_model,
                               variogram_function, weight):
    """Function that fits a variogram model when parameters are not specified.
    Returns variogram model parameters that minimize the RMSE between the
    specified variogram function and the actual calculated variogram points.

    Parameters
    ----------
    lags: 1d array
        binned lags/distances to use for variogram model parameter estimation
    semivariance: 1d array
        binned/averaged experimental semivariances to use for variogram model
        parameter estimation
    variogram_model: str/unicode
        specified variogram model to use for parameter estimation
    variogram_function: callable
        the actual funtion that evaluates the model variogram
    weight: bool
        flag for implementing the crude weighting routine, used in order to fit
        smaller lags better this is passed on to the residual calculation
        cfunction, where weighting is actually applied...

    Returns
    -------
    res: list
        list of estimated variogram model parameters

    NOTE that the estimation routine works in terms of the partial sill
    (psill = sill - nugget) -- setting bounds such that psill > 0 ensures that
    the sill will always be greater than the nugget...
    """

    if variogram_model == 'linear':
        x0 = [(np.amax(semivariance) - np.amin(semivariance)) /
              (np.amax(lags) - np.amin(lags)), np.amin(semivariance)]
        bnds = ([0., 0.], [np.inf, np.amax(semivariance)])
    elif variogram_model == 'power':
        x0 = [(np.amax(semivariance) - np.amin(semivariance)) /
              (np.amax(lags) - np.amin(lags)), 1.1, np.amin(semivariance)]
        bnds = ([0., 0.001, 0.], [np.inf, 1.999, np.amax(semivariance)])
    else:
        x0 = [np.amax(semivariance) - np.amin(semivariance),
              0.25*np.amax(lags), np.amin(semivariance)]
        bnds = ([0., 0., 0.], [10.*np.amax(semivariance), np.amax(lags),
                               np.amax(semivariance)])

    # use 'soft' L1-norm minimization in order to buffer against
    # potential outliers (weird/skewed points)
    res = least_squares(variogram_residuals, x0, bounds=bnds, loss='soft_l1',
                        args=(lags, semivariance, variogram_function, weight))

    return res.x

def variogram_residuals(params, x, y, variogram_function, weight):
    """Function used in variogram model estimation. Returns residuals between
    calculated variogram and actual data (lags/semivariance).
    Called by _calculate_variogram_model.

    Parameters
    ----------
    params: list or 1D array
        parameters for calculating the model variogram
    x: ndarray
        lags (distances) at which to evaluate the model variogram
    y: ndarray
        experimental semivariances at the specified lags
    variogram_function: callable
        the actual funtion that evaluates the model variogram
    weight: bool
        flag for implementing the crude weighting routine, used in order to
        fit smaller lags better

    Returns
    -------
    resid: 1d array
        residuals, dimension same as y
    """

    # this crude weighting routine can be used to better fit the model
    # variogram to the experimental variogram at smaller lags...
    # the weights are calculated from a logistic function, so weights at small
    # lags are ~1 and weights at the longest lags are ~0;
    # the center of the logistic weighting is hard-coded to be at 70% of the
    # distance from the shortest lag to the largest lag
    if weight:
        drange = np.amax(x) - np.amin(x)
        k = 2.1972 / (0.1 * drange)
        x0 = 0.7 * drange + np.amin(x)
        weights = 1. / (1. + np.exp(-k * (x0 - x)))
        weights /= np.sum(weights)
        resid = (variogram_function(params, x) - y) * weights
    else:
        resid = variogram_function(params, x) - y

    return resid

def read_attr(fname):
    # read hdf5
    with h5py.File(fname, 'r') as f:
        atr = dict(f.attrs)
        
    return atr

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
#########################################################################

INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019-2026, Yunmeng Cao   @vceSAR v2.0
   
   Modeling variance/structure functions of tropospheric delay in SAR images
'''

EXAMPLE = '''
    Usage:
            vce_modeling.py variogramTs.h5 
            vce_modeling.py variogramStack.h5 --model gaussian
            vce_modeling.py variogramTs.h5 --max-length 150 --model spherical

##############################################################################
'''


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Check common busrts for TOPS data.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('input_file',help='input file name (variogramStack.h5 or variogramTs.h5).')
    parser.add_argument('-m','--model', dest='model', default='spherical',
                      help='variogram model used to fit the variance samples')
    parser.add_argument('--max-length', dest='max_length',type=float, metavar='NUM',
                      help='used bin ratio for mdeling the structure model.')
    parser.add_argument('-o','--out_file', dest='out_file', metavar='FILE',
                      help='name of the output file')

    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 
    FILE = inps.input_file
    
    meta = read_attr(FILE)
    meta['model'] = inps.model
    date = ut.read_hdf5(FILE, datasetName='date')[0]
    variance_insar = ut.read_hdf5(FILE, datasetName='semivariance')[0]
    variance_insar_std = ut.read_hdf5(FILE, datasetName='semivariance_std')[0]
    Lags = ut.read_hdf5(FILE, datasetName='Lags')[0]
    
    lag = Lags
    if inps.max_length:
        max_lag = inps.max_length
    else:
        max_lag = max(lag) + 0.001
    meta['max_length'] = max_lag
    r0 = np.asarray(1/2*max_lag)
    range0 = r0.tolist()
    #r0 = np.asarray(r0,dtype=float)
    #range00 = matlab.double(range0)
    #range0 = r0
    LL0 = lag[lag < max_lag]
    
    datasetDict = dict()
    datasetDict['Lags'] = Lags
    
    if inps.out_file:
        OUT = os.path.out_file
    elif 'Ts' in os.path.basename(FILE):
        OUT = 'variogramTsModel.h5'
    else:
        OUT = 'variogramStackModel.h5'
    
    def resi_func(m,d,y):
        variogram_function =variogram_dict[inps.model] 
        return  y - variogram_function(m,d)
    
    if 'Ts' in os.path.basename(FILE):
        date_list = ut.read_hdf5(FILE, datasetName='date_list')[0]
        date = ut.read_hdf5(FILE, datasetName='date')[0]
        variance_sar = ut.read_hdf5(FILE, datasetName='semivarianceTs')[0]
        variance_sar_std = ut.read_hdf5(FILE, datasetName='semivarianceTs_std')[0]
        
        variance_sar_weight = ut.read_hdf5(FILE, datasetName='semivarianceTs_weight')[0]
        variance_sar_std_weight = ut.read_hdf5(FILE, datasetName='semivarianceTs_weight_std')[0]
        
        row,col = variance_sar.shape
        model_parameters = np.zeros((row,4),dtype='float32')   # sill, range, nugget, Rs
        model_parameters_weight = np.zeros((row,4),dtype='float32')
        
        variogram_function = variogram_dict[inps.model]
        
        for i in range(row):
            S0 = variance_sar[i,:]
            SS0 = S0[lag < max_lag]
            sill0 = max(SS0)
            sill0 = sill0.tolist()
            
           
            p0 = [sill0, range0, 0.0001]   
            SS01 = np.abs(SS0.copy())
            LL01 = LL0.copy()
            
            #print(SS01)
            #print(LL01)  
            #tt = calculate_variogram_model(LL01, SS01, inps.model,variogram_function, True)
            #corr, _ = pearsonr(SS0, variogram_function(tt,LL0))
            #model_parameters[i,0:3] = tt
            #model_parameters[i,3] = corr
            
            
            S0 = variance_sar_weight[i,:]
            SS0 = S0[lag < max_lag]
            sill0 = max(SS0)
            sill0 = sill0.tolist()
            
            
            SS01 = np.abs(SS0.copy())
            LL01 = LL0.copy()
            print(SS01)            

            tt = calculate_variogram_model(LL01, SS01, inps.model,variogram_function, True)
            corr, _ = pearsonr(SS0, variogram_function(tt,LL0))
            model_parameters_weight[i,0:3] = tt
            model_parameters_weight[i,3] = corr
            print(tt)
            print(corr)
                    
        #datasetDict['model_parameters_noweight'] = model_parameters 
        datasetDict['model_parameters'] = model_parameters_weight
        datasetDict['semivarianceTs'] = variance_sar
        datasetDict['semivarianceTs_std'] = variance_sar_std
        datasetDict['semivarianceTs_weight_std'] = variance_sar_std_weight
        datasetDict['semivarianceTs_weight'] = variance_sar_weight
        datasetDict['semivariance'] = variance_insar
        datasetDict['semivariance_std'] = variance_insar_std
        datasetDict['date_list'] = date_list
        datasetDict['date'] = date
    else:
        date_list = readfile.read(FILE, datasetName='date_list')[0]
        date = readfile.read(FILE, datasetName='date')[0]
        row,col = variance_insar.shape
        model_parameters = np.zeros((row,4),dtype='float32')   # sill, range, nugget, Rs
        for i in range(row):
            S0 = variance_insar[i,:]
            SS0 = S0[lag < max_lag]
            sill0 = max(SS0)
            sill0 = sill0.tolist()
            
            p0 = [sill0, range0, 0.0001]   
            SS01 = SS0.copy()
            LL01 = LL0.copy()
            
            tt = calculate_variogram_model(LL01, SS01, inps.model,variogram_function, True)
            corr, _ = pearsonr(SS0, variogram_function(tt,LL0))

            model_parameters[i,0:3] = tt
            model_parameters[i,3] = corr
            
            
        datasetDict['model_parameters'] = model_parameters  
        datasetDict['semivariance'] = variance_insar
        datasetDict['semivariance_std'] = variance_insar_std
        datasetDict['date_list'] = date_list
        datasetDict['date'] = date
    #eng.quit()
    write_variogram_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None)
    
    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[:])
