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
import random
import pykrige
from pykrige import OrdinaryKriging
from pykrige import variogram_models


#######################################################
variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model,
                      'hole-effect': variogram_models.hole_effect_variogram_model}

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

def distance_matrix(width,length,ref_x,ref_y):

    xv,yv = np.meshgrid(range(0,int(width)),range(0,int(length)))    
    distance_matrix = np.sqrt((xv - int(ref_x))**2 + (yv - int(ref_y))**2)
    
    return distance_matrix
    
    
#########################################################################

INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019-2026, Yunmeng Cao   @vceSAR v2.0
   
   Calculate spatio-temporal variance components of time-series InSAR deformation
'''

EXAMPLE = '''
    Usage:
            vce_tsinsar.py variogramTsModel.h5 
            vce_tsinsar.py variogramTsModel.h5 -o variogramTsWeight.h5 


##############################################################################
'''

def cmdLineParse():
    parser = argparse.ArgumentParser(description='Check common busrts for TOPS data.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('variogramTsModel',help='WNVCE/NVCE based variogram models of time-series SAR observations.')
    parser.add_argument('-o','--out_file', dest='out_file', metavar='FILE',help='name of the output file')

    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 
    FILE = inps.variogramTsModel
    
    if inps.out_file:
        OUT = inps.out_file
    else:
        OUT = 'timeseriesVariance.h5'
    
    meta = ut.read_attr(FILE)
    model = meta['model']
    date_list = ut.read_hdf5(FILE, datasetName='date_list')[0]
    model_parameters = ut.read_hdf5(FILE, datasetName='model_parameters')[0]
    width = meta['WIDTH']
    length = meta['LENGTH']
    ref_x = meta['REF_X']
    ref_y = meta['REF_Y']
    Resolution = float(meta['AZIMUTH_PIXEL_SIZE'])/1000.0
    
    dm = distance_matrix(width,length,ref_x,ref_y)
    dm = dm*Resolution
    variogram_function = variogram_dict[meta['model']]
    row,col = model_parameters.shape
    vcm = np.zeros((row,int(length),int(width)))
    
    for i in range(row):
        ps = model_parameters[i,0:3]
        if ps[0] < 0.5:
            ps[0] = 0.5 
        vcm0 = variogram_function(ps,dm)
        #print(vcm0)
        vcm[i,:,:] = vcm0/100 #change to m
    
    meta['UNIT'] = 'm'
    datasetDict = dict()
    datasetDict['timeseries'] = vcm
    datasetDict['date'] = date_list
    write_variogram_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None)
    
    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[:])
