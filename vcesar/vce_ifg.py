#! /usr/bin/env python
#################################################################
###  This program is part of vceSAR  v2.0                     ### 
###  Copy Right (c): 2019-2026, Yunmeng Cao                   ###  
###  Author: Yunmeng Cao                                      ###                                                          
###  Email : ymcmrs@gmail.com & ym.cao@csu.edu.cn             ###
###  Univ. : KAUST & GNS & CSU                                ###   
#################################################################

from numpy import *
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

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from icams import _utils as ut

###### mintPy modelu ###############################
from mintpy.objects import (stack,
                            stackDict)
from mintpy.objects import gnss
from mintpy.utils import (ptime,
                          readfile,
                          plot as pp)
from mintpy.multilook import multilook_data
from mintpy import subset, version
from vcesar import variance
#######################################################

def check_variable_name(path):
    s=path.split("/")[0]
    if len(s)>0 and s[0]=="$":
        p0=os.getenv(s[1:])
        path=path.replace(path.split("/")[0],p0)
    return path


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_lat_lon(meta):
    """extract lat/lon info of all grids into 2D matrix.
    For meta dict in geo-coordinates only.
    Returned lat/lon are corresponds to the pixel center
    Parameters: meta : dict, including LENGTH, WIDTH and Y/X_FIRST/STEP
                box  : 4-tuple of int for (x0, y0, x1, y1)
    Returns:    lats : 2D np.array for latitude  in size of (length, width)
                lons : 2D np.array for longitude in size of (length, width)
    """
    length, width = int(meta['LENGTH']), int(meta['WIDTH'])
    box = (0, 0, width, length)
    lat_num = box[3] - box[1]
    lon_num = box[2] - box[0]

    # generate 2D matrix for lat/lon
    lat_step = float(meta['Y_STEP'])
    lon_step = float(meta['X_STEP'])
    lat0 = float(meta['Y_FIRST']) + lat_step * (box[1])
    lon0 = float(meta['X_FIRST']) + lon_step * (box[0])
    lat1 = lat0 + lat_step * (lat_num - 1)
    lon1 = lon0 + lon_step * (lon_num - 1)
    lats, lons = np.mgrid[lat0:lat1:lat_num*1j,
                          lon0:lon1:lon_num*1j]

    lats = np.array(lats, dtype=np.float32)
    lons = np.array(lons, dtype=np.float32)
    lats = lats.flatten()
    lons = lons.flatten()
    
    return lats, lons

def get_xx_yy(atr):
    '''Get lat/lon of all pixels'''
    length = int(atr['LENGTH'])
    width = int(atr['WIDTH'])

    yy, xx = np.mgrid[range(length), range(width)]
    yy = yy.flatten()   # azimuth
    xx = xx.flatten()   # range
    return yy, xx 
    
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

def ok_variogram(data0):
    x_sample, y_sample, z_sample, MODEL, coord_type,BIN_NUMB, Unit, Radius, Resolution, Wavelength = data0
    uk = OrdinaryKriging(x_sample, y_sample, z_sample, variogram_model = MODEL, coordinates_type = coord_type, nlags=BIN_NUMB)
    Lags = uk.lags
    #print(Lags)
    Semivariance = 2*(uk.semivariance)
    #print(Semivariance)
    Model_parameters = uk.variogram_model_parameters
    STDs = 2*(uk.semivariance_std)
    #print(STDs)
    #print(Semivariance)
    #print(STDs)
    
    if coord_type == 'geographic':
        Lags = Lags/180*np.pi*Radius   # Radius_earth 
        Model_parameters[1] = Model_parameters[1]/180*np.pi*Radius
    else:
        Lags = Lags*Resolution
        Model_parameters[1] = Model_parameters[1]*Resolution
    
    if Unit == 'radian':
        Semivariance = Semivariance*((Wavelength/(4*np.pi))**2)
        Model_parameters[0] = Model_parameters[0]*((Wavelength/(4*np.pi))**2)
        Model_parameters[2] = Model_parameters[2]*((Wavelength/(4*np.pi))**2)
    elif Unit == 'm':
        Semivariance = Semivariance*10000
        Model_parameters[0] = Model_parameters[0]*10000
        Model_parameters[2] = Model_parameters[2]*10000
    elif Unit == 'dm':
        Semivariance = Semivariance*100
        Model_parameters[0] = Model_parameters[0]*100
        Model_parameters[2] = Model_parameters[2]*100
    elif Unit == 'cm':
        Semivariance = Semivariance
        Model_parameters[0] = Model_parameters[0]
        Model_parameters[2] = Model_parameters[2]

    Sill = Model_parameters[0]
    Range = Model_parameters[1]
    Nugget = Model_parameters[2]
    #print(Model_parameters)
    #print(Lags)
    #print(Semivariance)
    #print(Range)    
    return Semivariance,Lags,STDs,Sill,Range,Nugget

def parallel_process(array, function, n_jobs=16, use_kwargs=False):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return [function(**a) if use_kwargs else function(a) for a in tqdm(array[:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[:]]
        else:
            futures = [pool.submit(function, a) for a in array[:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return out    


def print_progress(iteration, total, prefix='calculating:', suffix='complete', decimals=1, barLength=50, elapsed_time=None):
    """Print iterations progress - Greenstick from Stack Overflow
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int) 
        barLength   - Optional  : character length of bar (Int) 
        elapsed_time- Optional  : elapsed time in seconds (Int/Float)
    
    Reference: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    if elapsed_time:
        sys.stdout.write('%s [%s] %s%s    %s    %s secs\r' % (prefix, bar, percents, '%', suffix, int(elapsed_time)))
    else:
        sys.stdout.write('%s [%s] %s%s    %s\r' % (prefix, bar, percents, '%', suffix))
    sys.stdout.flush()
    if iteration == total:
        print("\n")

    '''
    Sample Useage:
    for i in range(len(dateList)):
        print_progress(i+1,len(dateList))
    '''
    return

#########################################################################

INTRODUCTION = '''
#############################################################################
   Copy Right(c): 2019-2026, Yunmeng Cao   @vceSAR v2.0
   
   Estimate the variance components of the tropospheric delay in interferograms
'''

EXAMPLE = '''
    Usage:
            vce_ifg.py ifgramStack.h5 --parallel 4
            vce_ifg.py ifgramStack.h5 -m maskTempCoh.h5 --sample_numb 5000
            vce_ifg.py ifgramStack.h5 -m maskTempCoh.h5 --bin_numb 30
            
            vce_ifg.py timeseries.h5 --parallel 8
            vce_ifg.py timeseries.h5 --bin_numb 20
            vce_ifg.py timeseries.h5 -m maskTempCoh.h5 --bin_numb 30
##############################################################################
'''


def cmdLineParse():
    parser = argparse.ArgumentParser(description='Check common busrts for TOPS data.',\
                                     formatter_class=argparse.RawTextHelpFormatter,\
                                     epilog=INTRODUCTION+'\n'+EXAMPLE)

    parser.add_argument('input_file',help='input file name (ifgramStack.h5 or timeseires.h5).')
    parser.add_argument('-m','--mask', dest='mask_file', metavar='FILE',
                      help='mask file for masking those large deforming or low-coherence pixels')
    parser.add_argument('-o','--out_file', dest='out_file', metavar='FILE',
                      help='name of the output file')
    parser.add_argument('--variogram_model', dest='variogram_model', default='spherical',
                      help='variogram model used to fit the variance samples')
    parser.add_argument('--sample_numb', dest='sample_numb',type=int,default=3000,metavar='NUM',
                      help='number of samples used to calculate the variance sample')
    parser.add_argument('--bin_numb', dest='bin_numb',type=int,default=30, metavar='NUM',
                      help='number of bins used to fit the variogram model')
    parser.add_argument('--used_bin_ratio', dest='used_bin_ratio',type=float,default=1.0, metavar='NUM',
                      help='used bin ratio for mdeling the structure model.')
    parser.add_argument('--parallel', dest='parallelNumb',type=int,default=1, metavar='NUM',
                      help='Enable parallel processing and define the parallel processor number.')

    inps = parser.parse_args()

    return inps

################################################################################    
    
    
def main(argv):
    
    inps = cmdLineParse() 
    FILE = inps.input_file
    MODEL = inps.variogram_model
    SAMP_NUMB =inps.sample_numb
    BIN_NUMB = inps.bin_numb

    meta = readfile.read_attribute(FILE, datasetName=None) 
    bperp0 = ut.read_hdf5(FILE, datasetName='bperp')[0]
    date12 = ut.read_hdf5(FILE, datasetName='date')[0]
    
    
    mDates = [i.decode('utf8') for i in date12[:, 0]]
    sDates = [i.decode('utf8') for i in date12[:, 1]]
    date_list = sorted(list(set(mDates + sDates)))
    date_list = np.asarray(date_list)

    row = int(meta['LENGTH'])
    col = int(meta['WIDTH'])
    
    sliceList = readfile.get_slice_list(FILE)
    g_list0 = sliceList
    g_list = []
    for k0 in sliceList:
        if 'unwrap' in k0:
            g_list.append(k0)
    
    N_list = len(g_list)     
        
    if inps.used_bin_ratio:
        Ratio = inps.used_bin_ratio
    else:
        Ratio = 1.0
    #print(MASK0)
    
    MASK1 = np.ones((row,col))
    for i in range(N_list):
        dset = g_list[i]
        data = readfile.read(FILE, datasetName=dset)[0]
        MASK1 = MASK1*data
    MASK1[MASK1!=0]=1
    
    
    if inps.mask_file:
        data = readfile.read(inps.mask_file, datasetName='mask')[0]
        MASK0 = data
    else:
        MASK0 = MASK1
    
    if inps.out_file:
        OUT = inps.out_file
    else:
        OUT = 'variogramStack.h5'
        
    Resolution = float(meta['AZIMUTH_PIXEL_SIZE'])/1000.0
    Radius = float(meta['EARTH_RADIUS'])/1000.0
    Unit = meta['UNIT']
    Wavelength = float(meta['WAVELENGTH'])*100  # cm
    
    if 'X_FIRST' in meta:
        coord_type = 'geographic'
        yy,xx = get_lat_lon(meta)
    else:
        coord_type = 'euclidean'
        yy,xx = get_xx_yy(meta)
    
    Sill = np.zeros((N_list,1))
    Range = np.zeros((N_list,1))
    Nugget = np.zeros((N_list,1))
    
    BIN_NUMB0 = int(int(BIN_NUMB)*Ratio)
    
    Semivariance_all = np.zeros((N_list,int(BIN_NUMB0)))
    Lags_all = np.zeros((1,int(BIN_NUMB0)))
    STDs_all = np.zeros((N_list,int(BIN_NUMB0)))
    
    datasetDict = dict()
    
    idx_sample, y_sample, x_sample = variance.sample_data(yy, xx, MASK0, num_sample=SAMP_NUMB)
    
    data_para = []
    for i in range(N_list):
    #for i in range(1):
        #N = k0
        #print_progress(i, N_list, prefix='Data: ', suffix=sliceList[i])
        dset = g_list[i]
        #print(dset)
        data = readfile.read(FILE, datasetName=dset)[0]
        data = data*MASK0
        
        where_are_NaNs = isnan(data)
        data[where_are_NaNs] = 0
        #print(data)
        data1 = data.flatten()
        data0 = data.flatten()

        where_are_nan = np.isnan(data0)
        where_are_inf = np.isinf(data0)
        data0[where_are_nan] = 0
        data0[where_are_inf] = 0

        mask = data0
        mask[data0!=0]=1

        z_sample = data1[idx_sample]
        data0 = (x_sample, y_sample, z_sample, MODEL, coord_type,BIN_NUMB,Unit,Radius,Resolution,Wavelength)
        data_para.append(data0)

    futures = parallel_process(data_para, ok_variogram, n_jobs=inps.parallelNumb, use_kwargs=False)   
    
    for i in range(len(futures)):
        Semivariance,Lags,STDs,Sill0,Range0,Nugget0 = futures[i]
        Sill[i] = Sill0
        Range[i] = Range0
        Nugget[i] = Nugget0
        #print(Model_parameters)
        
        Semivariance_all[i,:]=Semivariance
        #Lags_all[i,:] = Lags
        STDs_all[i,:] = STDs

    datasetDict['mask'] = MASK0
    datasetDict['bperp'] = np.asarray(bperp0)
    datasetDict['date'] = np.asarray(date12)
    datasetDict['date_list'] = np.asarray(date_list,dtype = np.string_)
    
    datasetDict['Sill'] = np.float32(Sill)
    datasetDict['Range'] = np.float32(Range)
    datasetDict['Nugget'] = np.float32(Nugget)
    
    datasetDict['semivariance_std'] = np.float32(STDs_all)
    datasetDict['semivariance'] = np.float32(Semivariance_all)
    datasetDict['Lags'] = Lags
    meta['variogram_model'] = MODEL
    meta['sample_numb'] = SAMP_NUMB
    meta['bin_numb'] = BIN_NUMB
    write_variogram_h5(datasetDict, OUT, metadata=meta, ref_file=None, compression=None)
    
    
    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[:])
