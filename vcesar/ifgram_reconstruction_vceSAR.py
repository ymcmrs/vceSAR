#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2020, Yunmeng Cao                          #
# Author: Yunmeng Cao , 2020                               #
############################################################
import os
import h5py
import argparse
import numpy as np
from mintpy.objects import timeseries
from mintpy.utils import ptime,readfile, writefile


def read_exclude_date(ex_date_list, date_list_all, print_msg=True):
    """Read exclude dates info
    Parameters: ex_date_list  : list of string, date in YYMMDD or YYYYMMDD format,
                                or text file with date in it
                date_list_all : list of string, date in YYYYMMDD format
    Returns:    drop_date     : 1D array of bool in size of (num_date,)
    """
    # Read exclude date input
    ex_date_list = ptime.read_date_list(ex_date_list)
    if ex_date_list and print_msg:
        print(('exclude the following dates for DEM error estimation:'
               ' ({})\n{}').format(len(ex_date_list), ex_date_list))

    # convert to mark array
    
    N = len(date_list_all)
    drop_date = []
    idx_exclude = []
    for i in range(N):
        date0 = date_list_all[i]
        if date0 not in ex_date_list:
            drop_date.append(date0)
        else:
            idx_exclude.append(i)
    
    #drop_date = np.array([i not in ex_date_list for i in date_list_all],
    #                     dtype=np.bool_)
    return drop_date, idx_exclude

def generate_designMatrix(N,Nc):

    SS = (N-Nc)*Nc + sum(range(Nc))
    G = np.zeros((Nc*N,N+Nc))

    sk = 0
    for i in range(N):
        for j in range(Nc):
            if (i+j+1)<N:
                #G[i*Nc+j,i]=1
                #G[i*Nc+j,i+j+1]=-1
                G[sk,i] = 1
                G[sk,i+j+1] =-1
                sk = sk+1  
    
    G0 = G[0:SS,0:N]
    G0 = np.asarray(G0,dtype=int)
    
    return G0

def write_h5(datasetDict, out_file, metadata=None, ref_file=None, compression=None):

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

#####################################################################################
EXAMPLE = """example:
  ifgram_reconstruction_nvce.py  timeseries.h5  -n 2
  ifgram_reconstruction_nvce.py  timeseries_ECWMF_ramp_demErr.h5  
"""

def create_parser():
    parser = argparse.ArgumentParser(description='Reconstruct network of interferograms from time-series',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)

    parser.add_argument('timeseries_file', type=str, help='time-series file.')
    parser.add_argument('-n','--numb', dest='numb', type=int, default=2, help='connected date number')
    parser.add_argument('--ex', '--exclude', dest='excludeDate', nargs='*', default=[],
                        help='Exclude date(s) for ifg-reconstruction estimation.\n' +
                             'All dates will be corrected for DEM residual phase still.')
    parser.add_argument('--oneYear', dest='oneYear', action='store_true', help='Using rslc/rslcPar to check orbit history.')
    parser.add_argument('-o','--output', dest='out_file', default='reconUnwrapIfgram.h5',
                        help='output filename for the reconstructed interferograms.')
    return parser

def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    return inps


#####################################################################################
def timeseries2ifgram2(ts_file, Nc, exclude_date_list, out_file='reconUnwrapIfgram.h5'):
    # read time-series
    atr = readfile.read_attribute(ts_file)
    #atr0 = readfile.read_attribute(ifgram_file)
    range2phase = -4.*np.pi / float(atr['WAVELENGTH'])
    print('reading timeseries data from file {} ...'.format(ts_file))
    ts_data0 = readfile.read(ts_file)[0] * range2phase
    num_date, length, width = ts_data0.shape
    ts_data0 = ts_data0.reshape(num_date, -1)

    ts_obj = timeseries(ts_file)
    
    with h5py.File(ts_file, 'r') as f:
        date_list_all = [i.decode('utf8') for i in f['date'][:]]
        bperp = [i for i in f['bperp'][:]]
    # reconstruct unwrapPhase
    print('reconstructing the interferograms from timeseries')
    
    date_list, idx_exclude = read_exclude_date(exclude_date_list, date_list_all, print_msg=True)
    ts_data = np.delete(ts_data0,idx_exclude,0)
    num_date = len(date_list)
    
    A1 = generate_designMatrix(num_date,Nc); B1 = A1.copy(); C1 = A1.copy()
    num_ifgram = A1.shape[0]
    
    date12 = []
    bperp12 = []
    num_ifgram_final = 0
    for i in range(num_ifgram):
        pp = C1[i,:]
        pp = pp.reshape(len(pp.flatten()),)
        kk = np.where(pp!=0)
        kk = kk[0]
        #print(kk[0])
        #print(date_list)
        s0 = date_list[kk[0]];year0 = s0[0:4]
        s1 = date_list[kk[1]];year1 = s1[0:4]
        k0 = date_list[kk[0]] + '_' + date_list[kk[1]]
        bperp0 = bperp[kk[1]] - bperp[kk[0]]
        if year0==year1:
            #print('Only consider pairs within one Year.')
            print(k0)
            date12.append(k0)
            bperp12.append(bperp0)
            num_ifgram_final = num_ifgram_final + 1
        else:
             B1 = np.delete(B1, i, 0)

    #stack_obj = ifgramStack(ifgram_file)
    #stack_obj.open(print_msg=False)
    #A1 = stack_obj.get_design_matrix4timeseries(stack_obj.get_date12_list(dropIfgram=True))[0]
    #date_list = stack_obj.get_date_list(dropIfgram=True)
    #date12 = stack_obj.get_date12_list(dropIfgram=True)
    #print(date12)
   
    #with h5py.File(ifgram_file, 'r') as f:
    #    pbaseIfgram = f['bperp'][:]
    #    pbaseIfgram = pbaseIfgram[f['dropIfgram'][:]]

    
    #A0 = -1.*np.ones((num_ifgram, 1))
    #A = np.hstack((A0, A1))
    ifgram_est = np.dot(B1, ts_data).reshape(num_ifgram_final, length, width)
    ifgram_est = np.array(ifgram_est, dtype=ts_data.dtype)
    del ts_data
    
    date12m = np.zeros((len(date12),2),dtype = '<S8')
    for i in range(len(date12)):
        k0 = date12[i]
        date12m[i,0]=k0.split('_')[0] 
        date12m[i,1] = k0.split('_')[1]

    dropIfgram = np.ones((len(date12),),dtype = bool) 
    print(np.asarray(date12m,dtype = '<S8'))
    # write to ifgram file
    atr['FILE_TYPE'] = 'ifgramStack'                            
    dsDict = {}
    dsDict['unwrapPhase'] = ifgram_est
    dsDict['dropIfgram'] = dropIfgram
    dsDict['date_list'] = np.asarray(date_list,dtype = '<S8')
    dsDict['date'] = np.asarray(date12m,dtype = '<S8')
    dsDict['bperp'] = np.asarray(bperp12)
    atr['UNIT'] = 'radian'
    write_h5(dsDict, out_file = out_file, metadata=atr, ref_file=None, compression=None)
    
    #writefile.write(dsDict, out_file=out_file, ref_file=ifgram_file)
    return out_file


def timeseries2ifgram(ts_file, Nc, exclude_date_list, out_file='reconUnwrapIfgram.h5'):
    # read time-series
    atr = readfile.read_attribute(ts_file)
    #atr0 = readfile.read_attribute(ifgram_file)
    range2phase = -4.*np.pi / float(atr['WAVELENGTH'])
    print('reading timeseries data from file {} ...'.format(ts_file))
    ts_data0 = readfile.read(ts_file)[0] * range2phase
    num_date, length, width = ts_data0.shape
    ts_data0 = ts_data0.reshape(num_date, -1)

    ts_obj = timeseries(ts_file)
    
    with h5py.File(ts_file, 'r') as f:
        date_list_all = [i.decode('utf8') for i in f['date'][:]]
        bperp = [i for i in f['bperp'][:]]
    # reconstruct unwrapPhase
    print('reconstructing the interferograms from timeseries')
    
    date_list, idx_exclude = read_exclude_date(exclude_date_list, date_list_all, print_msg=True)
    ts_data = np.delete(ts_data0,idx_exclude,0)
    num_date = len(date_list)
    
    A1 = generate_designMatrix(num_date,Nc)
    num_ifgram = A1.shape[0]
    
    date12 = []
    bperp12 = []
    for i in range(num_ifgram):
        pp = A1[i,:]
        pp = pp.reshape(len(pp.flatten()),)
        kk = np.where(pp!=0)
        kk = kk[0]
        #print(kk[0])
        #print(date_list)
        s0 = date_list[kk[0]];year0 = s0[0:4]
        s1 = date_list[kk[1]];year1 = s1[0:4]
        k0 = date_list[kk[0]] + '_' + date_list[kk[1]]
        bperp0 = bperp[kk[1]] - bperp[kk[0]]

        date12.append(k0)
        bperp12.append(bperp0)
    
    #stack_obj = ifgramStack(ifgram_file)
    #stack_obj.open(print_msg=False)
    #A1 = stack_obj.get_design_matrix4timeseries(stack_obj.get_date12_list(dropIfgram=True))[0]
    #date_list = stack_obj.get_date_list(dropIfgram=True)
    #date12 = stack_obj.get_date12_list(dropIfgram=True)
    #print(date12)
   
    #with h5py.File(ifgram_file, 'r') as f:
    #    pbaseIfgram = f['bperp'][:]
    #    pbaseIfgram = pbaseIfgram[f['dropIfgram'][:]]

    
    #A0 = -1.*np.ones((num_ifgram, 1))
    #A = np.hstack((A0, A1))
    ifgram_est = np.dot(A1, ts_data).reshape(num_ifgram, length, width)
    ifgram_est = np.array(ifgram_est, dtype=ts_data.dtype)
    del ts_data
    
    date12m = np.zeros((len(date12),2),dtype = '<S8')
    for i in range(len(date12)):
        k0 = date12[i]
        date12m[i,0]=k0.split('_')[0] 
        date12m[i,1] = k0.split('_')[1]

    dropIfgram = np.ones((len(date12),),dtype = bool) 
    print(np.asarray(date12m,dtype = '<S8'))
    # write to ifgram file
    atr['FILE_TYPE'] = 'ifgramStack'                            
    dsDict = {}
    dsDict['unwrapPhase'] = ifgram_est
    dsDict['dropIfgram'] = dropIfgram
    dsDict['date_list'] = np.asarray(date_list,dtype = '<S8')
    dsDict['date'] = np.asarray(date12m,dtype = '<S8')
    dsDict['bperp'] = np.asarray(bperp12)
    atr['UNIT'] = 'radian'
    write_h5(dsDict, out_file = out_file, metadata=atr, ref_file=None, compression=None)
    
    #writefile.write(dsDict, out_file=out_file, ref_file=ifgram_file)
    return out_file

def main(iargs=None):
    inps = cmd_line_parse(iargs)
    if inps.oneYear:
        timeseries2ifgram2(inps.timeseries_file, inps.numb, inps.excludeDate, inps.out_file)
    else:
        timeseries2ifgram(inps.timeseries_file, inps.numb, inps.excludeDate, inps.out_file)
    return


#####################################################################################
if __name__ == '__main__':
    main()
