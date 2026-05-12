#! /usr/bin/env python
# variogram module used for variance calculate and structure function modeling
# Author: Yunmeng Cao   02 Oct., 2021 

import numpy as np
import pyproj
import random
import matplotlib.pyplot as plt
from mintpy.utils import ptime

import pykrige
from pykrige import OrdinaryKriging
from pykrige import variogram_models
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.stats.stats import pearsonr

variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model,
                      'hole-effect': variogram_models.hole_effect_variogram_model}

###################################################################################

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
    #print(x0)
    #res = leastsq(_variogram_residuals, x0,args=(lags, semivariance,variogram_function, weight))
    #print(res[0])
    return res.x


def get_lat_lon(atr):
    '''Get lat/lon of all pixels'''
    length = int(atr['LENGTH'])
    width = int(atr['WIDTH'])
    lat0 = float(atr['Y_FIRST'])
    lon0 = float(atr['X_FIRST'])
    lat_step = float(atr['Y_STEP'])
    lon_step = float(atr['X_STEP'])
    lat1 = lat0 + lat_step*length
    lon1 = lon0 + lon_step*width

    lat, lon = np.mgrid[lat0:lat1:length*1j, lon0:lon1:width*1j]
    lat = lat.flatten()
    lon = lon.flatten()
    return lat, lon


def sample_data(lat, lon, mask=None, num_sample=500):
    ''''''
    # Flatten input data
    for i in [lat, lon]:
        if len(i.shape) != 1:
            i = i.flatten()

    # Check number of samples and number of pixels
    num_pixel = len(lat)
    if num_sample > num_pixel:
        print('Number of samples > number of pixels, fix number of samples to 4/5 number of pixels.')
        num_sample = int(4/5*num_pixel)
    # Check input mask
    if mask is None:
        mask = np.ones((num_pixel))

    # Random select samples
    idx = np.arange(num_pixel)
    idx_sample = random.sample(list(idx[mask.flatten() == 1.0]), int(num_sample))
    #idx_sample = random.sample(idx.tolist(), int(num_sample))
    lat_sample = lat[idx_sample]
    lon_sample = lon[idx_sample]

    return idx_sample, lat_sample, lon_sample


def get_distance(lat, lon, i):
    '''Return the distance of all points in lat/lon from its ith point'''
    lat1 = lat[i]*np.ones(lat.shape)
    lon1 = lon[i]*np.ones(lon.shape)

    g = pyproj.Geod(ellps='WGS84')
    dist = g.inv(lon1, lat1, lon, lat)[2]
    return dist


def get_distance_matrix(lat, lon):
    '''Return the distance of all points in lat/lon from its ith point'''
    dist_matrix = np.zeros((len(lat),len(lat)))
    
    for i in range(len(lat)):
        lat1 = lat[i]*np.ones(lat.shape)
        lon1 = lon[i]*np.ones(lon.shape)

        g = pyproj.Geod(ellps='WGS84')
        dist0 = g.inv(lon1, lat1, lon, lat)[2]
        
        dist_matrix[:,i] = dist0
    return dist_matrix


def structure_function(data, lat, lon, step=5e3, min_pair_num=100e3, print_msg=True):
    num_sample = len(data)
    distance = np.zeros((num_sample**2))
    variance = np.zeros((num_sample**2))
    if print_msg:
        prog_bar = ptime.progressBar(maxValue=num_sample)
    for i in range(num_sample):
        distance[i*num_sample:(i+1)*num_sample] = get_distance(lat, lon, i)
        variance[i*num_sample:(i+1)*num_sample] = np.square(data - data[i])
        if print_msg:
            prog_bar.update(i+1, every=10)
    if print_msg:
        prog_bar.close()

    dist, std, stdStd = bin_variance(
        distance, variance, step=step, min_pair_num=min_pair_num, print_msg=print_msg)
    return dist, std, stdStd


def bin_variance(distance, variance, step=5e3, min_pair_num=100e3, print_msg=True):
    x_steps = np.arange(0, np.max(distance), step)
    num_step = len(x_steps)
    std = np.zeros(x_steps.shape)
    stdStd = np.zeros(std.shape)
    p_num = np.zeros(x_steps.shape)

    if print_msg:
        prog_bar = ptime.progressBar(maxValue=num_step)
    for i in range(num_step):
        x = x_steps[i]
        idx = (distance > max(0, x-step/2.)) * (distance < x+step/2.)
        p_num[i] = np.sum(idx)
        std[i] = np.mean(np.sqrt(variance[idx]))
        stdStd[i] = np.std(np.sqrt(variance[idx]))
        if print_msg:
            prog_bar.update(i+1, every=10)
    if print_msg:
        prog_bar.close()

    max_step_idx = int(max(np.argwhere(p_num > min_pair_num)))
    return x_steps[0:max_step_idx], std[0:max_step_idx], stdStd[0:max_step_idx]

def variance_modeling(data, lat, lon, samp_numb, step, max_length, model, weight=True):
    
    idx_sample, lat_sample, lon_sample = sample_data(lat, lon, mask=None, num_sample=samp_numb)
    data_sample = data[idx_sample]
    
    dist, std, stdStd = structure_function(data_sample, lat_sample, lon_sample, step=step, min_pair_num=30, print_msg=False)
    #print(dist)
    #print(std)
    
    std = std[dist < max_length] ## in meters
    stdStd = stdStd[dist < max_length]
    dist = dist[dist < max_length]
    
    #print(dist)
    #print(std)
    
    variogram_function = variogram_dict[model]
    tt = calculate_variogram_model(dist, std, model, variogram_function, weight) # weight = True, give large weights to small lags
    corr, _ = pearsonr(std, variogram_function(tt,dist))
    
    
    return tt, corr
