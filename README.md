[![Language](https://img.shields.io/badge/python-3.5%2B-blue.svg)](https://www.python.org/)
[![Citation](https://img.shields.io/badge/doi-10.1016%2Fj.jgr.solidearth.2020JB020952-blue)](https://doi.org/10.1029/2020JB020952)

# vceSAR

An open source module in python for time-series InSAR stochastic modeling and uncertainty evaluation. We estimate and model the variance components of the absolute tropospheric effects on SAR images based on geostatistic and variance-covariance-estimation (VCE) approach. Before applying this module, we recommend users to correct topo-corelated and large-scale tropospheric delay in interferograms or InSAR time-series firstly based on global weather models (e.g., ICAMS, GACOS or PyAPS).  Basically this module is developed for "troposphere", but we should note that here the "troposphere" actually means the sum of all "spatial-correlated errors" that are not corrected (e.g., residual orbital errors, residual topo-phases, or potential ionospheric delays, etc), of course, tropospheric delay, particularly the turbulent component, is always the dominant one. 

The application of vceSAR falls into two scenarios: temporary deformation case (e.g., co-seismic) and slow-moving deformation case (e.g., tectonic movements or general ground subsidence). For temporary case, we estimate variance components of SAR images before and after the event seperately; For slow-moving cases, we estimate the variance components for all SAR images directly (at one time).

The currently version of vceSAR is developed based on MintPy products, but the basic angorithms are suitable for both SBAS and PS/DS measurements, it also can be easily merged into any other softwares of InSAR time-series analysis, e.g., StaMPS. 

This is research code provided to you "as is" with NO WARRANTIES OF CORRECTNESS. Use at your own risk.

### 1 Download

Download the development version using git:   
   
    cd ~/python
    git clone https://github.com/ymcmrs/vceSAR
    
    
### 2 Installation

 1） To make vceSAR importable in python, by adding the path vceSAR directory to your $PYTHONPATH
     For csh/tcsh user, add to your **_~/.cshrc_** file for example:   

    ############################  Python  ###############################
    if ( ! $?PYTHONPATH ) then
        setenv PYTHONPATH ""
    endif
    
    ##--------- Anaconda ---------------## 
    setenv PYTHON3DIR    ~/python/anaconda3
    setenv PATH          ${PATH}:${PYTHON3DIR}/bin
    
    ##--------- PyINT ------------------## 
    setenv VCESAR_HOME    ~/python/vceSAR      
    setenv PYTHONPATH    ${PYTHONPATH}:${VCESAR_HOME}
    setenv PATH          ${PATH}:${VCESAR_HOME}/vceSAR
    
 2） Install dependencies
    
    $CONDA_PREFIX/bin/conda install -c conda-forge mintpy
    $CONDA_PREFIX/bin/pip install git+https://github.com/ymcmrs/PyKrige.git   
    

### 2 Running vceSAR

1). Calculate variogram of short-temporal baseline interferograms using #vce_ifg.py. 

    Usage:
            vce_ifg.py ifgramStack.h5 --parallel 4
            vce_ifg.py ifgramStack.h5 -m maskTempCoh.h5 --sample_numb 5000
            vce_ifg.py ifgramStack.h5 -m maskTempCoh.h5 --bin_numb 30
            
            vce_ifg.py timeseries.h5 --parallel 8
            vce_ifg.py timeseries.h5 --bin_numb 20
            vce_ifg.py timeseries.h5 -m maskTempCoh.h5 --bin_numb 30


2) Estimate of the variance components for time-series of SAR images using #vce_SAR.py

    Usage:
            vce_sar.py variogramStack.h5 --weight
            vce_sar.py variogramStack.h5 


3) Modeling the spatial variance components for SAR images one by one using [#vce_modeling.py]
   
    Usage:
            vce_modeling.py variogramTs.h5 
            vce_modeling.py variogramStack.h5 --model gaussian
            vce_modeling.py variogramTs.h5 --max-length 150 --model spherical









