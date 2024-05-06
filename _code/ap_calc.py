### PURPOSE: Calculate and plot apparent temperature
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 05/06/2024

##################################################################################################################
#%% IMPORT LIBRARIES, DATA, AND FORMAT
# Import libraries
from matplotlib.lines import Line2D, lineStyles
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
import glob
import copy
from importlib import reload #to use type reload(fun)
import matplotlib.patches as mpatches
from scipy import signal
import statsmodels.api as sm
import function_dependencies as fun
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import datetime
import geopandas as gpd
import rioxarray as rxr
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from matplotlib.patches import Patch
import pdb
from matplotlib import cm
import matplotlib.patches as mpatches
import seaborn as sns
import cartopy.io.shapereader as shpreader
from matplotlib import ticker
import time
# turn interactive plotting on/off
plt.ion();


## READ IN DATA
## MONTHLY ERA5 REANALYSIS 
# Note: this data is not in repository but can be downloaded from C3S CDS and regridded to correct resolution.
# See Hersbach et al., 2023. https://doi.org/10.24381/cds.f17050d7
wd_era = '../_data/ERA5/monthly/'
era = fun.reorient_netCDF(xr.open_mfdataset(glob.glob(wd_era+'*.nc')))
# Add on Tmax 2m from daily pre-processing
wd_era = '../_data/ERA5/hourly/'
mx2t = fun.reorient_netCDF(xr.open_mfdataset(glob.glob(wd_era+'*.nc')))
era = era.merge(mx2t)


# CESM-LENS2 CLIMATOLOGICAL RH_MIN CORRECTION
# Note: this file is not in repository but can be created using the LENS2 RH data.
# See Methods in Wan et al., 2024.
wd_lens2 = '../_data/LENS2/'
lens2_rh_err = fun.reorient_netCDF(xr.open_dataset(wd_lens2+'b.e21.BHISTsmbb.f09_g17.LE2-ensemble.mean.cam.clim.land.RHREFHT_err.2000010100-2009123100.nc'))


## CESM2 OUTPUT
# Create dictionaries to hold both 2010 and 2050 output
atm_monthly_base = {}
atm_daily_base = {}
lnd_monthly_base = {}
lnd_daily_base = {}

atm_monthly_st = {}
atm_daily_st = {}
lnd_monthly_st = {}
lnd_daily_st = {}

atm_monthly_ml = {}
atm_daily_ml = {}
lnd_monthly_ml = {}
lnd_daily_ml = {}


## Read in each CESM2 simulation as xarray dataset
yr_keys = ['2050', '2010']
sim_length = 30 #years
wd_data = '../_data/'
for yr in yr_keys:
    if yr=='2050':
        ## 1) CONTROL CASE
        casename_base = 'b.e22.B2000.f09_g17.2050cycle.001'
        atm_monthly_base[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.001.cam.h0.processed.0056-0085.nc')
        atm_daily_base[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.001.cam.h1.processed.0056-0085.nc')
        ## 2) MCB SUBTROPICAL SEEDING CASE
        casename_st = 'b.e22.B2000.f09_g17.2050cycle.MCB-mask15.001'
        atm_monthly_st[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.MCB-mask15.001.cam.h0.processed.0056-0085.nc')
        atm_daily_st[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.MCB-mask15.001.cam.h1.processed.0056-0085.nc')
        ## 3) MCB MID-LATITUDE SEEDING CASE
        casename_ml = 'b.e22.B2000.f09_g17.2050cycle.MCB-mask16.001'
        atm_monthly_ml[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.MCB-mask16.001.cam.h0.processed.0056-0085.nc')
        atm_daily_ml[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.MCB-mask16.001.cam.h1.processed.0056-0085.nc')
    elif yr=='2010':
        casename_base = 'b.e22.B2000.f09_g17.2010cycle.002'
        atm_monthly_base[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.002.cam.h0.processed.0056-0085.nc')
        atm_daily_base[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.002.cam.h1.processed.0056-0085.nc')
        ## 2) MCB SUBTROPICAL SEEDING CASE
        casename_st = 'b.e22.B2000.f09_g17.2010cycle.MCB-mask15.002'
        atm_monthly_st[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.MCB-mask15.002.cam.h0.processed.0056-0085.nc')
        atm_daily_st[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.MCB-mask15.002.cam.h1.processed.0056-0085.nc')
        ## 3) MCB SUBTROPICAL SEEDING CASE
        casename_ml = 'b.e22.B2000.f09_g17.2010cycle.MCB-mask16.002'
        atm_monthly_ml[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.MCB-mask16.002.cam.h0.processed.0056-0085.nc')
        atm_daily_ml[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.MCB-mask16.002.cam.h1.processed.0056-0085.nc')      


## Read in state IDs
# Note: this file is not in the repository but can be created by downloading the 20m resolution states shapefile from 
# the US Census Bureau and regridding to the correct resolution.
# https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
stateid = fun.reorient_netCDF(xr.open_dataset('../_data/cb_2018_us_state_0.9x1.25.nc'))
stateid = stateid.assign_coords(lat=(atm_monthly_ml['2010'].lat[::-1]), lon=(atm_monthly_ml['2010'].lon))


## Read in regridded population data
# Note: these files are also not in the repository but can be downloaded from SEDAC and regridded to the correct resolution.
# 2010: https://sedac.ciesin.columbia.edu/data/collection/gpw-v4
# 2050 (SSP2): https://sedac.ciesin.columbia.edu/data/collection/ssp/sets/browse
## 2010 ##
regrid_pop_count = xr.open_dataset(wd_data+'gpw-v4-population-count-rev11_totpop_192x288.nc')
## 2050 ##
regrid_pop_count_ssp2 = fun.reorient_netCDF(xr.open_dataset(wd_data+'ssp2_2050_manual_regrid_192x288.nc'))


# Create West Coast land mask
## Use administrative IDs to select specific states (WA=53., OR=41., CA=06.)
west_coast_state_mask = stateid.StateID * 0
## West Coast States
west_coast_state_mask = xr.where((stateid.StateID==53.) | (stateid.StateID==41.) | (stateid.StateID==6.),\
                        stateid.StateID,0)
        

## Match up lat and lon coordinates in ERA5 with CESM2 grid
era = era.assign_coords({'lon':atm_monthly_base['2010'].lon,\
                            'lat':atm_monthly_base['2050'].lat})
lens2_rh_err = lens2_rh_err.assign_coords({'lon':atm_monthly_base['2010'].lon,\
                            'lat':atm_monthly_base['2050'].lat})


## Convert units from ERA5 to CESM2 units
# t2m, mx2t K to deg C
era['t2m'] = era['t2m'] - 273.15 #deg C
era['t2m'].attrs['units'] = 'deg C'
era['mx2t'] = era['mx2t'] - 273.15 #deg C
era['mx2t'].attrs['units'] = 'deg C'


# r change name to 'fraction' instead of '%'
era['r'].attrs['units'] = 'fraction'
# select surface r has ~2m reference height
era['r'] = era['r'].isel(level=2)

# tp m to mm/day
m_to_mm = 1e3 #mm/m
# Create dictionary with number of days for each month (no leap calendar)
month_day_dict = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
# Loop through each month and convert from m to mm/day
for i in list(month_day_dict.keys()):
    tmp_month_day = month_day_dict[i]
    era['tp'].loc[{'time':[t for t in pd.to_datetime(era.time.values) if t.month==i]}]*=m_to_mm/month_day_dict[i]
era['tp'].attrs['units'] = 'mm/day'

# si10 change name to 'm/s/ instead of 'm s**-1'
era['si10'].attrs['units'] = 'm/s'

# slhf, sshf convert from J/m2 to W/m2
# According to ERA5 documentation (https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview), 
# divide by accumulation period (1 hour) in seconds
hr_to_s = 60*60
era['slhf'] = era['slhf']/ hr_to_s
era['slhf'].attrs['units'] = 'W/m2'
era['sshf'] = era['sshf']/ hr_to_s
era['sshf'].attrs['units'] = 'W/m2'

# sp is good (Pa)

# cc,tcc,lcc change name to 'fraction' instead of '(0 - 1)'
era['cc'].attrs['units'] = 'fraction'
era['tcc'].attrs['units'] = 'fraction'
era['lcc'].attrs['units'] = 'fraction'

# clwc is in kg kg**-1 (same as default CESM), but I don't have a delta P value to convert to kg/m2. will ignore this variable for now

## Rename ERA5 variables to match CESM2 names
era = era.rename({'t2m':'TREFHT','slhf':'LHFLX','tp':'PRECT','sp':'PS','r':'RHREFHT',\
                    'sshf':'SHFLX','mx2t':'TREFHTMX','si10':'U10','cc':'CLOUD','tcc':'CLDTOT','lcc':'CLDLOW'})



##################################################################################################################
#%% COMPUTE MONTHLY CLIMATOLOGIES AND BIAS CORRECTION
## Compute monthly climatologies
## ERA5
era_clim = era.groupby('time.month').mean(dim='time')

## CESM2 output
# Create list of ATM variable names to bias correct
atm_varnames_bc = ['PRECT','PS','TREFHT','TREFHTMX','RHREFHT','U10','LHFLX','SHFLX','CLDLOW','CLDTOT']


## 1a) MONTHLY ATMOSPHERE
# Create empty dictionaries for climatologies and anomalies
atm_monthly_base_clim = {}
atm_monthly_st_clim = {}
atm_monthly_ml_clim = {}
atm_monthly_base_clim_res = {}
atm_monthly_st_clim_res = {}
atm_monthly_ml_clim_res = {}
print('##ATM MONTHLY##')
for yr in yr_keys:
        print(yr)
        atm_monthly_base_clim[yr] = {}
        atm_monthly_st_clim[yr] = {}
        atm_monthly_ml_clim[yr] = {}
        atm_monthly_base_clim_res[yr] = {}
        atm_monthly_st_clim_res[yr] = {}
        atm_monthly_ml_clim_res[yr] = {}
        for varname in atm_varnames_bc:
            print(varname)
            # Compute monthly climatology for model output
            atm_monthly_base_clim[yr][varname] = atm_monthly_base[yr][varname].groupby('time.month').mean()
            atm_monthly_base_clim[yr][varname].attrs['units'] = atm_monthly_base[yr][varname].units
            atm_monthly_st_clim[yr][varname] = atm_monthly_st[yr][varname].groupby('time.month').mean()
            atm_monthly_st_clim[yr][varname].attrs['units'] = atm_monthly_st[yr][varname].units
            atm_monthly_ml_clim[yr][varname] = atm_monthly_ml[yr][varname].groupby('time.month').mean()
            atm_monthly_ml_clim[yr][varname].attrs['units'] = atm_monthly_ml[yr][varname].units
            # Compute bias correction residual from monthly climatologies
            atm_monthly_base_clim_res[yr][varname] = atm_monthly_base_clim[yr][varname] - era_clim[varname]
            atm_monthly_base_clim_res[yr][varname].attrs['units'] = atm_monthly_base_clim[yr][varname].units
            atm_monthly_st_clim_res[yr][varname] = atm_monthly_st_clim[yr][varname] - era_clim[varname]
            atm_monthly_st_clim_res[yr][varname].attrs['units'] = atm_monthly_st_clim[yr][varname].units
            atm_monthly_ml_clim_res[yr][varname] = atm_monthly_ml_clim[yr][varname] - era_clim[varname]
            atm_monthly_ml_clim_res[yr][varname].attrs['units'] = atm_monthly_ml_clim[yr][varname].units
                 

# bc_option = input('Bias correct (y or n)?:')
# rh_bc_option = input('Bias correct RH to RHmin (y or n)?: ')
bc_option = 'y'
rh_bc_option = 'y'
## Bias correct CESM2 daily and monthly atmospheric variables using 2010 control residuals
i_month = np.arange(1,13,1)
if (bc_option=='y'):
    for yr in yr_keys:
        print(yr)
        for varname in atm_varnames_bc:
            print(varname)
            for i in i_month:
                ## CONTROL
                atm_monthly_base[yr][varname].loc[{'time':[t for t in atm_monthly_base[yr].time.values if t.month==i]}] -=  atm_monthly_base_clim_res['2010'][varname].sel(month=i)
                atm_daily_base[yr][varname].loc[{'time':[t for t in atm_daily_base[yr].time.values if t.month==i]}] -=  atm_monthly_base_clim_res['2010'][varname].sel(month=i)
                ## ST MCB
                atm_monthly_st[yr][varname].loc[{'time':[t for t in atm_monthly_st[yr].time.values if t.month==i]}] -=  atm_monthly_base_clim_res['2010'][varname].sel(month=i)
                atm_daily_st[yr][varname].loc[{'time':[t for t in atm_daily_st[yr].time.values if t.month==i]}] -=  atm_monthly_base_clim_res['2010'][varname].sel(month=i)
                ## ML MCB
                atm_monthly_ml[yr][varname].loc[{'time':[t for t in atm_monthly_ml[yr].time.values if t.month==i]}] -=  atm_monthly_base_clim_res['2010'][varname].sel(month=i)
                atm_daily_ml[yr][varname].loc[{'time':[t for t in atm_daily_ml[yr].time.values if t.month==i]}] -=  atm_monthly_base_clim_res['2010'][varname].sel(month=i)
                if (varname=='RHREFHT' and rh_bc_option=='y'):
                    ## CONTROL
                    atm_monthly_base[yr]['RHREFHT'].loc[{'time':[t for t in atm_monthly_base[yr].time.values if t.month==i]}] -=  lens2_rh_err['RHREFHT_err'].sel(month=i)
                    atm_daily_base[yr]['RHREFHT'].loc[{'time':[t for t in atm_daily_base[yr].time.values if t.month==i]}] -=  lens2_rh_err['RHREFHT_err'].sel(month=i)
                    ## ST MCB
                    atm_monthly_st[yr]['RHREFHT'].loc[{'time':[t for t in atm_monthly_st[yr].time.values if t.month==i]}] -=  lens2_rh_err['RHREFHT_err'].sel(month=i)
                    atm_daily_st[yr]['RHREFHT'].loc[{'time':[t for t in atm_daily_st[yr].time.values if t.month==i]}] -=  lens2_rh_err['RHREFHT_err'].sel(month=i)
                    ## ML MCB
                    atm_monthly_ml[yr]['RHREFHT'].loc[{'time':[t for t in atm_monthly_ml[yr].time.values if t.month==i]}] -=  lens2_rh_err['RHREFHT_err'].sel(month=i)
                    atm_daily_ml[yr]['RHREFHT'].loc[{'time':[t for t in atm_daily_ml[yr].time.values if t.month==i]}] -=  lens2_rh_err['RHREFHT_err'].sel(month=i)


#%% COMPUTE CLIMATOLOGIES AND ANOMALIES FOR SELECT VARIABLES USING BIAS CORRECTED DATA
# Create list of ATM variable names
atm_varnames_monthly_subset = ['CCN3','CDNUMC','CLDLIQ','CLDLOW','FSNTOA','LANDFRAC','PM25_SRF',\
                               'PRECT','PS','RELHUM','SWCF','TREFHT','TREFHTMN','TREFHTMX','RHREFHT',\
                               'PSL','U10','LHFLX','SHFLX','U','V']
atm_varnames_daily_subset = ['TREFHT','TREFHTMN','TREFHTMX','RHREFHT','PRECT','CLDLOW','CLDTOT','PSL','PS','U10','LHFLX','SHFLX','LANDFRAC']


## 1a) MONTHLY ATMOSPHERE
# Create empty dictionaries for climatologies and anomalies
atm_monthly_base_clim = {}
atm_monthly_st_clim = {}
atm_monthly_ml_clim = {}
atm_monthly_st_anom = {}
atm_monthly_ml_anom = {}
print('##ATM MONTHLY##')
for yr in yr_keys:
    print(yr)
    atm_monthly_base_clim[yr] = {}
    atm_monthly_st_clim[yr] = {}
    atm_monthly_ml_clim[yr] = {}
    atm_monthly_st_anom[yr] = {}
    atm_monthly_ml_anom[yr] = {}
    for varname in atm_varnames_monthly_subset:
        print(varname)
        if (varname=='CCN3' or varname=='CDNUMC'):
            # Subset variable at 859.5 hPa
            var_base_859hpa = atm_monthly_base[yr][varname].isel({'lev':-7})
            var_st_859hpa = atm_monthly_st[yr][varname].isel({'lev':-7})
            var_ml_859hpa = atm_monthly_ml[yr][varname].isel({'lev':-7})
            # Compute monthly climatology
            atm_monthly_base_clim[yr][varname] = var_base_859hpa.groupby('time.month').mean()
            atm_monthly_base_clim[yr][varname].attrs['units'] = atm_monthly_base[yr][varname].units
            atm_monthly_st_clim[yr][varname] = var_st_859hpa.groupby('time.month').mean()
            atm_monthly_st_clim[yr][varname].attrs['units'] = atm_monthly_st[yr][varname].units
            atm_monthly_ml_clim[yr][varname] = var_ml_859hpa.groupby('time.month').mean()
            atm_monthly_ml_clim[yr][varname].attrs['units'] = atm_monthly_ml[yr][varname].units
        elif(varname=='U' or varname=='V'):
                # Subset variable at 859hPa
                var_base_859hpa = atm_monthly_base[yr][varname].isel({'lev':-7})
                var_st_859hpa = atm_monthly_st[yr][varname].isel({'lev':-7})
                var_ml_859hpa = atm_monthly_ml[yr][varname].isel({'lev':-7})                
                # Compute monthly climatology
                atm_monthly_base_clim[yr][varname] = var_base_859hpa.groupby('time.month').mean()
                atm_monthly_base_clim[yr][varname].attrs['units'] = atm_monthly_base[yr][varname].units
                atm_monthly_st_clim[yr][varname] = var_st_859hpa.groupby('time.month').mean()
                atm_monthly_st_clim[yr][varname].attrs['units'] = atm_monthly_st[yr][varname].units
                atm_monthly_ml_clim[yr][varname] = var_ml_859hpa.groupby('time.month').mean()
                atm_monthly_ml_clim[yr][varname].attrs['units'] = atm_monthly_ml[yr][varname].units
        else:
            # Compute monthly climatology
            atm_monthly_base_clim[yr][varname] = atm_monthly_base[yr][varname].groupby('time.month').mean()
            atm_monthly_base_clim[yr][varname].attrs['units'] = atm_monthly_base[yr][varname].units
            atm_monthly_st_clim[yr][varname] = atm_monthly_st[yr][varname].groupby('time.month').mean()
            atm_monthly_st_clim[yr][varname].attrs['units'] = atm_monthly_st[yr][varname].units
            atm_monthly_ml_clim[yr][varname] = atm_monthly_ml[yr][varname].groupby('time.month').mean()
            atm_monthly_ml_clim[yr][varname].attrs['units'] = atm_monthly_ml[yr][varname].units
        # Compute monthly anomalies (MCB - CONTROL)
        atm_monthly_st_anom[yr][varname] = atm_monthly_st_clim[yr][varname] - atm_monthly_base_clim[yr][varname]
        atm_monthly_st_anom[yr][varname].attrs['units'] = atm_monthly_base[yr][varname].units
        atm_monthly_ml_anom[yr][varname] = atm_monthly_ml_clim[yr][varname] - atm_monthly_base_clim[yr][varname]
        atm_monthly_ml_anom[yr][varname].attrs['units'] = atm_monthly_base[yr][varname].units


## 1b) ANNUAL ATMOSPHERE (FROM MONTHLY DATA)
# Create empty dictionaries for climatologies and anomalies
atm_annual_base_clim = {}
atm_annual_st_clim = {}
atm_annual_ml_clim = {}
atm_annual_st_anom = {}
atm_annual_ml_anom = {}
atm_annual_ctrl_anom = {}
print('##ATM ANNUAL##')
for yr in yr_keys:
    print(yr)
    atm_annual_base_clim[yr] = {}
    atm_annual_st_clim[yr] = {}
    atm_annual_ml_clim[yr] = {}
    atm_annual_st_anom[yr] = {}
    atm_annual_ml_anom[yr] = {}
    for varname in atm_varnames_monthly_subset:
            print(varname)
            if (varname=='CCN3' or varname=='CDNUMC'):
                    # Subset variable at 859.5 hPa
                    var_base_859hpa = atm_monthly_base[yr][varname].isel({'lev':-7})
                    var_st_859hpa = atm_monthly_st[yr][varname].isel({'lev':-7})
                    var_ml_859hpa = atm_monthly_ml[yr][varname].isel({'lev':-7})
                    # Compute annual climatology
                    atm_annual_base_clim[yr][varname] = fun.weighted_temporal_mean(var_base_859hpa).mean(dim='time')
                    atm_annual_base_clim[yr][varname].attrs['units'] = atm_monthly_base[yr][varname].units
                    atm_annual_st_clim[yr][varname] = fun.weighted_temporal_mean(var_st_859hpa).mean(dim='time')
                    atm_annual_st_clim[yr][varname].attrs['units'] = atm_monthly_st[yr][varname].units
                    atm_annual_ml_clim[yr][varname] = fun.weighted_temporal_mean(var_ml_859hpa).mean(dim='time')
                    atm_annual_ml_clim[yr][varname].attrs['units'] = atm_monthly_ml[yr][varname].units
            elif(varname=='U' or varname=='V'):
                    # Subset variable at 859hPa/525 hPa
                    var_base_859hpa = atm_monthly_base[yr][varname].isel({'lev':-7})
                    var_st_859hpa = atm_monthly_st[yr][varname].isel({'lev':-7})
                    var_ml_859hpa = atm_monthly_ml[yr][varname].isel({'lev':-7})
                    # Compute annual climatology
                    atm_annual_base_clim[yr][varname] = fun.weighted_temporal_mean(var_base_859hpa).mean(dim='time')
                    atm_annual_base_clim[yr][varname].attrs['units'] = atm_monthly_base[yr][varname].units
                    atm_annual_st_clim[yr][varname] = fun.weighted_temporal_mean(var_st_859hpa).mean(dim='time')
                    atm_annual_st_clim[yr][varname].attrs['units'] = atm_monthly_st[yr][varname].units
                    atm_annual_ml_clim[yr][varname] = fun.weighted_temporal_mean(var_ml_859hpa).mean(dim='time')
                    atm_annual_ml_clim[yr][varname].attrs['units'] = atm_monthly_ml[yr][varname].units             
            else:
                    # Compute annual climatology
                    atm_annual_base_clim[yr][varname] = fun.weighted_temporal_mean(atm_monthly_base[yr][varname]).mean(dim='time')
                    atm_annual_base_clim[yr][varname].attrs['units'] = atm_monthly_base[yr][varname].units
                    atm_annual_st_clim[yr][varname] = fun.weighted_temporal_mean(atm_monthly_st[yr][varname]).mean(dim='time')
                    atm_annual_st_clim[yr][varname].attrs['units'] = atm_monthly_st[yr][varname].units
                    atm_annual_ml_clim[yr][varname] = fun.weighted_temporal_mean(atm_monthly_ml[yr][varname]).mean(dim='time')
                    atm_annual_ml_clim[yr][varname].attrs['units'] = atm_monthly_ml[yr][varname].units
            # Compute annual anomalies (MCB - CONTROL)
            atm_annual_st_anom[yr][varname] = atm_annual_st_clim[yr][varname] - atm_annual_base_clim[yr][varname]
            atm_annual_st_anom[yr][varname].attrs['units'] = atm_monthly_base[yr][varname].units
            atm_annual_ml_anom[yr][varname] = atm_annual_ml_clim[yr][varname] - atm_annual_base_clim[yr][varname]
            atm_annual_ml_anom[yr][varname].attrs['units'] = atm_monthly_base[yr][varname].units


## 2) DAILY ATMOSPHERE 
# Create empty dictionaries for anomalies
atm_daily_st_anom = {}
atm_daily_ml_anom = {}
## Loop through subsetted varnames list.
print('##ATM DAILY##')
for yr in yr_keys:
    print(yr)
    atm_daily_st_anom[yr] = {}
    atm_daily_ml_anom[yr] = {}
    for varname in atm_varnames_daily_subset:
        print(varname)
        # Compute daily anomalies (MCB - CONTROL)
        atm_daily_st_anom[yr][varname] = atm_daily_st[yr][varname] - atm_daily_base[yr][varname]
        atm_daily_st_anom[yr][varname].attrs['units'] = atm_daily_base[yr][varname].units
        atm_daily_ml_anom[yr][varname] = atm_daily_ml[yr][varname] - atm_daily_base[yr][varname]
        atm_daily_ml_anom[yr][varname].attrs['units'] = atm_daily_base[yr][varname].units



#%% Calculate DAILY apparent temperature (AP)
# Store output in a dictionary
sim_keys = ['ST','ML','CTRL']
ap_max_h1 = {}
### National Weather Service Heat Index (Rothfusz, 1990)
# Populate dictionaries
for yr in yr_keys:
    print(yr)
    ap_max_h1[yr] = {}
    for i in sim_keys:
        if i =='ST':
                in_xr = atm_daily_st[yr]
        elif i =='ML':
                in_xr = atm_daily_ml[yr]
        elif i =='CTRL':
                in_xr = atm_daily_base[yr]
        # Define variables and apply unit conversions
        # Convert deg C to deg F
        T = (in_xr['TREFHTMX']*9/5)+32
        # Leave RH as integer percent
        RH = in_xr['RHREFHT']
        # Calculate simple formula HI
        HI = 0.5 * (T + 61.0 + ((T-68.0)*1.2) + (RH*0.094))
        # If HI >=80 F, apply full regression equation:
        HI_regress = xr.where(HI>=80, (-42.379 + 2.04901523*T + 10.14333127*RH - .22475541*T*RH - .00683783*T*T - .05481717*RH*RH + .00122874*T*T*RH + .00085282*T*RH*RH - .00000199*T*T*RH*RH), HI)
        # Calculate adjustments for the full regression equation
        HI_adjust_sub = xr.where((T>=80)&(T<=120)&(RH<13), ((13-RH)/4)*np.sqrt((17-np.abs(T-95.))/17),0)
        HI_adjust_add = xr.where((T>=80)&(T<=87)&(RH>85), ((RH-85)/10) * ((87-T)/5),0)
        # Apply corrections
        HI_regress-=HI_adjust_sub
        HI_regress+=HI_adjust_add
        # Calculate Ap and covert deg F back to deg C
        ap_max_h1[yr][i] = (HI_regress-32)*5/9
        # Load to get out of dask array 
        ap_max_h1[yr][i].load()


##################################################################################################################
### Population weighted CDFs for West Coast
# Set population for population weighted temperature
# Store counts and sums in dictionary
pop_count = {}
pop_sum_westcoast = {}
# Subset 2010 and 2050 populations
pop_count['2010'] = regrid_pop_count.sel(time=2010).pop_count
pop_count['2050'] = regrid_pop_count_ssp2.pop_count
# ## Uncomment to use the 2010 population for both periods
# pop_count['2050'] = regrid_pop_count.sel(time=2010).pop_count
# Calculate total population for WNA
pop_sum_westcoast['2010'] = np.nansum(pop_count['2010'].where((west_coast_state_mask>0),drop=True))
pop_sum_westcoast['2050'] = np.nansum(pop_count['2050'].where((west_coast_state_mask>0),drop=True))


# Create empty dictionaries
ap_max_h1_westcoast = {}
# Populate dictionary
for yr in yr_keys:
    print(yr)
    ap_max_h1_westcoast[yr] = {}
    pop_count_baseline = pop_count[yr]
    pop_count_westcoast = pop_sum_westcoast[yr]
    for i in sim_keys:      
        ap_max_h1_westcoast[yr][i] = ap_max_h1[yr][i].where((west_coast_state_mask>0),drop=True)


# Summer (JJA) AP histograms
ap_max_h1_westcoast_jja = {}
ap_max_westcoast_jja_flat = {}

for yr in yr_keys:
    print(yr)
    ap_max_h1_westcoast_jja[yr] = {}
    ap_max_westcoast_jja_flat[yr] = {}
    for i in sim_keys:
        print(i)
        ## TREFHTMX
        ap_max_h1_westcoast_jja[yr][i] = ap_max_h1_westcoast[yr][i].loc[{'time':[t for t in ap_max_h1_westcoast[yr][i].time.values if (t.month==6)|(t.month==7)|(t.month==8)]}]
        ap_max_westcoast_jja_flat[yr][i] = np.ndarray.flatten(ap_max_h1_westcoast_jja[yr][i].values)
        print(np.nanmean(ap_max_h1_westcoast_jja[yr][i]))


# Calculate cumulative distributions of extreme temperatures by state
## Absolute temperature values
ap_max_h1_westcoast_jja_bins = {}
ap_max_h1_westcoast_jja_counts = {}
## Anomalous temperatures
ap_max_anom_westcoast_jja_flat = {}
ap_max_anom_westcoast_jja_bins = {}
ap_max_anom_westcoast_jja_counts = {}

for yr in yr_keys:
    print(yr)
    ## Absolute temperature values
    ap_max_h1_westcoast_jja_bins[yr] = {}
    ap_max_h1_westcoast_jja_counts[yr] = {}
    ## Anomalous temperatures
    ap_max_anom_westcoast_jja_flat[yr] = {}
    ap_max_anom_westcoast_jja_bins[yr] = {}
    ap_max_anom_westcoast_jja_counts[yr] = {}
    for i in sim_keys:
        ## TREFHTMX
        ap_max_h1_westcoast_jja_bins[yr][i], ap_max_h1_westcoast_jja_counts[yr][i] = fun.count_cumsum(ap_max_westcoast_jja_flat[yr][i], 0.1, '<')
        ap_max_anom_westcoast_jja_flat[yr][i] = ap_max_westcoast_jja_flat[yr][i] - ap_max_westcoast_jja_flat[yr]['CTRL']
        ap_max_anom_westcoast_jja_bins[yr][i], ap_max_anom_westcoast_jja_counts[yr][i] = fun.count_cumsum(ap_max_anom_westcoast_jja_flat[yr][i], 0.01, '<')
        

## Calculate average number of people exposed to different heat indices per summer
## i.e. how many people are exposed to different heat indices each summer?
caution_peopleperday = {}
caution_peopleday = {}
not_caution_peopleday = {}
extreme_caution_peopleperday = {}
extreme_caution_peopleday = {}
not_extreme_caution_peopleday = {}
danger_peopleperday = {}
danger_peopleday = {}
not_danger_peopleday = {}

# Define exposure as AP ABOVE lower bound of heat index (new as of 2/1/23)
# Bounds (deg C)
caution_lower = 26.67
extreme_caution_lower=32.22
danger_lower=39.44
for yr in yr_keys:
    caution_peopleperday[yr] = {}
    caution_peopleday[yr] = {}
    not_caution_peopleday[yr] = {}
    extreme_caution_peopleperday[yr] = {}
    extreme_caution_peopleday[yr] = {}
    not_extreme_caution_peopleday[yr] = {}
    danger_peopleperday[yr] = {}
    danger_peopleday[yr] = {}
    not_danger_peopleday[yr] = {}
    pop_count_baseline = pop_count[yr]
    for i in sim_keys:
        caution_peopleperday[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=caution_lower)).sum(dim=('lat','lon')).mean(dim='time')
        caution_peopleday[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=caution_lower)).sum(dim=('lat','lon')).groupby('time.year').sum().mean()
        not_caution_peopleday[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]<caution_lower)).sum(dim=('lat','lon')).groupby('time.year').sum().mean()
        extreme_caution_peopleperday[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=extreme_caution_lower)).sum(dim=('lat','lon')).mean(dim='time')
        extreme_caution_peopleday[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=extreme_caution_lower)).sum(dim=('lat','lon')).groupby('time.year').sum().mean()
        not_extreme_caution_peopleday[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]<extreme_caution_lower)).sum(dim=('lat','lon')).groupby('time.year').sum().mean()
        danger_peopleperday[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=danger_lower)).sum(dim=('lat','lon')).mean(dim='time')
        danger_peopleday[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=danger_lower)).sum(dim=('lat','lon')).groupby('time.year').sum().mean()
        not_danger_peopleday[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]<danger_lower)).sum(dim=('lat','lon')).groupby('time.year').sum().mean()


## Calculate the pixel-level average number of people exposed to different heat indices per summer for MAPS
## i.e. how many people are exposed to different heat indices each summer?
caution_peopleday_pixel = {}
extreme_caution_peopleday_pixel = {}
danger_peopleday_pixel = {}
caution_peopleday_pixel_daily = {}
extreme_caution_peopleday_pixel_daily = {}
danger_peopleday_pixel_daily = {}
# Define exposure as AP ABOVE lower bound of heat index (new as of 2/1/23)
for yr in yr_keys:
    caution_peopleday_pixel[yr] = {}
    extreme_caution_peopleday_pixel[yr] = {}
    danger_peopleday_pixel[yr] = {}
    caution_peopleday_pixel_daily[yr] = {}
    extreme_caution_peopleday_pixel_daily[yr] = {}
    danger_peopleday_pixel_daily[yr] = {}
    pop_count_baseline = pop_count[yr]
    for i in sim_keys:
        # Compute annual averages
        caution_peopleday_pixel[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=caution_lower)).groupby('time.year').sum().mean(dim='year')
        extreme_caution_peopleday_pixel[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=extreme_caution_lower)).groupby('time.year').sum().mean(dim='year')
        danger_peopleday_pixel[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=danger_lower)).groupby('time.year').sum().mean(dim='year')
        # Compute daily values for significance testings
        caution_peopleday_pixel_daily[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=caution_lower)).groupby('time.year').sum().fillna(0)
        extreme_caution_peopleday_pixel_daily[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=extreme_caution_lower)).groupby('time.year').sum().fillna(0)
        danger_peopleday_pixel_daily[yr][i] = pop_count_baseline.where((ap_max_h1_westcoast_jja[yr][i]>=danger_lower)).groupby('time.year').sum().fillna(0)
        # Assign units
        caution_peopleday_pixel[yr][i].attrs['units'] = 'people-days'
        extreme_caution_peopleday_pixel[yr][i].attrs['units'] = 'people-days'
        danger_peopleday_pixel[yr][i].attrs['units'] = 'people-days'


# Calculate significance at the pixel level
caution_peopleday_pixel_pval = {}
extreme_caution_peopleday_pixel_pval = {}
danger_peopleday_pixel_pval = {}
for yr in yr_keys:
    caution_peopleday_pixel_pval[yr] = {}
    extreme_caution_peopleday_pixel_pval[yr] = {}
    danger_peopleday_pixel_pval[yr] = {}
    for i in sim_keys[:-1]:
        # Caution
        tmp_base = caution_peopleday_pixel_daily[yr]['CTRL']
        tmp_mcb = caution_peopleday_pixel_daily[yr][i]
        caution_peopleday_pixel_pval[yr][i] = stats.ttest_rel(tmp_base,tmp_mcb,axis=-1).pvalue
        # Extreme Caution
        tmp_base = extreme_caution_peopleday_pixel_daily[yr]['CTRL']
        tmp_mcb = extreme_caution_peopleday_pixel_daily[yr][i]
        extreme_caution_peopleday_pixel_pval[yr][i] = stats.ttest_rel(tmp_base,tmp_mcb,axis=-1).pvalue        
        # Danger
        tmp_base = danger_peopleday_pixel_daily[yr]['CTRL']
        tmp_mcb = danger_peopleday_pixel_daily[yr][i]
        danger_peopleday_pixel_pval[yr][i] = stats.ttest_rel(tmp_base,tmp_mcb,axis=-1).pvalue


# Calculate risk variables (absolute risk reduction, relative risk, relative effect, odds ratio)
# Save relative effect output for plotting. Print everything else to display
re_df = pd.DataFrame(columns=['Year','MCB','Heat_Index','Relative_effect'])
rr_df = pd.DataFrame(columns=['Year','MCB','Heat_Index','Relative_risk'])
for yr in yr_keys:
    print(yr)
    # Loop through MCB cases (drop control)
    for i in sim_keys[:2]:
        # Caution heat index
        print('##CAUTION##')
        print(i)
        caution_total_peopleday_mcb = caution_peopleday[yr][i] + not_caution_peopleday[yr][i]
        caution_total_peopleday_ctrl = caution_peopleday[yr]['CTRL'] + not_caution_peopleday[yr]['CTRL']
        caution_er_mcb = caution_peopleday[yr][i]/caution_total_peopleday_mcb
        caution_er_ctrl = caution_peopleday[yr]['CTRL']/caution_total_peopleday_ctrl
        arr = caution_er_ctrl-caution_er_mcb
        rr = caution_er_mcb/caution_er_ctrl
        re = rr-1
        odds_ratio = (caution_peopleday[yr][i] / not_caution_peopleday[yr][i]) /( caution_peopleday[yr]['CTRL'] / not_caution_peopleday[yr]['CTRL'])
        print('ARR=',round((float(arr)*100),4),'%')
        print('RR=',round(float(rr),4))
        print('RE=',round((float(re)*100),4),'%')
        print('OR=',round(float(odds_ratio),4))
        re_entry = pd.DataFrame({'Year':yr,'MCB':i,'Heat_Index':'Caution','Relative_effect':(float(re)*100)},index=[0])
        re_df = pd.concat([re_df,re_entry], ignore_index=True)
        rr_entry = pd.DataFrame({'Year':yr,'MCB':i,'Heat_Index':'Caution','Relative_risk':(float(rr))},index=[0])
        rr_df = pd.concat([rr_df,rr_entry], ignore_index=True)        
        # Extreme caution heat index
        print('##EXTREME CAUTION##')
        print(i)
        extreme_caution_total_peopleday_mcb = extreme_caution_peopleday[yr][i] + not_extreme_caution_peopleday[yr][i]
        extreme_caution_total_peopleday_ctrl = extreme_caution_peopleday[yr]['CTRL'] + not_extreme_caution_peopleday[yr]['CTRL']
        extreme_caution_er_mcb = extreme_caution_peopleday[yr][i]/extreme_caution_total_peopleday_mcb
        extreme_caution_er_ctrl = extreme_caution_peopleday[yr]['CTRL']/extreme_caution_total_peopleday_ctrl
        arr = extreme_caution_er_ctrl-extreme_caution_er_mcb
        rr = extreme_caution_er_mcb/extreme_caution_er_ctrl
        re = rr-1
        odds_ratio = (extreme_caution_peopleday[yr][i] / not_extreme_caution_peopleday[yr][i]) /( extreme_caution_peopleday[yr]['CTRL'] / not_extreme_caution_peopleday[yr]['CTRL'])
        print('ARR=',round((float(arr)*100),4),'%')
        print('RR=',round(float(rr),4))
        print('RE=',round((float(re)*100),4),'%')
        print('OR=',round(float(odds_ratio),4))
        re_entry = pd.DataFrame({'Year':yr,'MCB':i,'Heat_Index':'Extreme Caution','Relative_effect':(float(re)*100)},index=[0])
        re_df = pd.concat([re_df,re_entry], ignore_index=True)
        rr_entry = pd.DataFrame({'Year':yr,'MCB':i,'Heat_Index':'Extreme Caution','Relative_risk':(float(rr))},index=[0])
        rr_df = pd.concat([rr_df,rr_entry], ignore_index=True)    
        # Danger heat index
        print('##DANGER##')
        print(i)
        danger_total_peopleday_mcb = danger_peopleday[yr][i] + not_danger_peopleday[yr][i]
        danger_total_peopleday_ctrl = danger_peopleday[yr]['CTRL'] + not_danger_peopleday[yr]['CTRL']
        danger_er_mcb = danger_peopleday[yr][i]/danger_total_peopleday_mcb
        danger_er_ctrl = danger_peopleday[yr]['CTRL']/danger_total_peopleday_ctrl
        arr = danger_er_ctrl-danger_er_mcb
        rr = danger_er_mcb/danger_er_ctrl
        re = rr-1
        odds_ratio = (danger_peopleday[yr][i] / not_danger_peopleday[yr][i]) /( danger_peopleday[yr]['CTRL'] / not_danger_peopleday[yr]['CTRL'])
        print('ARR=',round((float(arr)*100),4),'%')
        print('RR=',round(float(rr),4))
        print('RE=',round((float(re)*100),4),'%')
        print('OR=',round(float(odds_ratio),4))
        re_entry = pd.DataFrame({'Year':yr,'MCB':i,'Heat_Index':'Danger','Relative_effect':(float(re)*100)},index=[0])
        re_df = pd.concat([re_df,re_entry], ignore_index=True)
        rr_entry = pd.DataFrame({'Year':yr,'MCB':i,'Heat_Index':'Danger','Relative_risk':(float(rr))},index=[0])
        rr_df = pd.concat([rr_df,rr_entry], ignore_index=True)  


##################################################################################################################
#%% PLOT FIGURES
## FIGURE 2
fmt1 = ticker.ScalarFormatter(useMathText=True)
fmt1.set_powerlimits((0, 0))
fmt2 = ticker.ScalarFormatter(useMathText=True)
fmt2.set_powerlimits((0, 0))

fig = plt.figure(figsize=(18,10),constrained_layout=True);
gs = fig.add_gridspec(ncols=6,nrows=2);
# Set min/max limits
abs_min = 0
abs_max = 1e7
delta_min = -2e5
delta_max = 2e5
### B. 2010 No MCB
yr='2010'
f1_ax2 = fig.add_subplot(gs[0,3],projection=ccrs.PlateCarree(central_longitude=0),transform=ccrs.PlateCarree(central_longitude=1080),frameon=False)
ax2,p1 = fun.plot_west_coast_panel_maps(in_xr=danger_peopleday_pixel[yr]['CTRL'].where((west_coast_state_mask>0),drop=True), cmin=abs_min, cmax=abs_max, ccmap='hot_r', colorbar=False,central_lon=0,\
                    nrow=2,ncol=3,mean_val='none')
plt.text(0, 0,'b',transform=f1_ax2.transAxes,fontsize=10, fontweight='bold');
plt.text(0,1,'No MCB',transform=f1_ax2.transAxes,fontsize=7, color='grey');
### C. 2010 ML MCB
f1_ax3 = fig.add_subplot(gs[0,4],projection=ccrs.PlateCarree(central_longitude=0),transform=ccrs.PlateCarree(central_longitude=0),frameon=False)
in_xr = danger_peopleday_pixel[yr]['ML']-danger_peopleday_pixel[yr]['CTRL']
in_xr.attrs['units']='$\Delta$(people-days)'
ax3,p2 = fun.plot_west_coast_panel_maps(in_xr=in_xr.where((west_coast_state_mask>0),drop=True), cmin=delta_min, cmax=delta_max, ccmap='PiYG_r', colorbar=False, central_lon=0,\
                    nrow=2,ncol=3,mean_val='none')
plt.text(0,0,'c',transform=f1_ax3.transAxes,fontsize=10, fontweight='bold');
plt.text(0,1,'Mid-latitude MCB',transform=f1_ax3.transAxes,fontsize=7, color='grey');
### D. 2010 ST MCB
f1_ax4 = fig.add_subplot(gs[0,5],projection=ccrs.PlateCarree(central_longitude=0),transform=ccrs.PlateCarree(central_longitude=0),frameon=False)
in_xr = danger_peopleday_pixel[yr]['ST']-danger_peopleday_pixel[yr]['CTRL']
in_xr.attrs['units']='$\Delta$(people-days)'
fun.plot_west_coast_panel_maps(in_xr=in_xr.where((west_coast_state_mask>0),drop=True), cmin=delta_min, cmax=delta_max, ccmap='PiYG_r', colorbar=False, central_lon=0,\
                    nrow=2,ncol=3,mean_val='none')
plt.text(0,0,'d',transform=f1_ax4.transAxes,fontsize=10, fontweight='bold');
plt.text(.9,0.05,'2010',transform=f1_ax4.transAxes,fontsize=12, fontweight='bold',rotation=270,color='grey');
plt.text(0,1,'Subtropical MCB',transform=f1_ax4.transAxes,fontsize=7, color='grey');
### E. 2050 No MCB
yr='2050'
f1_ax5 = fig.add_subplot(gs[1,3],projection=ccrs.PlateCarree(central_longitude=0),transform=ccrs.PlateCarree(central_longitude=0),frameon=False)
fun.plot_west_coast_panel_maps(in_xr=danger_peopleday_pixel[yr]['CTRL'].where((west_coast_state_mask>0),drop=True), cmin=abs_min, cmax=abs_max, ccmap='hot_r', colorbar=False,central_lon=0,\
                    nrow=2,ncol=3,mean_val='none')
plt.text(0,0,'e',transform=f1_ax5.transAxes,fontsize=10, fontweight='bold');
### F. 2050 ML MCB
f1_ax6 = fig.add_subplot(gs[1,4],projection=ccrs.PlateCarree(central_longitude=0),transform=ccrs.PlateCarree(central_longitude=0),frameon=False)
in_xr = danger_peopleday_pixel[yr]['ML']-danger_peopleday_pixel[yr]['CTRL']
in_xr.attrs['units']='$\Delta$(people-days)'
fun.plot_west_coast_panel_maps(in_xr=in_xr.where((west_coast_state_mask>0),drop=True), cmin=delta_min, cmax=delta_max, ccmap='PiYG_r', colorbar=False, central_lon=0,\
                    nrow=2,ncol=3,mean_val='none')
plt.text(0,0,'f',transform=f1_ax6.transAxes,fontsize=10, fontweight='bold');
### G. 2050 ST MCB
f1_ax7 = fig.add_subplot(gs[1,5],projection=ccrs.PlateCarree(central_longitude=0),transform=ccrs.PlateCarree(central_longitude=0),frameon=False)
in_xr = danger_peopleday_pixel[yr]['ST']-danger_peopleday_pixel[yr]['CTRL']
in_xr.attrs['units']='$\Delta$(people-days)'
fun.plot_west_coast_panel_maps(in_xr=in_xr.where((west_coast_state_mask>0),drop=True), cmin=delta_min, cmax=delta_max, ccmap='PiYG_r', colorbar=False, central_lon=0,\
                    nrow=2,ncol=3,mean_val='none')
plt.text(0,0,'g',transform=f1_ax7.transAxes,fontsize=10, fontweight='bold');
plt.text(.9,0.05,'2050',transform=f1_ax7.transAxes,fontsize=12, fontweight='bold',rotation=270,color='grey');

### PAUSE TO LET GRAPHICS LOAD ###

### A. Relative risk scatter
f1_ax1 = fig.add_subplot(gs[0:,0:3],frameon=False);
marker_size=100
# yr_keys=['2010'] # uncomment for 2010 scatter only for presentations
for yr in yr_keys:
    tmp_df = rr_df[rr_df['Year']==yr]
    tmp_ml = tmp_df[tmp_df['MCB']=='ML']
    tmp_st = tmp_df[tmp_df['MCB']=='ST']
    if yr=='2010':
        plt.scatter(tmp_ml['Relative_risk'],np.arange(0.5,3,1),c='orange',s=marker_size,label=yr+' ML MCB')
        plt.scatter(tmp_st['Relative_risk'],np.arange(0.5,3,1),marker = '^',c='darkblue',s=marker_size,label=yr+' ST MCB')
    elif yr=='2050':
         plt.scatter(tmp_ml['Relative_risk'],np.arange(0.5,3,1),edgecolors='orange',facecolors='none',s=marker_size,label=yr+' ML MCB')
         plt.scatter(tmp_st['Relative_risk'],np.arange(0.5,3,1),marker = '^',edgecolors='darkblue',facecolors='none',s=marker_size,label=yr+' ST MCB')
plt.xlim(0.32,1.1);
plt.ylim(0,3);
plt.xlabel('Relative Risk (MCB/No MCB)',fontsize=7);
plt.axvline(1,ymin=0,ymax=1, linestyle='--', linewidth=2, c='k');
plt.xticks(fontsize=7);
f1_ax1.legend([Line2D([0], [0],color='orange',fillstyle='left',linestyle='None',marker='o',markersize=8),\
            Line2D([0], [0],color='k',fillstyle='left',linestyle='None',marker='s',markersize=8),\
            Line2D([0], [0],color='darkblue',fillstyle='left',linestyle='None',marker='^',markersize=8)],\
            ['Mid-latitude MCB','2010/2050', 'Subtropical MCB'],bbox_to_anchor =(0.5,-0.23), loc='lower center',\
        ncol=2, fancybox=False, shadow=False,frameon=False,fontsize=7);
xmin, xmax = f1_ax1.get_xaxis().get_view_interval();
ymin, ymax = f1_ax1.get_yaxis().get_view_interval();
f1_ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=1));
# Turn off yaxis
f1_ax1.axes.get_yaxis().set_visible(False);
plt.tick_params(left = False);
# Add Caution section
left, bottom, width, height = (xmin, ymin, (xmax-xmin), 1)
plt.axhline(1, linestyle='-',linewidth=3, c='grey',alpha=0.5);
# add text with text() function in matplotlib
plt.text(xmin, bottom+0.1,'Caution',fontsize=10, color='k',fontweight='bold');
# Add Extreme Caution section
left, bottom, width, height = (xmin, height+ymin, (xmax-xmin), 1)
plt.axhline(2, linestyle='-',linewidth=3, c='grey',alpha=0.5);
# add text with text() function in matplotlib
plt.text(xmin, bottom+0.1,'Extreme Caution',fontsize=10, color='k',fontweight='bold');
# Add Danger Box
left, bottom, width, height = (xmin, bottom+height, (xmax-xmin), 1)
# add text with text() function in matplotlib
plt.text(xmin, bottom+0.1,'Danger',fontsize=10, color='k',fontweight='bold');
plt.text(xmin, bottom+height-0.1,'a',fontsize=10,fontweight='bold');
## Add figure colorbars
## NUMBER
cbar_ax = fig.add_axes([0.49, 0.1, 0.15, 0.025]) #rect kwargs [left, bottom, width, height];
cbar = fig.colorbar(p1, cax=cbar_ax,orientation='horizontal', extend='both',format=fmt1,pad=0.1);
cbar.ax.xaxis.get_offset_text().set_fontsize(7)
cbar.ax.tick_params(labelsize=7); cbar.set_label(label=danger_peopleday_pixel[yr]['CTRL'].units, size=7, loc = 'left')
## CHANGE
cbar_ax = fig.add_axes([0.65, 0.1, 0.33, 0.025]) #rect kwargs [left, bottom, width, height];
cbar = fig.colorbar(p2, cax=cbar_ax,orientation='horizontal', extend='both',format=fmt2,pad=0.1);
cbar.ax.xaxis.get_offset_text().set_fontsize(7);
cbar.ax.tick_params(labelsize=7); cbar.set_label(label=in_xr.units, size=7)



#%% REMOTE IMPACTS
# Store output in a dictionary
sim_keys = ['ST','ML','CTRL']
prect_h1 = {}
# Populate dictionaries
for yr in yr_keys:
    print(yr)
    prect_h1[yr] = {}
    for i in sim_keys:
        print(i)
        if i =='ST':
                # atm_in_xr = atm_daily_st
                atm_in_xr = atm_monthly_st[yr]
        elif i =='ML':
                # atm_in_xr = atm_daily_ml
                atm_in_xr = atm_monthly_ml[yr]
        elif i =='CTRL':
                # atm_in_xr = atm_daily_base
                atm_in_xr = atm_monthly_base[yr]
        PRECT = atm_in_xr['PRECT']
        # PRECT (mm/day to mm/month)
        # Create dictionary with number of days for each month (no leap calendar)
        month_day_dict = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
        # Make a new xarrays so we don't overwrite the original for now
        PRECT_converted = PRECT*1.0
        for mon in list(month_day_dict.keys()):
            tmp_month_day = month_day_dict[mon]
            PRECT_converted.loc[{'time':[t for t in PRECT_converted.time.values if t.month==mon]}]*=tmp_month_day
        # Compute aridity index (AI) at the annual level
        prect_h1[yr][i]  = PRECT_converted.groupby('time.year').sum()
        
# Mask out ocean grid cells for apparent temperature and aridity indices
ap_max_h1_jja_land = {}
ap_max_h1_djf_land = {}
ap_max_h1_land = {}
prect_h1_land = {}

for yr in yr_keys:
    print(yr)
    ap_max_h1_jja_land[yr] = {}
    ap_max_h1_djf_land[yr] = {}
    ap_max_h1_land[yr] = {}
    prect_h1_land[yr] = {}
    for i in sim_keys:
            # Mask out ocean
            ap_max_h1_land[yr][i] = xr.where(atm_monthly_base_clim[yr]['LANDFRAC'].mean(dim='month')>=0.1,ap_max_h1[yr][i],np.nan)
            ap_max_h1_jja_land[yr][i] = xr.where(atm_monthly_base_clim[yr]['LANDFRAC'].mean(dim='month')>=0.1,ap_max_h1[yr][i].loc[{'time':[t for t in ap_max_h1[yr][i].time.values if (t.month==6)|(t.month==7)|(t.month==8)]}],np.nan)
            ap_max_h1_djf_land[yr][i] = xr.where(atm_monthly_base_clim[yr]['LANDFRAC'].mean(dim='month')>=0.1,ap_max_h1[yr][i].loc[{'time':[t for t in ap_max_h1[yr][i].time.values if (t.month==1)|(t.month==2)|(t.month==12)]}],np.nan)
            prect_h1_land[yr][i] = xr.where(atm_monthly_base_clim[yr]['LANDFRAC'].mean(dim='month')>=0.1,prect_h1[yr][i],np.nan)
            # Assign units
            ap_max_h1_land[yr][i].attrs['units'] = '${\N{DEGREE SIGN}}$C'
            ap_max_h1_jja_land[yr][i].attrs['units'] = '${\N{DEGREE SIGN}}$C'
            ap_max_h1_djf_land[yr][i].attrs['units'] = '${\N{DEGREE SIGN}}$C'
            prect_h1_land[yr][i].attrs['units'] = 'mm/yr'

# Compute AP anomalies
ap_max_anom_h1_jja_land = {}
ap_max_anom_h1_djf_land = {}
ap_max_anom_h1_land = {}
prect_anom_h1_land = {}

for yr in yr_keys:
    print(yr)
    ap_max_anom_h1_jja_land[yr] = {}
    ap_max_anom_h1_djf_land[yr] = {}
    ap_max_anom_h1_land[yr] = {}
    prect_anom_h1_land[yr] = {}
    for i in sim_keys[0:-1]:
            ap_max_anom_h1_land[yr][i] = ap_max_h1_land[yr][i]-ap_max_h1_land[yr]['CTRL']
            ap_max_anom_h1_jja_land[yr][i] = ap_max_h1_jja_land[yr][i]-ap_max_h1_jja_land[yr]['CTRL']
            ap_max_anom_h1_djf_land[yr][i] = ap_max_h1_djf_land[yr][i]-ap_max_h1_djf_land[yr]['CTRL']
            prect_anom_h1_land[yr][i] = prect_h1_land[yr][i]-prect_h1_land[yr]['CTRL']
            # Assign units
            ap_max_anom_h1_land[yr][i].attrs['units'] = '${\N{DEGREE SIGN}}$C'
            ap_max_anom_h1_jja_land[yr][i].attrs['units'] = '${\N{DEGREE SIGN}}$C'
            ap_max_anom_h1_djf_land[yr][i].attrs['units'] = '${\N{DEGREE SIGN}}$C'
            prect_anom_h1_land[yr][i].attrs['units'] = 'mm/yr'


# Calculate p-values for AP anomalies
ap_max_h1_land_st_pval = {}
ap_max_h1_land_ml_pval = {}
ap_max_h1_jja_land_st_pval = {}
ap_max_h1_jja_land_ml_pval = {}
ap_max_h1_djf_land_st_pval = {}
ap_max_h1_djf_land_ml_pval = {}
prect_h1_land_st_pval = {}
prect_h1_land_ml_pval = {}
for yr in yr_keys:
    print(yr)
    ap_max_h1_land_st_pval[yr] = {}
    ap_max_h1_land_ml_pval[yr]  = {}
    ap_max_h1_jja_land_st_pval[yr]  = {}
    ap_max_h1_jja_land_ml_pval[yr]  = {}
    prect_h1_land_st_pval[yr] = {}
    prect_h1_land_ml_pval[yr] = {}
    # AP Max Annual
    tmp_base = ap_max_h1_land[yr]['CTRL']
    tmp_st = ap_max_h1_land[yr]['ST']
    tmp_st_pval = stats.ttest_rel(tmp_base,tmp_st,axis=-1).pvalue
    tmp_ml = ap_max_h1_land[yr]['ML']
    tmp_ml_pval = stats.ttest_rel(tmp_base,tmp_ml,axis=-1).pvalue
    ap_max_h1_land_st_pval[yr] = tmp_st_pval
    ap_max_h1_land_ml_pval[yr] = tmp_ml_pval
    # AP Max JJA
    tmp_base = ap_max_h1_jja_land[yr]['CTRL']
    tmp_st = ap_max_h1_jja_land[yr]['ST']
    tmp_st_pval = stats.ttest_rel(tmp_base,tmp_st,axis=-1).pvalue
    tmp_ml = ap_max_h1_jja_land[yr]['ML']
    tmp_ml_pval = stats.ttest_rel(tmp_base,tmp_ml,axis=-1).pvalue
    ap_max_h1_jja_land_st_pval[yr] = tmp_st_pval
    ap_max_h1_jja_land_ml_pval[yr] = tmp_ml_pval
    # AP Max DJF
    tmp_base = ap_max_h1_djf_land[yr]['CTRL']
    tmp_st = ap_max_h1_djf_land[yr]['ST']
    tmp_st_pval = stats.ttest_rel(tmp_base,tmp_st,axis=-1).pvalue
    tmp_ml = ap_max_h1_djf_land[yr]['ML']
    tmp_ml_pval = stats.ttest_rel(tmp_base,tmp_ml,axis=-1).pvalue
    ap_max_h1_djf_land_st_pval[yr] = tmp_st_pval
    ap_max_h1_djf_land_ml_pval[yr] = tmp_ml_pval
    # Precipitation
    tmp_base = prect_h1_land[yr]['CTRL']
    tmp_st = prect_h1_land[yr]['ST']
    tmp_st_pval = stats.ttest_rel(tmp_base,tmp_st,axis=-1).pvalue
    tmp_ml = prect_h1_land[yr]['ML']
    tmp_ml_pval = stats.ttest_rel(tmp_base,tmp_ml,axis=-1).pvalue
    prect_h1_land_st_pval[yr] = tmp_st_pval
    prect_h1_land_ml_pval[yr] = tmp_ml_pval


### FIGURE 3: max JJA AP and annual PRECIP
label_vec = ['a','b','c','d']
## Global (ROBINSON)
for i in sim_keys[0:-1]: 
    fig = plt.figure(figsize=(18,12));
    subplot_num = 0
    if i =='ST':
            ci_in_ap = ap_max_h1_jja_land_st_pval
            ci_in_prect = prect_h1_land_st_pval
    elif i=='ML':
            ci_in_ap = ap_max_h1_jja_land_ml_pval
            ci_in_prect = prect_h1_land_ml_pval
    for yr in yr_keys[::-1]:
        # AP
        in_xr = ap_max_anom_h1_jja_land[yr][i].mean(dim='time')
        in_xr.attrs['units'] = ap_max_anom_h1_jja_land[yr][i].units
        fun.plot_panel_maps(in_xr=in_xr, cmin=-2, cmax=2, ccmap='RdBu_r', plot_zoom='global', central_lon=0,\
                                CI_in=ci_in_ap[yr],CI_level=0.05,CI_display='inv_stipple', projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,\
                                mean_val='none')
        plt.title(label_vec[subplot_num],fontsize=10, fontweight='bold',loc='left');
        subplot_num += 1
        # PRECT
        in_xr = prect_anom_h1_land[yr][i].mean(dim='year')
        in_xr.attrs['units'] = 'mm/yr'
        fun.plot_panel_maps(in_xr=in_xr, cmin=-150, cmax=150, ccmap='BrBG', plot_zoom='global', central_lon=0,\
                                CI_in=ci_in_prect[yr],CI_level=0.05,CI_display='inv_stipple', projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,\
                                mean_val='none')
        plt.title(label_vec[subplot_num],fontsize=10, fontweight='bold',loc='left');
        subplot_num += 1
    ## Save figure and close
    fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.1,hspace=0.1);
   