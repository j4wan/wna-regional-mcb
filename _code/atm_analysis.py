### PURPOSE: Main script to analyze and plot CESM2 atm output
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 05/06/2024

##################################################################################################################
#%% IMPORT LIBRARIES, DATA, AND FORMAT
# Import libraries
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
import imageio
# turn interactive plotting on/off
plt.ion();

## Create dictionaries to hold both 2010 and 2050 output
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


##################################################################################################################
#%% COMPUTE CLIMATOLOGIES AND ANOMALIES FOR SELECT VARIABLES
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
atm_monthly_ctrl_anom = {}
## Loop through  varnames list
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
## Add on for 2050-2010 anomalies
for varname in atm_varnames_monthly_subset:
        atm_monthly_ctrl_anom[varname] = atm_monthly_base_clim['2050'][varname] - atm_monthly_base_clim['2010'][varname]
        atm_monthly_ctrl_anom[varname].attrs['units'] = atm_monthly_base[yr][varname].units


## 1b) ANNUAL ATMOSPHERE (FROM MONTHLY DATA)
# Create empty dictionaries for climatologies and anomalies
atm_annual_base_clim = {}
atm_annual_st_clim = {}
atm_annual_ml_clim = {}
atm_annual_st_anom = {}
atm_annual_ml_anom = {}
atm_annual_ctrl_anom = {}
## Loop through varnames list
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
                    # Subset variable at 859hPa
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
## Add on for 2050-2010 anomalies
for varname in atm_varnames_monthly_subset:
        atm_annual_ctrl_anom[varname] = atm_annual_base_clim['2050'][varname] - atm_annual_base_clim['2010'][varname]
        atm_annual_ctrl_anom[varname].attrs['units'] = atm_annual_base_clim[yr][varname].units


## 2) DAILY ATMOSPHERE 
# Create empty dictionaries for anomalies
atm_daily_st_anom = {}
atm_daily_ml_anom = {}
atm_daily_ctrl_anom = {}
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
## Add on for 2050-2010 anomalies
for varname in atm_varnames_daily_subset:
        atm_daily_ctrl_anom[varname] = atm_daily_base['2050'][varname] - atm_daily_base['2010'][varname]
        atm_daily_ctrl_anom[varname].attrs['units'] = atm_daily_base[yr][varname].units


## Calculate statistically significant pixels (Student's t-test)
# Set variable list to loop through (speed up code by cutting out variables not involved)
pval_varlist = ['CLDLOW','CLDLIQ','PSL','CDNUMC','TREFHT','FSNTOA','SHFLX']
## MONTHLY ATMOSPHERE
atm_monthly_ml_clim_pval = {}
atm_monthly_st_clim_pval = {}
atm_monthly_ctrl_clim_pval = {}
i_month = np.arange(1,13,1)
for yr in yr_keys:
        print(yr)
        atm_monthly_ml_clim_pval[yr] = {}
        atm_monthly_st_clim_pval[yr] = {}
        atm_monthly_ctrl_clim_pval[yr] = {}
        for varname in pval_varlist:
                print(varname)
                tmp_ml_array = np.full((np.shape(atm_monthly_base_clim[yr]['TREFHT'])), fill_value=np.nan, dtype=np.float64)
                tmp_st_array = np.full((np.shape(atm_monthly_base_clim[yr]['TREFHT'])), fill_value=np.nan, dtype=np.float64)
                tmp_ctrl_array = np.full((np.shape(atm_monthly_base_clim[yr]['TREFHT'])), fill_value=np.nan, dtype=np.float64)
                if (varname=='CCN3' or varname=='CDNUMC' or varname=='RELHUM'):
                        # Subset variable at 859.5 hPa
                        var_base = atm_monthly_base[yr][varname].isel({'lev':-7})
                        var_st = atm_monthly_st[yr][varname].isel({'lev':-7})
                        var_ml = atm_monthly_ml[yr][varname].isel({'lev':-7})
                        var_base_2050 = atm_monthly_base['2050'][varname].isel({'lev':-7})
                elif (varname=='U' or varname=='V'):
                        # Subset variable at 859hPa
                        var_base = atm_monthly_base[yr][varname].isel({'lev':-7})
                        var_st = atm_monthly_st[yr][varname].isel({'lev':-7})
                        var_ml = atm_monthly_ml[yr][varname].isel({'lev':-7})
                        var_base_2050 = atm_monthly_base['2050'][varname].isel({'lev':-7})
                else:
                        var_base = atm_monthly_base[yr][varname]
                        var_st = atm_monthly_st[yr][varname]
                        var_ml = atm_monthly_ml[yr][varname]
                        var_base_2050 = atm_monthly_base['2050'][varname]
                for i in i_month:
                        tmp_base = var_base.loc[{'time':[t for t in atm_monthly_base[yr].time.values if t.month==i]}]
                        tmp_st = var_st.loc[{'time':[t for t in atm_monthly_st[yr].time.values if t.month==i]}]
                        tmp_st_pval = stats.ttest_rel(tmp_base,tmp_st,axis=0).pvalue
                        tmp_st_array[int(i-1),:,:] = tmp_st_pval
                        tmp_ml = var_ml.loc[{'time':[t for t in atm_monthly_ml[yr].time.values if t.month==i]}]
                        tmp_ml_pval = stats.ttest_rel(tmp_base,tmp_ml,axis=0).pvalue
                        tmp_ml_array[int(i-1),:,:] = tmp_ml_pval
                        tmp_base_2050 = var_base_2050.loc[{'time':[t for t in atm_monthly_base['2050'].time.values if t.month==i]}]
                        tmp_base_2050_pval = stats.ttest_rel(tmp_base,tmp_base_2050,axis=0).pvalue
                        tmp_ctrl_array[int(i-1),:,:] = tmp_base_2050_pval
                atm_monthly_st_clim_pval[yr][varname] = tmp_st_array
                atm_monthly_ml_clim_pval[yr][varname] = tmp_ml_array
                atm_monthly_ctrl_clim_pval[yr][varname] = tmp_ctrl_array


## ANNUAL ATMOSPHERE
atm_annual_ml_clim_pval = {}
atm_annual_st_clim_pval = {}
atm_annual_ctrl_clim_pval = {}
# Loop through subsetted varnames list
for yr in yr_keys:
        print(yr)
        atm_annual_ml_clim_pval[yr] = {}
        atm_annual_st_clim_pval[yr] = {}
        atm_annual_ctrl_clim_pval[yr] = {}
        for varname in pval_varlist:
                print(varname)
                if (varname=='CCN3' or varname=='CDNUMC' or varname=='RELHUM'):
                        # Subset variable at 859.5 hPa
                        var_base = atm_monthly_base[yr][varname].isel({'lev':-7})
                        var_st = atm_monthly_st[yr][varname].isel({'lev':-7})
                        var_ml = atm_monthly_ml[yr][varname].isel({'lev':-7})
                        var_base_2050 = atm_monthly_base['2050'][varname].isel({'lev':-7})
                elif (varname=='U' or varname=='V'):
                        # Subset variable at 859hPa
                        var_base = atm_monthly_base[yr][varname].isel({'lev':-7})
                        var_st = atm_monthly_st[yr][varname].isel({'lev':-7})
                        var_ml = atm_monthly_ml[yr][varname].isel({'lev':-7}) 
                        var_base_2050 = atm_monthly_base['2050'][varname].isel({'lev':-7})                       
                else:
                        var_base = atm_monthly_base[yr][varname]
                        var_st = atm_monthly_st[yr][varname]
                        var_ml = atm_monthly_ml[yr][varname]
                        var_base_2050 = atm_monthly_base['2050'][varname]
                tmp_base = var_base.groupby('time.year').mean()
                tmp_st = var_st.groupby('time.year').mean()
                tmp_st_pval = stats.ttest_rel(tmp_base,tmp_st,axis=0).pvalue
                tmp_ml = var_ml.groupby('time.year').mean()
                tmp_ml_pval = stats.ttest_rel(tmp_base,tmp_ml,axis=0).pvalue
                tmp_base_2050 = var_base_2050.groupby('time.year').mean()
                tmp_base_2050_pval = stats.ttest_rel(tmp_base,tmp_base_2050,axis=0).pvalue
                atm_annual_st_clim_pval[yr][varname] = tmp_st_pval
                atm_annual_ml_clim_pval[yr][varname] = tmp_ml_pval
                atm_annual_ctrl_clim_pval[yr][varname] = tmp_base_2050_pval



##################################################################################################################
#%% GLOBAL MEAN TEMPERATURE CALCULATIONS
### CALCULATE GLOBAL AREA WEIGHTED MEANS AND STANDARD DEVIATIONS
atm_monthly_st_anom_global_val = {}
atm_monthly_ml_anom_global_val = {}
atm_monthly_ctrl_anom_global_val = {}

# Loop through subsetted varnames list
for yr in yr_keys:
        print(yr)
        atm_monthly_ml_anom_global_val[yr] = {}
        atm_monthly_st_anom_global_val[yr] = {}
        atm_monthly_ctrl_anom_global_val[yr] = {}
        for varname in atm_varnames_monthly_subset:
                if (varname=='CCN3' or varname=='CDNUMC' or varname=='RELHUM'):
                        # Subset variable at 859.5 hPa
                        var_base = atm_monthly_base[yr][varname].isel({'lev':-7})
                        var_st = atm_monthly_st[yr][varname].isel({'lev':-7})
                        var_ml = atm_monthly_ml[yr][varname].isel({'lev':-7})
                        var_base_2050 = atm_monthly_base['2050'][varname].isel({'lev':-7})
                elif (varname=='U' or varname=='V'):
                        # Subset variable at 859hPa/525 hPa
                        var_base = atm_monthly_base[yr][varname].isel({'lev':-7})
                        var_st = atm_monthly_st[yr][varname].isel({'lev':-7})
                        var_ml = atm_monthly_ml[yr][varname].isel({'lev':-7})
                        var_base_2050 = atm_monthly_base['2050'][varname].isel({'lev':-7})
                        # var_base = atm_monthly_base[varname].isel({'lev':20})
                        # var_st = atm_monthly_st[varname].isel({'lev':20})
                        # var_ml = atm_monthly_ml[varname].isel({'lev':20})
                else:
                        var_base = atm_monthly_base[yr][varname]
                        var_st = atm_monthly_st[yr][varname]
                        var_ml = atm_monthly_ml[yr][varname]
                        var_base_2050 = atm_monthly_base['2050'][varname]
                st_anom = fun.weighted_temporal_mean(var_st) - fun.weighted_temporal_mean(var_base)
                atm_monthly_st_anom_global_val[yr][varname] = fun.calc_weighted_mean_sd(st_anom)
                ml_anom = fun.weighted_temporal_mean(var_ml) - fun.weighted_temporal_mean(var_base)
                atm_monthly_ml_anom_global_val[yr][varname] = fun.calc_weighted_mean_sd(ml_anom)
                ctrl_anom = fun.weighted_temporal_mean(var_base) - fun.weighted_temporal_mean(var_base_2050)
                atm_monthly_ctrl_anom_global_val[yr][varname] = fun.calc_weighted_mean_sd(ctrl_anom)


### CALCULATE WNA REGIONAL AREA WEIGHTED MEANS AND STANDARD DEVIAIONS
atm_monthly_st_anom_wna_val = {}
atm_monthly_ml_anom_wna_val = {}
atm_monthly_ctrl_anom_wna_val = {}

atm_varnames_monthly_wna_subset = ['TREFHT']
# jja_only = 'y'
jja_only='n'
# Loop through subsetted varnames list
for yr in yr_keys:
        print(yr)
        atm_monthly_ml_anom_wna_val[yr] = {}
        atm_monthly_st_anom_wna_val[yr] = {}
        atm_monthly_ctrl_anom_wna_val[yr] = {}
        for varname in atm_varnames_monthly_wna_subset:
                if (varname=='CCN3' or varname=='CDNUMC' or varname=='RELHUM'):
                        # Subset variable at 859.5 hPa
                        var_base = atm_monthly_base[yr][varname].isel({'lev':-7})
                        var_st = atm_monthly_st[yr][varname].isel({'lev':-7})
                        var_ml = atm_monthly_ml[yr][varname].isel({'lev':-7})
                        var_base_2050 = atm_monthly_base['2050'][varname].isel({'lev':-7})
                elif (varname=='U' or varname=='V'):
                        # Subset variable at 859hPa/525 hPa
                        var_base = atm_monthly_base[yr][varname].isel({'lev':-7})
                        var_st = atm_monthly_st[yr][varname].isel({'lev':-7})
                        var_ml = atm_monthly_ml[yr][varname].isel({'lev':-7})
                        var_base_2050 = atm_monthly_base['2050'][varname].isel({'lev':-7})
                        # var_base = atm_monthly_base[varname].isel({'lev':20})
                        # var_st = atm_monthly_st[varname].isel({'lev':20})
                        # var_ml = atm_monthly_ml[varname].isel({'lev':20})
                else:
                        var_base = atm_monthly_base[yr][varname]
                        var_st = atm_monthly_st[yr][varname]
                        var_ml = atm_monthly_ml[yr][varname]
                        var_base_2050 = atm_monthly_base['2050'][varname]
                # Mask out West Coast States
                var_base = var_base.where((west_coast_state_mask>0),drop=True)
                var_st = var_st.where((west_coast_state_mask>0),drop=True)
                var_ml = var_ml.where((west_coast_state_mask>0),drop=True)
                var_base_2050 = var_base_2050.where((west_coast_state_mask>0),drop=True)
                # Subset for summer months only
                if jja_only=='y':
                        var_base = var_base.loc[{'time':[t for t in var_base.time.values if (t.month==6)|(t.month==7)|(t.month==8)]}]
                        var_st = var_st.loc[{'time':[t for t in var_st.time.values if (t.month==6)|(t.month==7)|(t.month==8)]}]
                        var_ml = var_ml.loc[{'time':[t for t in var_ml.time.values if (t.month==6)|(t.month==7)|(t.month==8)]}]
                        var_base_2050 = var_base_2050.loc[{'time':[t for t in var_base_2050.time.values if (t.month==6)|(t.month==7)|(t.month==8)]}]
                # Calculate anomalies and calculate weighted mean and std
                st_anom = fun.weighted_temporal_mean(var_st) - fun.weighted_temporal_mean(var_base)
                atm_monthly_st_anom_wna_val[yr][varname] = fun.calc_weighted_mean_sd(st_anom)
                ml_anom = fun.weighted_temporal_mean(var_ml) - fun.weighted_temporal_mean(var_base)
                atm_monthly_ml_anom_wna_val[yr][varname] = fun.calc_weighted_mean_sd(ml_anom)
                ctrl_anom = fun.weighted_temporal_mean(var_base) - fun.weighted_temporal_mean(var_base_2050)
                atm_monthly_ctrl_anom_wna_val[yr][varname] = fun.calc_weighted_mean_sd(ctrl_anom)



##################################################################################################################
#%% GENERATE FIGURES
## FIGURE 1: PANEL SUMMARY LOW CLOUD FRACTION + (T w/ winds) (horizontal)
# READ IN LENS2 DATA 
# Note: LENS2 data is not in repository and must be downloaded from CESM2-LENS2 website.
# https://www.cesm.ucar.edu/community-projects/lens2
lens2_cldlow = fun.read_ensemble('CLDLOW','../_data/LENS2/')


# READ IN ISCCP H-series 
# Note: ISCCP data is not in repository and must be downloaed from NCEI webiste.
# https://www.ncei.noaa.gov/data/international-satellite-cloud-climate-project-isccp-h-series-data/access/isccp-basic/hgm/
wd = '../_data/ISCCP'
filename = 'ISCCP-Basic.HGM.v01r00.GLOBAL.1984.01-2009.12.99.9999.GPC.10KM.CS00.EA1.00.nc'
isccp = fun.reorient_netCDF(xr.open_dataset(wd+filename))
isccp_cldlow = (isccp.cldamt_types.sel(cloud_type=slice(0,6))).sum(dim='cloud_type')/100
isccp_cldlow.attrs['units'] = 'fraction'


# Subset LENS2 to match time of ISCCP (1984-2009) (cut off first 4 years)
lens2_cldlow_subset = lens2_cldlow.isel(time=slice(12*4,None))


# Get land mask variables
## ISCCP (subset at 1 time step-- should be the same for all)
isccp_landmask = isccp.eqland.isel(time=-1)
isccp_landmask = xr.where(isccp_landmask>80, np.nan,1)
## CESM2
cesm2_landmask = atm_monthly_base['2010']['LANDFRAC'] #1 for land, 0 for ocean grid boxes
# Replace lat/lon in landfrac file with the LENS2 lat/lon because of rounding errors
cesm2_landmask['lat'] = lens2_cldlow['lat']
cesm2_landmask['lon'] = lens2_cldlow['lon']
# Convert 0's to nan's
cesm2_landmask = xr.where(cesm2_landmask==0, 1, np.nan)


# Mask out land
# mask_land = input('Mask out land (y or n)?: ')
mask_land = 'y'
if mask_land=='y':
    isccp_cldlow = isccp_cldlow*isccp_landmask
    isccp_cldlow.attrs['units'] = 'fraction'
    lens2_cldlow_subset = lens2_cldlow_subset*cesm2_landmask
    lens2_cldlow_subset.attrs['units'] = 'fraction'
elif mask_land=='n':
    print('No land mask')


#%% COMPUTE CLIMATOLOGIES AND ENSEMBLE MEANS
## CLDLOW
# Compute annual ensemble mean
lens2_cldlow_ensemble = fun.weighted_temporal_mean(lens2_cldlow_subset.mean(dim='case')).mean(dim='time')
lens2_cldlow_ensemble.attrs['units'] = lens2_cldlow_subset.units
# Compute monthly climatology
lens2_cldlow_monthly_clim = lens2_cldlow_subset.groupby('time.month').mean().mean(dim='case')
lens2_cldlow_monthly_clim.attrs['units'] = lens2_cldlow_subset.units
# Compute monthly climatological anomalies 
lens2_cldlow_monthly_anom = lens2_cldlow_monthly_clim - lens2_cldlow_ensemble
lens2_cldlow_monthly_anom.attrs['units'] = lens2_cldlow_subset.units


## ISCCP CLDLOW
# Compute annual ensemble mean
isccp_cldlow_ensemble = fun.weighted_temporal_mean(isccp_cldlow).mean(dim='time')
isccp_cldlow_ensemble.attrs['units'] = isccp_cldlow.units
# Compute monthly climatology
isccp_cldlow_monthly_clim = isccp_cldlow.groupby('time.month').mean()
isccp_cldlow_monthly_clim.attrs['units'] = isccp_cldlow.units
# Compute monthly climatological anomalies 
isccp_cldlow_monthly_anom = isccp_cldlow_monthly_clim - isccp_cldlow_ensemble
isccp_cldlow_monthly_anom.attrs['units'] = isccp_cldlow.units


### Calculate seed subset
## Amount
lens2_cldlow_seed_clim = fun.weighted_temporal_mean_clim(lens2_cldlow_monthly_clim.isel(month=slice(2,11)))
isccp_cldlow_seed_clim = fun.weighted_temporal_mean_clim(isccp_cldlow_monthly_clim.isel(month=slice(2,11)))
# Assign units
lens2_cldlow_seed_clim.attrs['units'] = 'cloud fraction'
isccp_cldlow_seed_clim.attrs['units'] = 'cloud fraction'
## Anomalies
lens2_cldlow_seed_anom = fun.weighted_temporal_mean_clim(lens2_cldlow_monthly_anom.isel(month=slice(2,11)))
isccp_cldlow_seed_anom = fun.weighted_temporal_mean_clim(isccp_cldlow_monthly_anom.isel(month=slice(2,11)))
# Assign units
lens2_cldlow_seed_anom.attrs['units'] = 'cloud fraction'
isccp_cldlow_seed_anom.attrs['units'] = 'cloud fraction'


# Get overlay MCB seeding mask files
seeding_mask_st = fun.reorient_netCDF(xr.open_dataset('../_data/mask_CESM2_0.9x1.25_v15.nc'))
seeding_mask_ml = fun.reorient_netCDF(xr.open_dataset('../_data/mask_CESM2_0.9x1.25_v16.nc'))
# Rename time as month
seeding_mask_st = seeding_mask_st.rename({'time':'month'})
seeding_mask_ml = seeding_mask_ml.rename({'time':'month'})


# Subset mean seeded grid cells 
seeding_mask_st_seed = fun.weighted_temporal_mean_clim(seeding_mask_st.mask.isel(month=slice(2,11)))
seeding_mask_ml_seed = fun.weighted_temporal_mean_clim(seeding_mask_ml.mask.isel(month=slice(2,11)))
# Convert to 1s and 0s for plotting
seeding_mask_st_seed = xr.where(seeding_mask_st_seed>0.5, 1, 0)
seeding_mask_ml_seed = xr.where(seeding_mask_ml_seed>0.5, 1, 0)
# Add cyclical point for ML 
seeding_mask_ml_seed_wrap, lon_wrap = add_cyclic_point(seeding_mask_ml_seed,coord=seeding_mask_ml_seed.lon)


## PLOT FIGURE 1
yr = '2010'
fig = plt.figure(figsize=(18,10));
# a ISCCP LOW CLOUD
subplot_num = 0
ax1,p1=fun.plot_panel_maps(in_xr=isccp_cldlow_seed_clim, cmin=0, cmax=.8, ccmap='Blues', plot_zoom='pacific_ocean', central_lon=180,\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,mean_val='none',cbar=False);
plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(west_coast_state_mask.lon,west_coast_state_mask.lat,west_coast_state_mask,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title('a',fontsize=10, fontweight='bold',loc='left');
subplot_num += 1
# c ML T 
atm_annual_ml_anom[yr]['TREFHT'].attrs['units'] = '${\N{DEGREE SIGN}}$C'
fun.plot_panel_maps(in_xr=atm_annual_ml_anom[yr]['TREFHT'], cmin=-3, cmax=3, ccmap='coolwarm', plot_zoom='pacific_ocean', central_lon=180,\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,\
                        CI_in=atm_annual_ml_clim_pval[yr]['TREFHT'],CI_level=0.05,CI_display='mask',\
                        mean_val='none',cbar=False)
plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(west_coast_state_mask.lon,west_coast_state_mask.lat,west_coast_state_mask,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
m1 = plt.quiver(atm_annual_base_clim[yr]['U'].lon.values[::10], atm_annual_base_clim[yr]['U'].lat.values[::10],\
        atm_annual_base_clim[yr]['U'].values[::10,::10],atm_annual_base_clim[yr]['V'].values[::10,::10],\
        transform=ccrs.PlateCarree(), units='width', pivot='middle', color='k',width=0.004,scale=35,scale_units='inches');
plt.title('c',fontsize=10, fontweight='bold',loc='left');
subplot_num += 1
# b LENS2 LOW CLOUD
fun.plot_panel_maps(in_xr=lens2_cldlow_seed_clim, cmin=0, cmax=.8, ccmap='Blues', plot_zoom='pacific_ocean', central_lon=180,\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,mean_val='none',cbar=False);
plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(west_coast_state_mask.lon,west_coast_state_mask.lat,west_coast_state_mask,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title('b',fontsize=10, fontweight='bold',loc='left');
subplot_num +=1
# d ST T
atm_annual_st_anom[yr]['TREFHT'].attrs['units'] = '${\N{DEGREE SIGN}}$C'
ax2,p2=fun.plot_panel_maps(in_xr=atm_annual_st_anom[yr]['TREFHT'], cmin=-3, cmax=3, ccmap='coolwarm', plot_zoom='pacific_ocean', central_lon=180,\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,\
                        CI_in=atm_annual_st_clim_pval[yr]['TREFHT'],CI_level=0.05,CI_display='mask',\
                        mean_val='none',cbar=False)
plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(west_coast_state_mask.lon,west_coast_state_mask.lat,west_coast_state_mask,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
m1 = plt.quiver(atm_annual_base_clim[yr]['U'].lon.values[::10], atm_annual_base_clim[yr]['U'].lat.values[::10],\
        atm_annual_base_clim[yr]['U'].values[::10,::10],atm_annual_base_clim[yr]['V'].values[::10,::10],\
        transform=ccrs.PlateCarree(), units='width', pivot='middle', color='k',width=0.004,scale=35,scale_units='inches');
# plt.quiverkey(m1, X=4, Y=.85, U= 10, label='10 ms$^{-1}$', labelpos='E', coordinates = 'inches');
plt.title('d',fontsize=10, fontweight='bold',loc='left');
fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.15,hspace=0.05);
# plt.tight_layout();
## Add PRECT quiver key
plt.quiverkey(m1, X=5.3, Y=0.15, U= 10, label='10 ms$^{-1}$',labelpos='E', coordinates = 'inches', fontproperties={'size':7});
# plt.annotate('2015-2016 El Niño + MCB', xy=(.095,.39), xycoords='figure fraction',color='k');
## Add colorbars to bottom of figure
cbar_ax = fig.add_axes([0.12, 0.07, 0.36, 0.025]) #rect kwargs [left, bottom, width, height];
cbar = plt.colorbar(p1, cax = cbar_ax, orientation='horizontal', extend='both',pad=0.1);
cbar.ax.tick_params(labelsize=7); cbar.set_label(label='W/m$^{2}$', size=7)
cbar_ax = fig.add_axes([0.54, 0.07, 0.36, 0.025]) #rect kwargs [left, bottom, width, height];
cbar=plt.colorbar(p2, cax = cbar_ax, orientation='horizontal', extend='both',pad=0.1);
cbar.ax.tick_params(labelsize=7); cbar.set_label(label='\N{DEGREE SIGN}C', size=7)


# Calculate temperature change in target region per forcing
# Define dictionaries to populate
tnorm_ml_anom = {}
tnorm_st_anom = {}
tnorm_monthly_ml_anom = {}
tnorm_monthly_st_anom = {}
for yr in yr_keys:
        print(yr)
        # Annual
        tnorm_ml_anom[yr] =  atm_annual_ml_anom[yr]['TREFHT']/np.abs(atm_monthly_ml_anom_global_val[yr]['FSNTOA'][0])
        tnorm_st_anom[yr] =  atm_annual_st_anom[yr]['TREFHT']/np.abs(atm_monthly_st_anom_global_val[yr]['FSNTOA'][0])
        # Monthly
        tnorm_monthly_ml_anom[yr] =  (atm_monthly_ml[yr]['TREFHT']-atm_monthly_base[yr]['TREFHT'])/np.abs(atm_monthly_ml_anom_global_val[yr]['FSNTOA'][0])
        tnorm_monthly_st_anom[yr] =  (atm_monthly_st[yr]['TREFHT']-atm_monthly_base[yr]['TREFHT'])/np.abs(atm_monthly_st_anom_global_val[yr]['FSNTOA'][0])      
        # Compute WNA target region mean
        ml_anom = fun.calc_weighted_mean_sd(fun.weighted_temporal_mean(tnorm_monthly_ml_anom[yr].where((west_coast_state_mask>0),drop=True)))
        print('ML: ',ml_anom, 'deg C/W/m2')
        st_anom = fun.calc_weighted_mean_sd(fun.weighted_temporal_mean(tnorm_monthly_st_anom[yr].where((west_coast_state_mask>0),drop=True)))
        print('ST: ',st_anom, 'deg C/W/m2')       


## PLOT ED2
yr = '2010'
fig = plt.figure(figsize=(18,10));
# a ISCCP LOW CLOUD
subplot_num = 0
ax1,p1=fun.plot_panel_maps(in_xr=isccp_cldlow_seed_clim, cmin=0, cmax=.8, ccmap='Blues', plot_zoom='pacific_ocean', central_lon=180,\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,mean_val='none',cbar=False);
plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(west_coast_state_mask.lon,west_coast_state_mask.lat,west_coast_state_mask,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title('a',fontsize=10, fontweight='bold',loc='left');
subplot_num += 1
# c ML T 
# Calculate normalized temperature response by global mean absolute forcing
tnorm_ml_anom =  atm_annual_ml_anom[yr]['TREFHT']/np.abs(atm_monthly_ml_anom_global_val[yr]['FSNTOA'][0])
tnorm_ml_anom.attrs['units'] = '${\N{DEGREE SIGN}}$C/Wm$^{-2}$'
tnorm_ml_mean = [float(fun.calc_weighted_mean_tseries(tnorm_ml_anom).values),np.nan]
fun.plot_panel_maps(in_xr=tnorm_ml_anom, cmin=-3, cmax=3, ccmap='coolwarm', plot_zoom='pacific_ocean', central_lon=180,\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,\
                        mean_val=tnorm_ml_mean,cbar=False)
plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(west_coast_state_mask.lon,west_coast_state_mask.lat,west_coast_state_mask,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title('c',fontsize=10, fontweight='bold',loc='left');
subplot_num += 1
# b LENS2 LOW CLOUD
fun.plot_panel_maps(in_xr=lens2_cldlow_seed_clim, cmin=0, cmax=.8, ccmap='Blues', plot_zoom='pacific_ocean', central_lon=180,\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,mean_val='none',cbar=False);
plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(west_coast_state_mask.lon,west_coast_state_mask.lat,west_coast_state_mask,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title('b',fontsize=10, fontweight='bold',loc='left');
subplot_num +=1
# d ST T
tnorm_st_anom =  atm_annual_st_anom[yr]['TREFHT']/np.abs(atm_monthly_st_anom_global_val[yr]['FSNTOA'][0])
tnorm_st_anom.attrs['units'] = '${\N{DEGREE SIGN}}$C/Wm$^{-2}$'
tnorm_st_mean = [float(fun.calc_weighted_mean_tseries(tnorm_st_anom).values),np.nan]
ax2,p2=fun.plot_panel_maps(in_xr=tnorm_st_anom, cmin=-3, cmax=3, ccmap='coolwarm', plot_zoom='pacific_ocean', central_lon=180,\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,\
                        mean_val=tnorm_st_mean,cbar=False)
plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(west_coast_state_mask.lon,west_coast_state_mask.lat,west_coast_state_mask,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
m1 = plt.quiver(atm_annual_base_clim[yr]['U'].lon.values[::10], atm_annual_base_clim[yr]['U'].lat.values[::10],\
        atm_annual_base_clim[yr]['U'].values[::10,::10],atm_annual_base_clim[yr]['V'].values[::10,::10],\
        transform=ccrs.PlateCarree(), units='width', pivot='middle', color='k',width=0.004,scale=35,scale_units='inches');
# plt.quiverkey(m1, X=4, Y=.85, U= 10, label='10 ms$^{-1}$', labelpos='E', coordinates = 'inches');
plt.title('d',fontsize=10, fontweight='bold',loc='left');
fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.15,hspace=0.05);
# plt.tight_layout();
# plt.annotate('2015-2016 El Niño + MCB', xy=(.095,.39), xycoords='figure fraction',color='k');
## Add colorbars to bottom of figure
cbar_ax = fig.add_axes([0.12, 0.07, 0.36, 0.025]) #rect kwargs [left, bottom, width, height];
cbar = plt.colorbar(p1, cax = cbar_ax, orientation='horizontal', extend='both',pad=0.1);
cbar.ax.tick_params(labelsize=7); cbar.set_label(label='W/m$^{2}$', size=7)
cbar_ax = fig.add_axes([0.54, 0.07, 0.36, 0.025]) #rect kwargs [left, bottom, width, height];
cbar=plt.colorbar(p2, cax = cbar_ax, orientation='horizontal', extend='both',pad=0.1);
cbar.ax.tick_params(labelsize=7); cbar.set_label(label='${\N{DEGREE SIGN}}$C/Wm$^{-2}$', size=7)



#%%  FIG S5
fig = plt.figure(figsize=(18,11));
subplot_num = 0
# 2010 control cloud cover
in_xr = fun.weighted_temporal_mean_clim(atm_monthly_base_clim['2010']['CLDLOW'].isel(month=slice(2,11)))
in_xr.attrs['units'] = 'fraction'
fun.plot_panel_maps(in_xr=in_xr, cmin=0, cmax=.8, ccmap='Blues', plot_zoom='pacific_ocean', central_lon=180,\
                        projection='PlateCarree',nrow=2,ncol=2,subplot_num=subplot_num,mean_val='none');
plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title('a',fontsize=10, fontweight='bold',loc='left');
subplot_num += 1
# 2050 control cloud cover
in_xr = fun.weighted_temporal_mean_clim(atm_monthly_base_clim['2050']['CLDLOW'].isel(month=slice(2,11)))
in_xr.attrs['units'] = 'fraction'
fun.plot_panel_maps(in_xr=in_xr, cmin=0, cmax=.8, ccmap='Blues', plot_zoom='pacific_ocean', central_lon=180,\
                        projection='PlateCarree',nrow=2,ncol=2,subplot_num=subplot_num,mean_val='none');
plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title('b',fontsize=10, fontweight='bold',loc='left');
subplot_num += 1
# Difference between 2050 and 2010 control cloud cover
in_xr = fun.weighted_temporal_mean_clim(atm_monthly_base_clim['2050']['CLDLOW'].isel(month=slice(2,11))) - fun.weighted_temporal_mean_clim(atm_monthly_base_clim['2010']['CLDLOW'].isel(month=slice(2,11)))
in_xr.attrs['units'] = 'fraction'
fun.plot_panel_maps(in_xr=in_xr, cmin=-.2, cmax=.2, ccmap='PuOr', plot_zoom='pacific_ocean', central_lon=180,\
                        projection='PlateCarree',nrow=2,ncol=2,subplot_num=subplot_num,mean_val='none');
plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title('c',fontsize=10, fontweight='bold',loc='left');
fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.15,hspace=0.05);


#%% TABLE S1: REGIONAL/GLOBAL MEAN T RESPONSE AND RF FROM MAR-NOV
# Global mean RF
for yr in yr_keys:
        print(yr)
        print('Global RF (ML): ', [round(x, 2) for x in atm_monthly_ml_anom_global_val[yr]['FSNTOA']])
        print('Global RF (ST): ', [round(x, 2) for x in atm_monthly_st_anom_global_val[yr]['FSNTOA']])

# Seeding region RF
# Define seeding mask variables
seeding_mask_st = fun.reorient_netCDF(xr.open_dataset('../_data/mask_CESM2_0.9x1.25_v15.nc')).mask
seeding_mask_ml = fun.reorient_netCDF(xr.open_dataset('../_data/mask_CESM2_0.9x1.25_v16.nc')).mask
# Filter out negligble grid boxes and create climatology
seeding_mask_st = xr.where(seeding_mask_st>0.5,1,0)
seeding_mask_ml = xr.where(seeding_mask_ml>0.5,1,0)

seed_fsntoa_mean_std_ml = {}
seed_fsntoa_mean_std_st = {}
for yr in yr_keys:
        print(yr)
        seed_fsntoa_mean_std_ml[yr] = fun.calc_weighted_mean_sd(fun.weighted_temporal_mean((atm_monthly_ml[yr]['FSNTOA'] - atm_monthly_base[yr]['FSNTOA']).where(seeding_mask_ml>0)))
        seed_fsntoa_mean_std_st[yr] = fun.calc_weighted_mean_sd(fun.weighted_temporal_mean((atm_monthly_st[yr]['FSNTOA'] - atm_monthly_base[yr]['FSNTOA']).where(seeding_mask_st>0)))
        print('MCB region RF (ML): ', [round(x, 2) for x in seed_fsntoa_mean_std_ml[yr]])
        print('MCB region  RF (ST): ', [round(x, 2) for x in seed_fsntoa_mean_std_st[yr]])

# Global mean T response
for yr in yr_keys:
        print(yr)
        print('Global T (ML): ', [round(x, 2) for x in atm_monthly_ml_anom_global_val[yr]['TREFHT']])
        print('Global T (ST): ', [round(x, 2) for x in atm_monthly_st_anom_global_val[yr]['TREFHT']])

# WNA T response
for yr in yr_keys:
        print(yr)
        print('WNA T (ML): ', [round(x, 2) for x in atm_monthly_ml_anom_wna_val[yr]['TREFHT']])
        print('WNA T (ST): ', [round(x, 2) for x in atm_monthly_st_anom_wna_val[yr]['TREFHT']])

# Global mean T response/ global mean RF
global_tnorm_mean_std_ml={}
global_tnorm_mean_std_st={}
for yr in yr_keys:
       print(yr)
       global_tnorm_mean_std_ml[yr] = atm_monthly_ml_anom_global_val[yr]['TREFHT'][0]/np.abs(atm_monthly_ml_anom_global_val[yr]['FSNTOA'][0])
       global_tnorm_mean_std_st[yr] = atm_monthly_st_anom_global_val[yr]['TREFHT'][0]/np.abs(atm_monthly_st_anom_global_val[yr]['FSNTOA'][0])
       print('Global T/Global RF (ML): ', round(global_tnorm_mean_std_ml[yr], 2))
       print('Global T/Global RF (ST): ', round(global_tnorm_mean_std_st[yr], 2))

# WNA mean T response/ global mean RF
wna_tnorm_mean_std_ml={}
wna_tnorm_mean_std_st={}
for yr in yr_keys:
       print(yr)
       wna_tnorm_mean_std_ml[yr] =  atm_monthly_ml_anom_wna_val[yr]['TREFHT'][0]/np.abs(atm_monthly_ml_anom_global_val[yr]['FSNTOA'][0])
       wna_tnorm_mean_std_st[yr] = atm_monthly_st_anom_wna_val[yr]['TREFHT'][0]/np.abs(atm_monthly_st_anom_global_val[yr]['FSNTOA'][0])
       print('WNA T/Global RF (ML): ', round(wna_tnorm_mean_std_ml[yr], 2))
       print('WNA T/Global RF (ST): ', round(wna_tnorm_mean_std_st[yr], 2))


#%% ED FIGURES 1 & 6
for yr in yr_keys:
        fig = plt.figure(figsize=(18,6));
        subplot_num = 0
        ## ML
        # CDNUMC
        atm_annual_ml_anom[yr]['CDNUMC'].attrs['units']='#/cm$^{3}$'
        fun.plot_panel_maps(in_xr=atm_annual_ml_anom[yr]['CDNUMC'], cmin=-50, cmax=50, ccmap='BrBG', plot_zoom='global', central_lon=180,\
                                CI_in=atm_annual_ml_clim_pval[yr]['CDNUMC'],CI_level=0.05,CI_display='mask',nrow=2,ncol=4,subplot_num=subplot_num,\
                                projection='Robinson',mean_val=atm_monthly_ml_anom_global_val[yr]['CDNUMC'])
        plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title('a',fontsize=10, fontweight='bold',loc='left');
        subplot_num += 1
        # CLDLOW
        atm_annual_ml_anom[yr]['CLDLOW'].attrs['units']='cloud fraction'
        fun.plot_panel_maps(in_xr=atm_annual_ml_anom[yr]['CLDLOW'], cmin=-.2, cmax=.2, ccmap='PuOr', plot_zoom='global', central_lon=180,\
                                CI_in=atm_annual_ml_clim_pval[yr]['CLDLOW'],CI_level=0.05,CI_display='mask',nrow=2,ncol=4,subplot_num=subplot_num,\
                                projection='Robinson',mean_val=atm_monthly_ml_anom_global_val[yr]['CLDLOW'])
        plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title('b',fontsize=10, fontweight='bold',loc='left');
        subplot_num += 1
        # FSNTOA
        atm_annual_ml_anom[yr]['FSNTOA'].attrs['units']='W/m$^{2}$'
        fun.plot_panel_maps(in_xr=atm_annual_ml_anom[yr]['FSNTOA'], cmin=-20, cmax=20, ccmap='seismic', plot_zoom='global', central_lon=180,\
                                CI_in=atm_annual_ml_clim_pval[yr]['FSNTOA'],CI_level=0.05,CI_display='mask',nrow=2,ncol=4,subplot_num=subplot_num,\
                                projection='Robinson',mean_val=atm_monthly_ml_anom_global_val[yr]['FSNTOA'])
        plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title('c',fontsize=10, fontweight='bold',loc='left');
        subplot_num += 1
        # TREFHT
        atm_annual_ml_anom[yr]['TREFHT'].attrs['units']='${\N{DEGREE SIGN}}$C'
        fun.plot_panel_maps(in_xr=atm_annual_ml_anom[yr]['TREFHT'], cmin=-3, cmax=3, ccmap='coolwarm', plot_zoom='global', central_lon=180,\
                                CI_in=atm_annual_ml_clim_pval[yr]['TREFHT'],CI_level=0.05,CI_display='mask',nrow=2,ncol=4,subplot_num=subplot_num,\
                                projection='Robinson',mean_val=atm_monthly_ml_anom_global_val[yr]['TREFHT'])
        plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title('d',fontsize=10, fontweight='bold',loc='left');
        subplot_num += 1
        ## ST
        # CDNUMC
        atm_annual_st_anom[yr]['CDNUMC'].attrs['units']='#/cm$^{3}$'
        fun.plot_panel_maps(in_xr=atm_annual_st_anom[yr]['CDNUMC'], cmin=-50, cmax=50, ccmap='BrBG', plot_zoom='global', central_lon=180,\
                                CI_in=atm_annual_st_clim_pval[yr]['CDNUMC'],CI_level=0.05,CI_display='mask',nrow=2,ncol=4,subplot_num=subplot_num,\
                                projection='Robinson',mean_val=atm_monthly_st_anom_global_val[yr]['CDNUMC'])
        plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title('e',fontsize=10, fontweight='bold',loc='left');
        subplot_num += 1
        # CLDLOW
        atm_annual_st_anom[yr]['CLDLOW'].attrs['units']='cloud fraction'
        fun.plot_panel_maps(in_xr=atm_annual_st_anom[yr]['CLDLOW'], cmin=-.2, cmax=.2, ccmap='PuOr', plot_zoom='global', central_lon=180,\
                                CI_in=atm_annual_st_clim_pval[yr]['CLDLOW'],CI_level=0.05,CI_display='mask',nrow=2,ncol=4,subplot_num=subplot_num,\
                                projection='Robinson',mean_val=atm_monthly_st_anom_global_val[yr]['CLDLOW'])
        plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title('f',fontsize=10, fontweight='bold',loc='left');
        subplot_num += 1
        # FSNTOA
        atm_annual_st_anom[yr]['FSNTOA'].attrs['units']='W/m$^{2}$'
        fun.plot_panel_maps(in_xr=atm_annual_st_anom[yr]['FSNTOA'], cmin=-20, cmax=20, ccmap='seismic', plot_zoom='global', central_lon=180,\
                                CI_in=atm_annual_st_clim_pval[yr]['FSNTOA'],CI_level=0.05,CI_display='mask',nrow=2,ncol=4,subplot_num=subplot_num,\
                                projection='Robinson',mean_val=atm_monthly_st_anom_global_val[yr]['FSNTOA'])
        plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title('g',fontsize=10, fontweight='bold',loc='left');
        subplot_num += 1
        # TREFHT
        atm_annual_st_anom[yr]['TREFHT'].attrs['units']='${\N{DEGREE SIGN}}$C'
        fun.plot_panel_maps(in_xr=atm_annual_st_anom[yr]['TREFHT'], cmin=-3, cmax=3, ccmap='coolwarm', plot_zoom='global', central_lon=180,\
                                CI_in=atm_annual_st_clim_pval[yr]['TREFHT'],CI_level=0.05,CI_display='mask',nrow=2,ncol=4,subplot_num=subplot_num,\
                                projection='Robinson',mean_val=atm_monthly_st_anom_global_val[yr]['TREFHT'])
        plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title('h',fontsize=10, fontweight='bold',loc='left');
        fig.subplots_adjust(bottom=0.1, top=0.98, wspace=0.1,hspace=0.15);


## ED FIGURE 10
yr='2050'
fig = plt.figure(figsize=(18,13));
subplot_num = 0
# ML
# TREFHT
atm_annual_ml_anom[yr]['TREFHT'].attrs['units'] = '${\N{DEGREE SIGN}}$C'
fun.plot_panel_maps(in_xr=atm_annual_ml_anom[yr]['TREFHT'], cmin=-3, cmax=3, ccmap='coolwarm', plot_zoom='global', central_lon=180,\
                        CI_in=atm_annual_ml_clim_pval[yr]['TREFHT'],CI_level=0.05,CI_display='mask',\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,\
                        mean_val=atm_monthly_ml_anom_global_val[yr]['TREFHT'])
plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
m1 = plt.quiver(atm_annual_ml_clim[yr]['U'].lon.values[::10], atm_annual_ml_clim[yr]['U'].lat.values[::10],\
        atm_annual_ml_clim[yr]['U'].values[::10,::10],atm_annual_ml_clim[yr]['V'].values[::10,::10],\
        transform=ccrs.PlateCarree(), units='width', pivot='middle', color='k',width=0.003);
plt.title('a',fontsize=10, fontweight='bold',loc='left');
subplot_num += 1
# SHFLX
atm_annual_ml_anom[yr]['SHFLX'].attrs['units'] = 'W/m$^{2}$'
fun.plot_panel_maps(in_xr=atm_annual_ml_anom[yr]['SHFLX'], cmin=-10, cmax=10, ccmap='seismic', plot_zoom='global', central_lon=180,\
                        CI_in=atm_annual_ml_clim_pval[yr]['SHFLX'],CI_level=0.05,CI_display='mask',\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,\
                        mean_val=atm_monthly_ml_anom_global_val[yr]['SHFLX'])
plt.contour(lon_wrap,seeding_mask_ml_seed.lat,seeding_mask_ml_seed_wrap,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title('b',fontsize=10, fontweight='bold',loc='left');
subplot_num += 1
# ST
atm_annual_st_anom[yr]['TREFHT'].attrs['units'] = '${\N{DEGREE SIGN}}$C'
fun.plot_panel_maps(in_xr=atm_annual_st_anom[yr]['TREFHT'], cmin=-3, cmax=3, ccmap='coolwarm', plot_zoom='global', central_lon=180,\
                        CI_in=atm_annual_st_clim_pval[yr]['TREFHT'],CI_level=0.05,CI_display='mask',\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,\
                        mean_val=atm_monthly_st_anom_global_val[yr]['TREFHT'])
plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
m1 = plt.quiver(atm_annual_st_clim[yr]['U'].lon.values[::10], atm_annual_st_clim[yr]['U'].lat.values[::10],\
        atm_annual_st_clim[yr]['U'].values[::10,::10],atm_annual_st_clim[yr]['V'].values[::10,::10],\
        transform=ccrs.PlateCarree(), units='width', pivot='middle', color='k',width=0.003);
plt.quiverkey(m1, X=4, Y=.9, U= 10, label='10 ms$^{-1}$', labelpos='E', coordinates = 'inches');
plt.title('c',fontsize=10, fontweight='bold',loc='left');
subplot_num += 1
# SHFLX
atm_annual_st_anom[yr]['SHFLX'].attrs['units'] = 'W/m$^{2}$'
fun.plot_panel_maps(in_xr=atm_annual_st_anom[yr]['SHFLX'], cmin=-10, cmax=10, ccmap='seismic', plot_zoom='global', central_lon=180,\
                        CI_in=atm_annual_st_clim_pval[yr]['SHFLX'],CI_level=0.05,CI_display='mask',\
                        projection='Robinson',nrow=2,ncol=2,subplot_num=subplot_num,\
                        mean_val=atm_monthly_st_anom_global_val[yr]['SHFLX'])
plt.contour(seeding_mask_st_seed.lon,seeding_mask_st_seed.lat,seeding_mask_st_seed,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='springgreen', linewidths=1,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title('d',fontsize=10, fontweight='bold',loc='left');
fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.1,hspace=0.1);


#%% ML TELECONNECTION MECHANISM ANALYSIS
### iii) Is mean advection responsible transporting cool anomalies to the subtropics in the ML case? (transient response in first 2 years)
sim_year = input('2010 or 2050?: ')
if sim_year=='2010':
        ## 1) CONTROL CASE
        atm_monthly_base = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.001.cam.h0.processed.0001-0004.nc')
        ## 2) MCB SUBTROPICAL SEEDING CASE
        atm_monthly_st = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.MCB-mask15.001.cam.h0.processed.0001-0004.nc')
        ## 3) MCB MIDLATITUDE SEEDING CASE
        atm_monthly_ml = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.MCB-mask16.001.cam.h0.processed.0001-0004.nc')
elif sim_year=='2050':
        ## 1) CONTROL CASE
        atm_monthly_base = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.001.cam.h0.processed.0055-0058.nc')
        ## 2) MCB SUBTROPICAL SEEDING CASE
        atm_monthly_st = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.MCB-mask15.002.cam.h0.processed.0055-0058.nc')
        ## 3) MCB MIDLATITUDE SEEDING CASE
        atm_monthly_ml = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.MCB-mask16.002.cam.h0.processed.0055-0058.nc')


# Convert PSL (Pa to hPa)
atm_monthly_base['PSL'] = atm_monthly_base['PSL']*0.01
atm_monthly_base['PSL'].attrs['units'] = 'hPa'
atm_monthly_st['PSL'] = atm_monthly_st['PSL']*0.01
atm_monthly_st['PSL'].attrs['units'] = 'hPa'
atm_monthly_ml['PSL'] = atm_monthly_ml['PSL']*0.01
atm_monthly_ml['PSL'].attrs['units'] = 'hPa'


# COMPUTE CLIMATOLOGIES AND ANOMALIES FOR SELECT VARIABLES
# Create list of ATM variable names
atm_varnames_monthly_subset = ['TREFHT','U','V','PSL']

## 1b) ANNUAL ATMOSPHERE (FROM MONTHLY DATA)
# Create empty dictionaries for climatologies and anomalies
atm_annual_base_clim = {}
atm_annual_st_clim = {}
atm_annual_ml_clim = {}
atm_annual_st_anom = {}
atm_annual_ml_anom = {}
## Loop through subsetted varnames list. 
print('##ATM ANNUAL##')
for varname in atm_varnames_monthly_subset:
        print(varname)
        if(varname=='U' or varname=='V'):
                # Subset variable at 993hPa
                var_base_859hpa = atm_monthly_base[varname].isel({'lev':-1})
                var_st_859hpa = atm_monthly_st[varname].isel({'lev':-1})
                var_ml_859hpa = atm_monthly_ml[varname].isel({'lev':-1})
                # Compute annual climatology
                atm_annual_base_clim[varname] = fun.weighted_temporal_mean(var_base_859hpa)
                atm_annual_base_clim[varname].attrs['units'] = atm_monthly_base[varname].units
                atm_annual_st_clim[varname] = fun.weighted_temporal_mean(var_st_859hpa)
                atm_annual_st_clim[varname].attrs['units'] = atm_monthly_st[varname].units
                atm_annual_ml_clim[varname] = fun.weighted_temporal_mean(var_ml_859hpa)
                atm_annual_ml_clim[varname].attrs['units'] = atm_monthly_ml[varname].units                
        else:
                # Compute annual climatology
                atm_annual_base_clim[varname] = fun.weighted_temporal_mean(atm_monthly_base[varname])
                atm_annual_base_clim[varname].attrs['units'] = atm_monthly_base[varname].units
                atm_annual_st_clim[varname] = fun.weighted_temporal_mean(atm_monthly_st[varname])
                atm_annual_st_clim[varname].attrs['units'] = atm_monthly_st[varname].units
                atm_annual_ml_clim[varname] =  fun.weighted_temporal_mean(atm_monthly_ml[varname])
                atm_annual_ml_clim[varname].attrs['units'] = atm_monthly_ml[varname].units
        # Compute annual anomalies (MCB - CONTROL)
        atm_annual_st_anom[varname] = atm_annual_st_clim[varname] - atm_annual_base_clim[varname]
        atm_annual_st_anom[varname].attrs['units'] = atm_monthly_base[varname].units
        atm_annual_ml_anom[varname] = atm_annual_ml_clim[varname] - atm_annual_base_clim[varname]
        atm_annual_ml_anom[varname].attrs['units'] = atm_monthly_base[varname].units


## Plot TREFHT maps with wind vectors for first 4 years
# ML
label_vec = ['a','b','c','d','e','f','g','h']
yr_vec = np.arange(0,4)
fig = plt.figure(figsize=(18,22));
subplot_num = 0
for yr in range(len(yr_vec)):       
        ## TREFHT + WINDS
        in_xr = atm_annual_ml_anom['TREFHT'].isel(time=yr)
        in_xr.attrs['units'] = '${\N{DEGREE SIGN}}$C'
        fun.plot_panel_maps(in_xr=in_xr, cmin=-3, cmax=3, ccmap='RdYlBu_r', plot_zoom='pacific_ocean', central_lon=180,\
                                nrow=4,ncol=2,subplot_num=subplot_num,\
                                mean_val='none')
        # Add winds
        m1 = plt.quiver(atm_annual_ml_anom['U'].isel(time=yr).lon.values[::10], atm_annual_ml_anom['U'].isel(time=yr).lat.values[::10],\
                atm_annual_ml_anom['U'].isel(time=yr).values[::10,::10],atm_annual_ml_anom['V'].isel(time=yr).values[::10,::10],\
                transform=ccrs.PlateCarree(), units='width', pivot='middle', color='k',width=0.005);
        plt.title(label_vec[subplot_num],fontsize=10,loc='left',fontweight='bold');
        plt.quiverkey(m1, X=6, Y=14.8, U= 2, label='2 ms$^{-1}$', labelpos='E', coordinates = 'inches', fontproperties={'size':7});
        subplot_num += 1
        # PSL
        fun.plot_panel_maps(in_xr=atm_annual_ml_anom['PSL'].isel(time=yr), cmin=-3, cmax=3, ccmap='bwr', plot_zoom='pacific_ocean', central_lon=180,\
                                nrow=4,ncol=2,subplot_num=subplot_num,\
                                mean_val='none')
        plt.title(label_vec[subplot_num],fontsize=10,loc='left',fontweight='bold');
        subplot_num += 1
## Save figure and close
fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.15,hspace=0.1);

# ST
label_vec = ['a','b','c','d','e','f','g','h']
yr_vec = np.arange(0,4)
fig = plt.figure(figsize=(18,22));
subplot_num = 0
for yr in range(len(yr_vec)):
        ## TREFHT + WINDS
        in_xr = atm_annual_st_anom['TREFHT'].isel(time=yr)
        in_xr.attrs['units'] = '${\N{DEGREE SIGN}}$C'
        fun.plot_panel_maps(in_xr=in_xr, cmin=-3, cmax=3, ccmap='RdYlBu_r', plot_zoom='pacific_ocean', central_lon=180,\
                                nrow=4,ncol=2,subplot_num=subplot_num,\
                                mean_val='none')
        # Add winds
        m1 = plt.quiver(atm_annual_st_anom['U'].isel(time=yr).lon.values[::10], atm_annual_st_anom['U'].isel(time=yr).lat.values[::10],\
                atm_annual_st_anom['U'].isel(time=yr).values[::10,::10],atm_annual_st_anom['V'].isel(time=yr).values[::10,::10],\
                transform=ccrs.PlateCarree(), units='width', pivot='middle', color='k',width=0.005);
        plt.title(label_vec[subplot_num],fontsize=10,loc='left',fontweight='bold');
        plt.quiverkey(m1, X=6, Y=14.8, U= 2, label='2 ms$^{-1}$', labelpos='E', coordinates = 'inches', fontproperties={'size':7});
        subplot_num += 1
        # PSL
        fun.plot_panel_maps(in_xr=atm_annual_st_anom['PSL'].isel(time=yr), cmin=-3, cmax=3, ccmap='bwr', plot_zoom='pacific_ocean', central_lon=180,\
                                nrow=4,ncol=2,subplot_num=subplot_num,\
                                mean_val='none')
        plt.title(label_vec[subplot_num],fontsize=10,loc='left',fontweight='bold');
        subplot_num += 1
## Save figure and close
fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.15,hspace=0.1);

