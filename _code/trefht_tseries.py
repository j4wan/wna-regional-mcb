### PURPOSE: Analyze and plot temperature time series for full simulations
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
import lens2_preanalysis_functions as fun
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


# Create dictionaries to hold both 2010 and 2050 output
atm_monthly_base = {}
atm_monthly_st = {}
atm_monthly_ml = {}


# Read in monthly TREFHT for the full simulation for each time period
yr_keys = ['2050', '2010']
wd_data = '../_data/'
for yr in yr_keys:
    if yr=='2050':
        ## 1) CONTROL CASE
        casename_base = 'b.e22.B2000.f09_g17.2050cycle.001'
        atm_monthly_base[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.001-002.cam.h0.TREFHT.0001-0085.nc')
        ## 2) MCB SUBTROPICAL SEEDING CASE
        casename_st = 'b.e22.B2000.f09_g17.2050cycle.MCB-mask15.001'
        atm_monthly_st[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.MCB-mask15.001-002.cam.h0.TREFHT.0001-0085.nc')
        ## 3) MCB MID-LATITUDE SEEDING CASE
        casename_ml = 'b.e22.B2000.f09_g17.2050cycle.MCB-mask16.001'
        atm_monthly_ml[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.MCB-mask16.001-002.cam.h0.TREFHT.0001-0085.nc')
    elif yr=='2010':
        ## 1) CONTROL CASE
        casename_base = 'b.e22.B2000.f09_g17.2010cycle.001-002'
        atm_monthly_base[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.001.cam.h0.TREFHT.0001-0085.nc')
        ## 2) MCB SUBTROPICAL SEEDING CASE
        casename_st = 'b.e22.B2000.f09_g17.2010cycle.MCB-mask15.001-002'
        atm_monthly_st[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.MCB-mask15.001.cam.h0.TREFHT.0001-0085.nc')
        ## 3) MCB MID-LATITUDE SEEDING CASE
        casename_ml = 'b.e22.B2000.f09_g17.2010cycle.MCB-mask16.001-002'
        atm_monthly_ml[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.MCB-mask16.001.cam.h0.TREFHT.0001-0085.nc')



#%% COMPUTE CLIMATOLOGIES AND ANOMALIES FOR SELECT VARIABLES
# Create list of ATM variable names
atm_varnames_monthly_subset = ['TREFHT']

## 1a) MONTHLY ATMOSPHERE
# Create empty dictionaries for climatologies and anomalies
atm_monthly_base_clim = {}
atm_monthly_st_clim = {}
atm_monthly_ml_clim = {}
atm_monthly_st_anom = {}
atm_monthly_ml_anom = {}
atm_monthly_ctrl_anom = {}
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



#%% GLOBAL MEAN TEMPERATURE TIME SERIES AND CALCULATIONS
### PLOT GLOBAL MEAN TREFHT TIME SERIES
# Function to calculate 12-month rolling average
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def calc_weighted_mean(DataArray):
    '''
    Calculate area-weighted aggregate mean of a variable in an input DataArray
    Adapted from https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html
    Returns a two values: the area-weighted mean over time of the variable and standard deviation of the means
        over whatever spatial domain is specified in the input DataArray
    '''
    # create array of weights for each grid cell based on latitude
    weights = np.cos(np.deg2rad(DataArray.lat))
    weights.name = "weights"
    array_weighted = DataArray.weighted(weights)
    weighted_mean = array_weighted.mean(("lon", "lat"))
    return weighted_mean


# Remove seasonal cycle
trefht_monthly_base = {}
trefht_monthly_st = {}
trefht_monthly_ml = {}
atm_monthly_base_trefht_movingavg = {}
atm_monthly_st_trefht_movingavg = {}
atm_monthly_ml_trefht_movingavg = {}
for yr in yr_keys:
        print(yr)
        trefht_monthly_base[yr] = calc_weighted_mean(atm_monthly_base[yr]['TREFHT'])
        trefht_monthly_st[yr] = calc_weighted_mean(atm_monthly_st[yr]['TREFHT'])
        trefht_monthly_ml[yr] = calc_weighted_mean(atm_monthly_ml[yr]['TREFHT'])
        for mon in list(atm_monthly_base_clim[yr]['TREFHT'].month.values):
                trefht_monthly_base[yr].loc[{'time':[t for t in trefht_monthly_base[yr].time.values if t.month==mon]}]-=calc_weighted_mean(atm_monthly_base_clim[yr]['TREFHT'].sel(month=mon)).values
                trefht_monthly_st[yr].loc[{'time':[t for t in trefht_monthly_st[yr].time.values if t.month==mon]}]-=calc_weighted_mean(atm_monthly_st_clim[yr]['TREFHT'].sel(month=mon)).values
                trefht_monthly_ml[yr].loc[{'time':[t for t in trefht_monthly_ml[yr].time.values if t.month==mon]}]-=calc_weighted_mean(atm_monthly_ml_clim[yr]['TREFHT'].sel(month=mon)).values
        # Check if removing seasonal cycle worked (should be close to 0)
        print(np.nanmean(trefht_monthly_base[yr]))
        print(np.nanmean(trefht_monthly_st[yr]))
        print(np.nanmean(trefht_monthly_ml[yr]))
        # Calculate 12-month rolling average
        n = 12
        atm_monthly_base_trefht_movingavg[yr] = moving_average(trefht_monthly_base[yr].values, n=n)
        atm_monthly_st_trefht_movingavg[yr] = moving_average(trefht_monthly_st[yr].values, n=n)
        atm_monthly_ml_trefht_movingavg[yr] = moving_average(trefht_monthly_ml[yr].values, n=n)


# FIG S8
ymax = 1
ymin=-1
fig = plt.figure(figsize=(18,15));
yr = yr_keys[1]
ax = fig.add_subplot(3, 2, 1)
plt.plot(np.arange(1,len(trefht_monthly_base[yr].time)+1), trefht_monthly_base[yr], c='g', alpha=0.3);
plt.plot(np.arange(6,len(trefht_monthly_base[yr].time)-5), atm_monthly_base_trefht_movingavg[yr], c='g', label='Control');
plt.plot(np.arange(1,len(trefht_monthly_base[yr].time)+1), np.repeat(0,len(trefht_monthly_base[yr].time)), c='grey', linestyle='dotted')
plt.plot(np.arange(len(trefht_monthly_base[yr].time)-(30*12), len(trefht_monthly_base[yr].time)), np.repeat(trefht_monthly_base[yr][-30*12:].mean().values,len(trefht_monthly_base[yr][-30*12:].time)), c='g', linestyle='--')
plt.legend(loc='upper left',fontsize=7);
plt.ylim(ymin,ymax);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
plt.ylabel('Temperature [\N{DEGREE SIGN}C]',fontsize=7); #plt.xlabel('Time [months]');
ax.set_title('a', fontsize=10, fontweight='bold',loc='left');
ax = fig.add_subplot(3, 2, 3)
plt.plot(np.arange(1,len(trefht_monthly_st[yr].time)+1), trefht_monthly_st[yr], c='darkblue', alpha=0.3);
plt.plot(np.arange(6,len(trefht_monthly_st[yr].time)-5), atm_monthly_st_trefht_movingavg[yr], c='darkblue', label='Subtropical MCB');
plt.plot(np.arange(1,len(trefht_monthly_st[yr].time)+1), np.repeat(0,len(trefht_monthly_st[yr].time)), c='grey', linestyle='dotted')
plt.plot(np.arange(len(trefht_monthly_st[yr].time)-(30*12), len(trefht_monthly_st[yr].time)), np.repeat(trefht_monthly_st[yr][-30*12:].mean().values,len(trefht_monthly_st[yr][-30*12:].time)), c='darkblue', linestyle='--')
plt.legend(loc='upper left',fontsize=7);
plt.ylabel('Temperature [\N{DEGREE SIGN}C]',fontsize=7); #plt.xlabel('Time [months]');
plt.ylim(ymin,ymax);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
ax.set_title('b', fontsize=10, fontweight='bold',loc='left');
ax = fig.add_subplot(3, 2, 5)
plt.plot(np.arange(1,len(trefht_monthly_ml[yr].time)+1), trefht_monthly_ml[yr], c='orange', alpha=0.3);
plt.plot(np.arange(6,len(trefht_monthly_ml[yr].time)-5), atm_monthly_ml_trefht_movingavg[yr], c='orange', label='Mid-latitude MCB');
plt.plot(np.arange(1,len(trefht_monthly_ml[yr].time)+1), np.repeat(0,len(trefht_monthly_ml[yr].time)), c='grey', linestyle='dotted')
plt.plot(np.arange(len(trefht_monthly_ml[yr].time)-(30*12), len(trefht_monthly_ml[yr].time)), np.repeat(trefht_monthly_ml[yr][-30*12:].mean().values,len(trefht_monthly_ml[yr][-30*12:].time)), c='orange', linestyle='--')
plt.legend(loc='upper left',fontsize=7);
plt.ylabel('Temperature [\N{DEGREE SIGN}C]',fontsize=7); plt.xlabel('Time [months]',fontsize=7);
plt.ylim(ymin,ymax);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
ax.set_title('c', fontsize=10, fontweight='bold',loc='left');

yr = yr_keys[0]
ax = fig.add_subplot(3, 2, 2)
plt.plot(np.arange(1,len(trefht_monthly_base[yr].time)+1), trefht_monthly_base[yr], c='g', alpha=0.3);
plt.plot(np.arange(6,len(trefht_monthly_base[yr].time)-5), atm_monthly_base_trefht_movingavg[yr], c='g', label='Control');
plt.plot(np.arange(1,len(trefht_monthly_base[yr].time)+1), np.repeat(0,len(trefht_monthly_base[yr].time)), c='grey', linestyle='dotted')
plt.plot(np.arange(len(trefht_monthly_base[yr].time)-(30*12), len(trefht_monthly_base[yr].time)), np.repeat(trefht_monthly_base[yr][-30*12:].mean().values,len(trefht_monthly_base[yr][-30*12:].time)), c='g', linestyle='--')
plt.legend(loc='upper left',fontsize=7);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
plt.ylim(ymin,ymax);
ax.set_title('d', fontsize=10, fontweight='bold',loc='left');
ax = fig.add_subplot(3, 2, 4)
plt.plot(np.arange(1,len(trefht_monthly_st[yr].time)+1), trefht_monthly_st[yr], c='darkblue', alpha=0.3);
plt.plot(np.arange(6,len(trefht_monthly_st[yr].time)-5), atm_monthly_st_trefht_movingavg[yr], c='darkblue', label='Subtropical MCB');
plt.plot(np.arange(1,len(trefht_monthly_st[yr].time)+1), np.repeat(0,len(trefht_monthly_st[yr].time)), c='grey', linestyle='dotted')
plt.plot(np.arange(len(trefht_monthly_st[yr].time)-(30*12), len(trefht_monthly_st[yr].time)), np.repeat(trefht_monthly_st[yr][-30*12:].mean().values,len(trefht_monthly_st[yr][-30*12:].time)), c='darkblue', linestyle='--')
plt.legend(loc='upper left',fontsize=7);
plt.ylim(ymin,ymax);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
ax.set_title('e', fontsize=10, fontweight='bold',loc='left');
ax = fig.add_subplot(3, 2, 6)
plt.plot(np.arange(1,len(trefht_monthly_ml[yr].time)+1), trefht_monthly_ml[yr], c='orange', alpha=0.3);
plt.plot(np.arange(6,len(trefht_monthly_ml[yr].time)-5), atm_monthly_ml_trefht_movingavg[yr], c='orange', label='Mid-latitude MCB');
plt.plot(np.arange(1,len(trefht_monthly_ml[yr].time)+1), np.repeat(0,len(trefht_monthly_ml[yr].time)), c='grey', linestyle='dotted')
plt.plot(np.arange(len(trefht_monthly_ml[yr].time)-(30*12), len(trefht_monthly_ml[yr].time)), np.repeat(trefht_monthly_ml[yr][-30*12:].mean().values,len(trefht_monthly_ml[yr][-30*12:].time)), c='orange', linestyle='--')
plt.legend(loc='upper left',fontsize=7);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
plt.ylim(ymin,ymax);
plt.xlabel('Time [months]',fontsize=7);
ax.set_title('f', fontsize=10, fontweight='bold',loc='left');
plt.tight_layout();

