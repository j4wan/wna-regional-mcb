### PURPOSE: Main script to analyze and plot CESM2 ocn output
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
from matplotlib.lines import Line2D
# turn interactive plotting on/off
plt.ion();


#%% READ IN CONVERTED DATA FROM MCB SUBTROPICAL AND MIDLATITUDE CASES AND CONTROL CASE
# Create dictionaries to hold both 2010 and 2050 output
ocn_monthly_base = {}
ocn_monthly_st = {}
ocn_monthly_ml = {}

# Hard code the exact time periods you want to read in for each simulation
yr_keys = ['2050', '2010']
sim_length = 30 #years
wd_data = '../_data/'
for yr in yr_keys:
    if yr=='2050':
        ## 1) CONTROL CASE
        casename_base = 'b.e22.B2000.f09_g17.2050cycle.001'
        ocn_monthly_base[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.002.ocn.h0.processed.0056-0085.nc')
        ## 2) MCB SUBTROPICAL SEEDING CASE
        casename_st = 'b.e22.B2000.f09_g17.2050cycle.MCB-mask15.001'
        ocn_monthly_st[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.MCB-mask15.002.ocn.h0.processed.0056-0085.nc')
        ## 3) MCB MID-LATITUDE SEEDING CASE
        casename_ml = 'b.e22.B2000.f09_g17.2050cycle.MCB-mask16.001'
        ocn_monthly_ml[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2010cycle.MCB-mask16.002.ocn.h0.processed.0056-0085.nc')
    elif yr=='2010':
        ## 1) CONTROL CASE
        casename_base = 'b.e22.B2000.f09_g17.2010cycle.002'
        ocn_monthly_base[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.001.ocn.h0.processed.0056-0085.nc')
        ## 2) MCB SUBTROPICAL SEEDING CASE
        casename_st = 'b.e22.B2000.f09_g17.2010cycle.MCB-mask15.002'
        ocn_monthly_st[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.MCB-mask15.001.ocn.h0.processed.0056-0085.nc')
        ## 3) MCB MID-LATITUDE SEEDING CASE
        casename_ml = 'b.e22.B2000.f09_g17.2010cycle.MCB-mask16.002'
        ocn_monthly_ml[yr] = xr.open_dataset(wd_data+'b.e22.B2000.f09_g17.2050cycle.MCB-mask16.001.ocn.h0.processed.0056-0085.nc')


## Calculate differences for all variables
var_list = ['SSH','MOC','N_HEAT'] 
ocn_monthly_st_anom = {}
ocn_monthly_ml_anom = {}
for yr in yr_keys:
    ocn_monthly_st_anom[yr] = {}
    ocn_monthly_ml_anom[yr] = {}
    for var in var_list:
        print(var)
        ocn_monthly_st_anom[yr][var] = ocn_monthly_st[yr][var] - ocn_monthly_base[yr][var]
        ocn_monthly_ml_anom[yr][var] = ocn_monthly_ml[yr][var] - ocn_monthly_base[yr][var]


#%% PLOT FIGURES
## FIG S1: N_heat transport
## Plot change in zonal mean ocean heat transport for AMOC
plt.subplots(1,1,figsize=(8.8,7.3));
## Plot ATLANTIC differences
plt.subplot(1,1,1);
yr='2050'
plt.plot(ocn_monthly_ml_anom[yr]['N_HEAT'].lat_aux_grid,fun.weighted_temporal_mean(ocn_monthly_ml_anom[yr]['N_HEAT']).mean(dim='time').isel(transport_reg=1,transport_comp=1),\
         linestyle='dotted',linewidth=2,color='orange',label='Mid-latitude MCB '+str(yr));
plt.plot(ocn_monthly_st_anom[yr]['N_HEAT'].lat_aux_grid,fun.weighted_temporal_mean(ocn_monthly_st_anom[yr]['N_HEAT']).mean(dim='time').isel(transport_reg=1,transport_comp=1),\
         linestyle='dotted',linewidth=2,color='darkblue',label='Subtropical MCB '+str(yr));
yr='2010'
plt.plot(ocn_monthly_ml_anom[yr]['N_HEAT'].lat_aux_grid,fun.weighted_temporal_mean(ocn_monthly_ml_anom[yr]['N_HEAT']).mean(dim='time').isel(transport_reg=1,transport_comp=1),\
         linestyle='solid',linewidth=2,color='orange',label='Mid-latitude MCB '+str(yr));
plt.plot(ocn_monthly_st_anom[yr]['N_HEAT'].lat_aux_grid,fun.weighted_temporal_mean(ocn_monthly_st_anom[yr]['N_HEAT']).mean(dim='time').isel(transport_reg=1,transport_comp=1),\
         linestyle='solid',linewidth=2,color='darkblue',label='Subtropical MCB '+str(yr));
plt.xlabel('Latitude',fontsize=7);plt.ylabel('$\Delta$PW',fontsize=7);
plt.xlim(-60,80);plt.ylim(-.1,.3);
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.axhline(0, c = 'grey', linestyle='--', linewidth=1, alpha=0.8);
plt.legend([Line2D([0], [0],color='orange',fillstyle='full',linestyle='None',marker='s',markersize=7),\
            Line2D([0], [0],color='darkblue',fillstyle='full',linestyle='None',marker='s',markersize=7),\
            Line2D([0], [0],color='k',linestyle='solid'),\
            Line2D([0], [0],color='k',linestyle='dotted')],\
            ['Mid-latitude MCB', 'Subtropical MCB', '2010','2050'], loc='upper left',\
    ncol=1, fancybox=True, shadow=False, fontsize=7);
plt.tight_layout();


## MOC ANALYSIS
# Function to calculate 12-month rolling average
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Reference point for AMOC time series
moc_z_ref = 1000 #m
lat_aux_grid_ref = 35 #N


# Define lat variable
lat = ocn_monthly_base['2010']['MOC'].lat_aux_grid
moc_z = ocn_monthly_base['2010']['MOC'].moc_z


# Are you plotting time series for reference point, max, or min AMOC strength?
# t_option = input('ref, mean, max, or min?: ')
t_option = 'max'


# FIG S2
plt.subplots(1,2,sharey=True,figsize=(18,6));
plt.subplot(1,2,1);
yr='2010'
# CONTROL
if t_option=='ref':
    in_xr = ocn_monthly_base[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='mean':
    in_xr = ocn_monthly_base[yr]['MOC'].where((lat>15)&(lat<60)&(moc_z<2500),drop=True).isel(transport_reg=1,moc_comp=0).mean(dim=('moc_z','lat_aux_grid'))
elif t_option=='max':
    in_xr = ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    in_xr = ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
plt.plot(np.arange(1,len(in_xr.time)+1),in_xr,linewidth=1,color='green',alpha=0.2)
plt.plot(np.arange(6,len(in_xr.time)-5), moving_average(in_xr.values,n=12),linewidth=2,color='green')
# ML MCB
if t_option=='ref':
    in_xr = ocn_monthly_ml[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='mean':
    in_xr = ocn_monthly_ml[yr]['MOC'].where((lat>15)&(lat<60)&(moc_z<2500),drop=True).isel(transport_reg=1,moc_comp=0).mean(dim=('moc_z','lat_aux_grid'))
elif t_option=='max':
    in_xr = ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    in_xr = ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
plt.plot(np.arange(1,len(in_xr.time)+1),in_xr,linewidth=1,color='orange',alpha=0.2)
plt.plot(np.arange(6,len(in_xr.time)-5), moving_average(in_xr.values,n=12),linewidth=2,color='orange')
# ST MCB
if t_option=='ref':
    in_xr = ocn_monthly_st[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='mean':
    in_xr = ocn_monthly_st[yr]['MOC'].where((lat>15)&(lat<60)&(moc_z<2500),drop=True).isel(transport_reg=1,moc_comp=0).mean(dim=('moc_z','lat_aux_grid'))
elif t_option=='max':
    in_xr = ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    in_xr = ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
plt.plot(np.arange(1,len(in_xr.time)+1),in_xr,linewidth=1,color='darkblue',alpha=0.2)
plt.plot(np.arange(6,len(in_xr.time)-5), moving_average(in_xr.values,n=12),linewidth=2,color='darkblue')
# Aesthetics
if t_option=='ref':
    plt.ylim(-10,30);
elif t_option=='max':
    plt.ylim(5,35);
elif t_option=='min':
    plt.ylim(-20,0);
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.xlabel('Months', fontsize=7);plt.ylabel('Sv',fontsize=7);
plt.title('a',fontsize=10,fontweight='bold',loc='left');
yr='2050'
plt.subplot(1,2,2);
# CONTROL
if t_option=='ref':
    in_xr = ocn_monthly_base[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='mean':
    in_xr = ocn_monthly_base[yr]['MOC'].where((lat>15)&(lat<60)&(moc_z<2500),drop=True).isel(transport_reg=1,moc_comp=0).mean(dim=('moc_z','lat_aux_grid'))
elif t_option=='max':
    in_xr = ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    in_xr = ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
plt.plot(np.arange(1,len(in_xr.time)+1),in_xr,linewidth=1,color='green',alpha=0.2)
plt.plot(np.arange(6,len(in_xr.time)-5), moving_average(in_xr.values,n=12),linewidth=2,color='green')
# ML MCB
if t_option=='ref':
    in_xr = ocn_monthly_ml[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='mean':
    in_xr = ocn_monthly_ml[yr]['MOC'].where((lat>15)&(lat<60)&(moc_z<2500),drop=True).isel(transport_reg=1,moc_comp=0).mean(dim=('moc_z','lat_aux_grid'))
elif t_option=='max':
    in_xr = ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    in_xr = ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
plt.plot(np.arange(1,len(in_xr.time)+1),in_xr,linewidth=1,color='orange',alpha=0.2)
plt.plot(np.arange(6,len(in_xr.time)-5), moving_average(in_xr.values,n=12),linewidth=2,color='orange')
# ST MCB
if t_option=='ref':
    in_xr = ocn_monthly_st[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='mean':
    in_xr = ocn_monthly_st[yr]['MOC'].where((lat>15)&(lat<60)&(moc_z<2500),drop=True).isel(transport_reg=1,moc_comp=0).mean(dim=('moc_z','lat_aux_grid'))
elif t_option=='max':
    in_xr = ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    in_xr = ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
plt.plot(np.arange(1,len(in_xr.time)+1),in_xr,linewidth=1,color='darkblue',alpha=0.2)
plt.plot(np.arange(6,len(in_xr.time)-5), moving_average(in_xr.values,n=12),linewidth=2,color='darkblue')
# Aesthetics
if t_option=='ref':
    plt.ylim(-10,30);
elif t_option=='max':
    plt.ylim(5,35);
elif t_option=='min':
    plt.ylim(-20,0);
plt.xticks(fontsize=7); plt.yticks(fontsize=7);
plt.xlabel('Months', fontsize=7);#plt.ylabel('Sv',fontsize=12);
plt.title('b',fontsize=10,fontweight='bold',loc='left');
plt.legend([Line2D([0], [0],color='green',fillstyle='full',linestyle='None',marker='s',markersize=7),\
            Line2D([0], [0],color='orange',fillstyle='full',linestyle='None',marker='s',markersize=7),\
            Line2D([0], [0],color='darkblue',fillstyle='full',linestyle='None',marker='s',markersize=7)],\
            ['No MCB','Mid-latitude MCB', 'Subtropical MCB'], loc='upper right',\
    ncol=1, fancybox=True, shadow=False, fontsize=7);
plt.tight_layout();


# Quick calculation of percent change in MOC strength
if t_option=='ref':
    yr='2050'
    moc_base_2050_mean = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).mean()
    moc_base_2050_std = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).std()
    moc_ml_2050_mean = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).mean()
    moc_ml_2050_std = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).std()
    moc_st_2050_mean = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).mean()
    moc_st_2050_std = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).std()
    yr='2010'
    moc_base_2010_mean = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).mean()
    moc_base_2010_std = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).std()
    moc_ml_2010_mean = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).mean()
    moc_ml_2010_std = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).std()
    moc_st_2010_mean = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).mean()
    moc_st_2010_std = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)).std()
elif t_option=='max':
    yr='2050'
    moc_base_2050_mean = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).mean()
    moc_base_2050_std = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).std()
    moc_ml_2050_mean = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).mean()
    moc_ml_2050_std = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).std()
    moc_st_2050_mean = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).mean()
    moc_st_2050_std = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).std()
    yr='2010'
    moc_base_2010_mean = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).mean()
    moc_base_2010_std = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).std()
    moc_ml_2010_mean = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).mean()
    moc_ml_2010_std = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).std()
    moc_st_2010_mean = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).mean()
    moc_st_2010_std = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))).std()
elif t_option=='min':
    yr='2050'
    moc_base_2050_mean = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).mean()
    moc_base_2050_std = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).std()
    moc_ml_2050_mean = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).mean()
    moc_ml_2050_std = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).std()
    moc_st_2050_mean = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).mean()
    moc_st_2050_std = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).std()
    yr='2010'
    moc_base_2010_mean = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).mean()
    moc_base_2010_std = fun.weighted_temporal_mean(ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).std()
    moc_ml_2010_mean = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).mean()
    moc_ml_2010_std = fun.weighted_temporal_mean(ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).std()
    moc_st_2010_mean = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).mean()
    moc_st_2010_std = fun.weighted_temporal_mean(ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))).std()


## T-test for statistical significance
# 2010 and 2050 controls
yr='2050'
if t_option=='ref':
    moc_base_2050 = ocn_monthly_base[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='max':
    moc_base_2050 = ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    moc_base_2050 = ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
yr='2010'
if t_option=='ref':
    moc_base_2010 = ocn_monthly_base[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='max':
    moc_base_2010 = ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    moc_base_2010 = ocn_monthly_base[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
print(stats.ttest_ind(moc_base_2050.values,moc_base_2010.values).pvalue)
# 2010 ML MCB and control
yr='2010'
if t_option=='ref':
    moc_ml_2010 = ocn_monthly_ml[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='max':
    moc_ml_2010 = ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    moc_ml_2010 = ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
print(stats.ttest_ind(moc_ml_2010.values,moc_base_2010.values).pvalue)
# 2050 ML MCB and control
yr='2050'
if t_option=='ref':
    moc_ml_2050 = ocn_monthly_ml[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='max':
    moc_ml_2050 = ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    moc_ml_2050 = ocn_monthly_ml[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
print(stats.ttest_ind(moc_ml_2050.values,moc_base_2050.values).pvalue)
# 2010 ST MCB and control
yr='2010'
if t_option=='ref':
    moc_st_2010 = ocn_monthly_st[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='max':
    moc_st_2010 = ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    moc_st_2010 = ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
print(stats.ttest_ind(moc_st_2010.values,moc_base_2010.values).pvalue)
# 2050 ST MCB and control
yr='2050'
if t_option=='ref':
    moc_st_2050 = ocn_monthly_st[yr]['MOC'].sel(moc_z=moc_z_ref,lat_aux_grid=lat_aux_grid_ref,method='nearest').isel(transport_reg=1,moc_comp=0)
elif t_option=='max':
    moc_st_2050 = ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).max(dim=('moc_z','lat_aux_grid'))
elif t_option=='min':
    moc_st_2050 = ocn_monthly_st[yr]['MOC'].where(lat>15,drop=True).isel(transport_reg=1,moc_comp=0).min(dim=('moc_z','lat_aux_grid'))
print(stats.ttest_ind(moc_st_2050.values,moc_base_2050.values).pvalue)


# Calculate linear regresssion for each time series
slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0,len(moc_base_2010.time)),moc_base_2010.values)
print('Control 2010: ', slope,p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0,len(moc_ml_2010.time)),moc_ml_2010.values)
print('ML 2010: ',slope,p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0,len(moc_st_2010.time)),moc_st_2010.values)
print('ST 2010: ',slope,p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0,len(moc_base_2050.time)),moc_base_2050.values)
print('Control 2050: ', slope,p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0,len(moc_ml_2050.time)),moc_ml_2050.values)
print('ML 2050: ',slope,p_value)
slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0,len(moc_st_2050.time)),moc_st_2050.values)
print('ST 2050: ',slope,p_value)


## FIGURE 4. AMOC contour + scatter plot of ref point 
fig = plt.figure(figsize=(18,7.5),constrained_layout=True);
gs = fig.add_gridspec(ncols=3,nrows=1);
f1_ax2 = fig.add_subplot(gs[0:1,0:2])
# Plot contour plot of AMOC response from 2050 ML MCB
yr = '2050' 
in_xr = fun.weighted_temporal_mean(ocn_monthly_ml_anom[yr]['MOC'].isel(transport_reg=1,moc_comp=0)).mean(dim='time')
moc_z_mesh, lat_aux_grid_mesh = np.meshgrid(in_xr.lat_aux_grid, in_xr.moc_z)
p = f1_ax2.pcolormesh(moc_z_mesh,lat_aux_grid_mesh,in_xr,cmap='Oranges',vmin=0,vmax=6);
CS = f1_ax2.contour(in_xr.lat_aux_grid,in_xr.moc_z,in_xr, 5, linewidths = 1, colors='k');
f1_ax2.clabel(CS, inline=True, fontsize=7);
# Add reference point
if t_option=='ref':
    plt.scatter(lat_aux_grid_ref,moc_z_ref,color='cyan',marker='*',s=200);
plt.gca().invert_yaxis();
plt.ylim(4000,0);#plt.xlim(-30,75);
plt.xticks(fontsize=7);plt.yticks(fontsize=7);
plt.xlabel('Latitude',fontsize=7);
plt.ylabel('Depth [m]',fontsize=7);
plt.title('a',fontsize=10,fontweight='bold',loc='left');
cbar=plt.colorbar(p,  orientation='vertical', extend='max');
cbar.ax.tick_params(labelsize=7); 
# Plot scatter of AMOC strength at reference point
f1_ax1 = fig.add_subplot(gs[0:1,2:],frameon=False);
marker_size=50
for yr in yr_keys:
    if yr=='2010':
            # Add horizontal line for control run
        plt.axhline(moc_base_2010_mean, linestyle='dotted',color='grey');
        plt.scatter(0.5,moc_base_2010_mean,marker = 'D',c='green',s=marker_size,label=yr+' No MCB')
        plt.errorbar(0.5,moc_base_2010_mean,yerr=2*moc_base_2010_std,capsize=5,c='green')
        plt.scatter(1,moc_ml_2010_mean,c='orange',s=marker_size,label=yr+' ML MCB')
        plt.errorbar(1,moc_ml_2010_mean,yerr=2*moc_ml_2010_std,capsize=5,c='orange')
        plt.scatter(1.5,moc_st_2010_mean,marker = '^',c='darkblue',s=marker_size,label=yr+' ST MCB')
        plt.errorbar(1.5,moc_st_2010_mean,yerr=2*moc_st_2010_std,capsize=5,c='darkblue')
    elif yr=='2050':
        # Add horizontal line for control run
        plt.axhline(moc_base_2050_mean, linestyle='dotted',color='grey');
        plt.scatter(0.5,moc_base_2050_mean,marker = 'D',edgecolors='green',facecolors='none',s=marker_size,label=yr+' No MCB')
        plt.errorbar(0.5,moc_base_2050_mean,yerr=2*moc_base_2050_std,capsize=5,c='green')
        plt.scatter(1,moc_ml_2050_mean,edgecolors='orange',facecolors='none',s=marker_size,label=yr+' ML MCB')
        plt.errorbar(1,moc_ml_2050_mean,yerr=2*moc_ml_2050_std,capsize=5,c='orange')
        plt.scatter(1.5,moc_st_2050_mean,marker = '^',edgecolors='darkblue',facecolors='none',s=marker_size,label=yr+' ST MCB')
        plt.errorbar(1.5,moc_st_2050_mean,yerr=2*moc_st_2050_std,capsize=5,c='darkblue')
plt.tick_params(left = False,bottom=False);
plt.xticks([]);plt.yticks(fontsize=7);
plt.xlim(0,2);
plt.ylabel('Sv', fontsize=7);
f1_ax1.legend([Line2D([0], [0],color='green',fillstyle='left',linestyle='None',marker='D',markersize=7),\
               Line2D([0], [0],color='orange',fillstyle='left',linestyle='None',marker='o',markersize=7),\
               Line2D([0], [0],color='darkblue',fillstyle='left',linestyle='None',marker='^',markersize=7),\
               Line2D([0], [0],color='k',fillstyle='left',linestyle='None',marker='s',markersize=7)],\
            ['No MCB','Mid-latitude MCB', 'Subtropical MCB','2010/2050'],bbox_to_anchor =(.5,-.2), loc='lower center',\
        ncol=2, fancybox=False, shadow=False,frameon=False,fontsize=7);
xmin, xmax = f1_ax1.get_xaxis().get_view_interval();
ymin, ymax = f1_ax1.get_yaxis().get_view_interval();
f1_ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2));
plt.title('b',fontsize=10,fontweight='bold',loc='left');


## FIG ED8
fig= plt.figure(figsize=(18,13));
subplot_num=0
## ML 2010 ## 
yr = '2010' 
ax=plt.subplot(2,2,int(1+subplot_num));
in_xr = fun.weighted_temporal_mean(ocn_monthly_ml_anom[yr]['MOC'].isel(transport_reg=1,moc_comp=0)).mean(dim='time')
moc_z_mesh, lat_aux_grid_mesh = np.meshgrid(in_xr.lat_aux_grid, in_xr.moc_z)
p = ax.pcolormesh(moc_z_mesh,lat_aux_grid_mesh,in_xr,cmap='PuOr_r',vmin=-10,vmax=10);
CS = ax.contour(in_xr.lat_aux_grid,in_xr.moc_z,in_xr, 5, linewidths = 1, colors='k');
ax.clabel(CS, inline=True, fontsize=7);
plt.gca().invert_yaxis();
plt.ylim(4000,0);#plt.xlim(-30,75);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
# plt.xlabel('Latitude');
plt.ylabel('Depth (m)',fontsize=7);
plt.title('a',fontsize=10,fontweight='bold',loc='left');
subplot_num += 1
## ML 2050 ##
yr = '2050' 
ax=plt.subplot(2,2,int(1+subplot_num));
in_xr = fun.weighted_temporal_mean(ocn_monthly_ml_anom[yr]['MOC'].isel(transport_reg=1,moc_comp=0)).mean(dim='time')
moc_z_mesh, lat_aux_grid_mesh = np.meshgrid(in_xr.lat_aux_grid, in_xr.moc_z)
p = ax.pcolormesh(moc_z_mesh,lat_aux_grid_mesh,in_xr,cmap='PuOr_r',vmin=-10,vmax=10);
CS = ax.contour(in_xr.lat_aux_grid,in_xr.moc_z,in_xr, 5, linewidths = 1, colors='k');
ax.clabel(CS, inline=True, fontsize=7);
plt.gca().invert_yaxis();
plt.ylim(4000,0);#plt.xlim(-30,75);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
plt.title('b',fontsize=10,fontweight='bold',loc='left');
subplot_num += 1
# Controls (2050-2010)
ax=plt.subplot(2,2,int(1+subplot_num));
in_xr = fun.weighted_temporal_mean(ocn_monthly_base['2050']['MOC']).mean(dim='time').isel(transport_reg=1,moc_comp=0) - fun.weighted_temporal_mean(ocn_monthly_base['2010']['MOC']).mean(dim='time').isel(transport_reg=1,moc_comp=0)
moc_z_mesh, lat_aux_grid_mesh = np.meshgrid(in_xr.lat_aux_grid, in_xr.moc_z)
p = ax.pcolormesh(moc_z_mesh,lat_aux_grid_mesh,in_xr,cmap='PuOr_r',vmin=-10,vmax=10);
CS = ax.contour(in_xr.lat_aux_grid,in_xr.moc_z,in_xr, 5, linewidths = 1, colors='k');
ax.clabel(CS, inline=True, fontsize=7);
plt.gca().invert_yaxis();
plt.ylim(4000,0);#plt.xlim(-30,75);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
plt.xlabel('Latitude',fontsize=7);
plt.ylabel('Depth (m)',fontsize=7);
plt.title('c',fontsize=10,fontweight='bold',loc='left');
subplot_num += 1
# 2050 ML MCB - 2010 No MCB
ax=plt.subplot(2,2,int(1+subplot_num));
in_xr = fun.weighted_temporal_mean(ocn_monthly_ml['2050']['MOC']).mean(dim='time').isel(transport_reg=1,moc_comp=0) - fun.weighted_temporal_mean(ocn_monthly_base['2010']['MOC']).mean(dim='time').isel(transport_reg=1,moc_comp=0)
moc_z_mesh, lat_aux_grid_mesh = np.meshgrid(in_xr.lat_aux_grid, in_xr.moc_z)
p = ax.pcolormesh(moc_z_mesh,lat_aux_grid_mesh,in_xr,cmap='PuOr_r',vmin=-10,vmax=10);
CS = ax.contour(in_xr.lat_aux_grid,in_xr.moc_z,in_xr, 5, linewidths = 1, colors='k');
ax.clabel(CS, inline=True, fontsize=7);
plt.gca().invert_yaxis();
plt.ylim(4000,0);#plt.xlim(-30,75);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
plt.xlabel('Latitude',fontsize=7);
# plt.ylabel('Depth (m)',fontsize=7);
plt.title('d',fontsize=10,fontweight='bold',loc='left');
# Plot aesthetics
fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.3,hspace=0.35);
cbar_ax = fig.add_axes([0.075, 0.07, 0.85, 0.03]) #rect kwargs [left, bottom, width, height];
cbar = fig.colorbar(p, cax=cbar_ax,orientation='horizontal', extend='both',pad=0.15);
cbar.ax.tick_params(labelsize=7); cbar.set_label(label='Sv', size=7)


## FIG ED9
## CHANGE IN AMOC CONTOUR PLOTS 
fig= plt.figure(figsize=(18,13));
subplot_num=0
## ST 2010 ## 
yr = '2010' 
ax=plt.subplot(2,2,int(1+subplot_num));
in_xr = fun.weighted_temporal_mean(ocn_monthly_st_anom[yr]['MOC'].isel(transport_reg=1,moc_comp=0)).mean(dim='time')
moc_z_mesh, lat_aux_grid_mesh = np.meshgrid(in_xr.lat_aux_grid, in_xr.moc_z)
p = ax.pcolormesh(moc_z_mesh,lat_aux_grid_mesh,in_xr,cmap='PuOr_r',vmin=-10,vmax=10);
CS = ax.contour(in_xr.lat_aux_grid,in_xr.moc_z,in_xr, 5, linewidths = 1, colors='k');
ax.clabel(CS, inline=True, fontsize=7);
plt.gca().invert_yaxis();
plt.ylim(4000,0);#plt.xlim(-30,75);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
plt.ylabel('Depth (m)',fontsize=7);
plt.title('a',fontsize=10,fontweight='bold',loc='left');
subplot_num += 1
## st 2050 ##
yr = '2050' 
ax=plt.subplot(2,2,int(1+subplot_num));
in_xr = fun.weighted_temporal_mean(ocn_monthly_st_anom[yr]['MOC'].isel(transport_reg=1,moc_comp=0)).mean(dim='time')
moc_z_mesh, lat_aux_grid_mesh = np.meshgrid(in_xr.lat_aux_grid, in_xr.moc_z)
p = ax.pcolormesh(moc_z_mesh,lat_aux_grid_mesh,in_xr,cmap='PuOr_r',vmin=-10,vmax=10);
CS = ax.contour(in_xr.lat_aux_grid,in_xr.moc_z,in_xr, 5, linewidths = 1, colors='k');
ax.clabel(CS, inline=True, fontsize=7);
plt.gca().invert_yaxis();
plt.ylim(4000,0);#plt.xlim(-30,75);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
plt.title('b',fontsize=10,fontweight='bold',loc='left');
subplot_num += 1
# Controls (2050-2010)
ax=plt.subplot(2,2,int(1+subplot_num));
in_xr = fun.weighted_temporal_mean(ocn_monthly_base['2050']['MOC']).mean(dim='time').isel(transport_reg=1,moc_comp=0) - fun.weighted_temporal_mean(ocn_monthly_base['2010']['MOC']).mean(dim='time').isel(transport_reg=1,moc_comp=0)
moc_z_mesh, lat_aux_grid_mesh = np.meshgrid(in_xr.lat_aux_grid, in_xr.moc_z)
p = ax.pcolormesh(moc_z_mesh,lat_aux_grid_mesh,in_xr,cmap='PuOr_r',vmin=-10,vmax=10);
CS = ax.contour(in_xr.lat_aux_grid,in_xr.moc_z,in_xr, 5, linewidths = 1, colors='k');
ax.clabel(CS, inline=True, fontsize=7);
plt.gca().invert_yaxis();
plt.ylim(4000,0);#plt.xlim(-30,75);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
plt.xlabel('Latitude',fontsize=7);
plt.ylabel('Depth (m)',fontsize=7);
plt.title('c',fontsize=10,fontweight='bold',loc='left');
subplot_num += 1
# 2050 ST MCB - 2010 No MCB
ax=plt.subplot(2,2,int(1+subplot_num));
in_xr = fun.weighted_temporal_mean(ocn_monthly_st['2050']['MOC']).mean(dim='time').isel(transport_reg=1,moc_comp=0) - fun.weighted_temporal_mean(ocn_monthly_base['2010']['MOC']).mean(dim='time').isel(transport_reg=1,moc_comp=0)
moc_z_mesh, lat_aux_grid_mesh = np.meshgrid(in_xr.lat_aux_grid, in_xr.moc_z)
p = ax.pcolormesh(moc_z_mesh,lat_aux_grid_mesh,in_xr,cmap='PuOr_r',vmin=-10,vmax=10);
CS = ax.contour(in_xr.lat_aux_grid,in_xr.moc_z,in_xr, 5, linewidths = 1, colors='k');
ax.clabel(CS, inline=True, fontsize=7);
plt.gca().invert_yaxis();
plt.ylim(4000,0);#plt.xlim(-30,75);
plt.yticks(fontsize=7);plt.xticks(fontsize=7);
plt.xlabel('Latitude',fontsize=7);
plt.title('d',fontsize=10,fontweight='bold',loc='left');
# Plot aesthetics
fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.3,hspace=0.35);
cbar_ax = fig.add_axes([0.075, 0.07, 0.85, 0.03]) #rect kwargs [left, bottom, width, height];
cbar = fig.colorbar(p, cax=cbar_ax,orientation='horizontal', extend='both',pad=0.15);
cbar.ax.tick_params(labelsize=7); cbar.set_label(label='Sv', size=7)

