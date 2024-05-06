### PURPOSE: Script to store functions used in all analysis scripts in _code/
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 04/30/2024


### TABLE OF CONTENTS ### -------------------------------------------------------------
# 1) reorient_netCDF : orient and save netcdf wrt -180,180 longitude
# 2a) read_ensemble : read in all files for a given variable and store as xarray w/ dims [time, lat, lon, case]
# 2b) read_ensemble_lev : read in all files for a given variable and store as xarray w/ dims [time, lev, lat, lon, case]
# 3) globalarea: computes grid cell area for globe
# 4) dateshift_netCDF : shift and save netcdf with dates at midpoint of month
# 5) plot_monthly_maps: plot maps of global/regional monthly means/anomalies
# 6) plot_contourf_monthly_maps: plot maps of global/regional monthly means/anomalies
# 7) plot_annual_maps: plot maps of global/regional monthly means/anomalies
# 8) count_cumsum: calculate cumulative sum of the number occurences that a condition is met
# 9) plot_panel_maps: plot maps of global/regional monthly means/anomalies by subplot panel
# 10) calc_weighted_mean_sd: calculate area weighted mean and standard deviation
# 11) calc_weighted_mean_tseries: calculate area weighted mean for time series
# 12) plot_west_coast_panel_maps: plot maps of west coast monthly means/anomalies by subplot panel
# 13) weighted_temporal_mean: calculate day-weighted mean
# 14) weighted_temporal_mean_clim: calculate day-weighted mean for monthly climatology
### ------------------------------------------------------------------------------- ###


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
import glob
import datetime
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
from matplotlib import ticker


# 1) reorient_netCDF : orient and save netcdf wrt -180,180 longitude
def reorient_netCDF(fp):
    """
    Function to orient and save netcdf wrt -180,180 longitude.
    :param fp: dataarray to be reoriented
    """
    f = fp
    if np.max(f.coords['lon'])>180:
        new_lon = [-360.00 + num if num > 180 else num for num in f.coords['lon'].values]
        f = f.assign_coords({'lon':new_lon})
        f.assign_coords(lon=(np.mod(f.lon + 180, 360) - 180))
        f = f.sortby(f.coords['lon'])
    return f



# 2a) read_ensemble : read in all files for a given variable and store as xarray w/ dims [time, lat, lon, case]
def read_ensemble(cesm_var,fpath):
    """
    Function to read all files for a given variable and store as xarray w/ dims [time, lat, lon, case].
    :param cesm_var: string of CESM2 variable name
    :param fpath: string of file directory
    NOTE: edited 10/11/23 to require fpath to be more versatile
    """
    v = cesm_var
    # Get list of all case runs for each variable
    fname = glob.glob(fpath+'*.'+v+'.*.nc')
    models = []
    for i in fname:
        if v in i:
            start = i.find('LE2-') + len('LE2-')
            end = i.find('.cam')
            tmp = i[start:end]
            if tmp not in models:
                models.append(tmp)
    models = sorted(models)
    print(models)

    # Create list of each xarray
    # Set first file to compare time, lat, lon
    file_check = dateshift_netCDF(reorient_netCDF(xr.open_mfdataset(glob.glob(fpath+'*'+models[0]+'*.'+v+'.*.nc'))[v]))
    var_list = []
    for i in models:
        filename = glob.glob(fpath+'*'+i+'*.'+v+'.*.nc')
        tmp = dateshift_netCDF(reorient_netCDF(xr.open_mfdataset(filename)[v]))
        time_check = np.mean(tmp.time==file_check.time)
        lat_check = np.mean(tmp.lat==file_check.lat)
        lon_check = np.mean(tmp.lon==file_check.lon)
        if time_check == lat_check == lon_check == 1:
            var_list.append(tmp)
        else:
            print('Error: incompatible dimensions')

    # Convert lists to 4D array where the dimensions are [time,lat,lon,case #]
    var_all_cases = np.stack(var_list,axis=-1)

    var_combined_out = xr.DataArray(name=v,data=var_all_cases,dims=["time","lat","lon","case"],\
            coords=dict(lon=file_check.lon,lat=file_check.lat,time=file_check.time,case=models),\
            attrs=dict(long_name=file_check.long_name,units=file_check.units))
    
    return(var_combined_out)



# 2b) read_ensemble_lev : read in all files for a given variable and store as xarray w/ dims [time, lev, lat, lon, case]
def read_ensemble_lev(cesm_var,fpath):
    """
    Function to read all files for a given variable and store as xarray w/ dims [time, lev, lat, lon, case].
    :param cesm_var: string of CESM2 variable name
    :param fpath: string of file directory
    NOTE: edited 10/11/23 to require fpath to be more versatile
    """
    v = cesm_var
    # Get list of all case runs for each variable
    # fname = glob.glob('/zdata/NCAR-ECIP/LENS2/*.'+v+'.*.nc')
    fname = glob.glob(fpath+'*.'+v+'.*.nc')
    models = []
    for i in fname:
        if v in i:
            start = i.find('LE2-') + len('LE2-')
            end = i.find('.cam')
            tmp = i[start:end]
            if tmp not in models:
                models.append(tmp)
    models = sorted(models)
    print(models)

    # Create list of each xarray
    # Set first file to compare time, lat, lon
    file_check = reorient_netCDF(xr.open_dataset(glob.glob(fpath+'*'+models[0]+'*.'+v+'.*.nc')[0])[v])
    var_list = []
    for i in models:
        filename = glob.glob(fpath+'*'+i+'*.'+v+'.*.nc')
        tmp = reorient_netCDF(xr.open_dataset(filename[0])[v])
        time_check = np.mean(tmp.time==file_check.time)
        lev_check = np.mean(tmp.lev==file_check.lev)
        lat_check = np.mean(tmp.lat==file_check.lat)
        lon_check = np.mean(tmp.lon==file_check.lon)
        if time_check == lev_check == lat_check == lon_check == 1:
            var_list.append(tmp)
        else:
            print('Error: incompatible dimensions')

    # Convert lists to 4D array where the dimensions are [time,lat,lon,case #]
    var_all_cases = np.stack(var_list,axis=-1)

    var_combined_out = xr.DataArray(name=v,data=var_all_cases,dims=["time","lev","lat","lon","case"],\
            coords=dict(lon=file_check.lon,lat=file_check.lat,time=file_check.time,lev=file_check.lev,case=models),\
            attrs=dict(long_name=file_check.long_name,units=file_check.units))
    
    return(var_combined_out)



# 3) globalarea: computes grid cell area for globe
def globalarea(xr_in):
    """
    globalarea(xr_in) gives the area [km^2] for each grid point of a regular global grid.
    default is to have lat/lon as 360 and 720, which is a 0.5 x 0.5 degree grid
    """
    lat_len = len(xr_in.lat)
    lon_len = len(xr_in.lon)
    dims=[lat_len,lon_len]
    theta = np.linspace(0,np.pi,dims[0]+1) # Define regular angle for latitude
    theta1 = theta[:-1]
    theta2 = theta[1:]
    Re = 6371. # radius of Earth (in km)
    dA = 2.*Re*Re*np.pi/dims[1]*(-np.cos(theta2)+np.cos(theta1))
    dAmat = np.tile(dA,(dims[1],1)).T # Repeat for each longitude
    return dAmat



# 4) dateshift_netCDF : shift and save netcdf with dates at midpoint of month
def dateshift_netCDF(fp):
    """
    Function to shift and save netcdf with dates at midpoint of month.
    :param fp: dataarray to be reoriented
    """
    f = fp
    if np.unique(fp.indexes['time'].day)[0]==1 & len(np.unique(fp.indexes['time'].day))==1:
        new_time = fp.indexes['time']-datetime.timedelta(days=16)
        f = f.assign_coords({'time':new_time})
    return f



# 5) plot_monthly_maps: plot maps of global/regional monthly means/anomalies
def plot_monthly_maps(in_xr,cmin, cmax, ccmap, plot_zoom,overlay_mask,central_lon=0,CI_in='none',CI_level='none',CI_display='none'):
    """
    Function to plot maps of global/regional ensemble mean, seasonal climatologies, and seasonal anomalies
    :param in_xr: xarray w/ dims [month,lat,lon] representing climatological monthly mean/anomaly
    :param cmin: float minimum value for ensemble mean and seasonal climatology plots
    :param cmax: float maximum value for ensemble mean and seasonal climatology plots
    :param ccmap: string colormap pallette for ensemble mean and seasonal climatology plots
    :param plot_zoom: string to specify zoom of plot ('global','conus', 'west_coast', 'pacific_ocean')
    :param overlay_mask: string to specify seeding mask to overlay (Name of mask or None if no mask is to be overlaid)
    :param central_lon: float specifying central longitude for plotting (default=0) If central_lon=180, need to add cyclical point to remove white line
    :param CI_in: xarray w/ same dims as in_xr specifying 1's where the grid cells are significant to the CI and 0's elsewhere. Default is None.
    :param CI_level: float specifying signficance level for plotting.
    :param CI_display: string specifying how to show CI (default='none'). Options include stippling significant pixels, inverted stippling where insignficant pixels are stippled, or masking out insignificant pixels.
    """

    x = in_xr
    x_ci = CI_in
    month_dict = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
    plot_proj = ccrs.PlateCarree(central_longitude=central_lon)
    

    if overlay_mask=='None':
        # Remove white line if plotting over Pacific Ocean.
        if central_lon==180:
            lat = x.lat
            lon = x.lon
            data = x
            data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
            lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)
            cm = 1/2.54  # centimeters in inches

            # fig = plt.figure(figsize=(14,8));
            fig = plt.figure(figsize=(18*cm,10*cm));
            subplot_num=0
            for i in range(len(month_dict)):
                ax = plt.subplot(3,4,int(1+subplot_num), projection=plot_proj,transform=plot_proj)
                if plot_zoom=='global':
                    if CI_display == 'mask':
                        ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                        ax.scatter(lat_mesh[::4,::6][x_ci[i,::4,::6]<=CI_level],lon_mesh[::4,::6][x_ci[i,::4,::6]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                        ax.scatter(lat_mesh[::4,::6][x_ci[i,::4,::6]>CI_level],lon_mesh[::4,::6][x_ci[i,::4,::6]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                    p.axes.set_global();
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True
                elif plot_zoom=='conus':
                    if CI_display == 'mask':
                        ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]<=CI_level],lon_mesh[:,:][x_ci[i,:,:]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]>CI_level],lon_mesh[:,:][x_ci[i,:,:]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(15,60);ax.set_xlim((-150+180),(-65+180));
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
                elif plot_zoom=='west_coast':
                    if CI_display == 'mask':
                        ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]<=CI_level],lon_mesh[:,:][x_ci[i,:,:]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]>CI_level],lon_mesh[:,:][x_ci[i,:,:]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(30,55);ax.set_xlim((-140+180),(-100+180));
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
                elif plot_zoom=='pacific_ocean':
                    if CI_display == 'mask':
                        ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                        ax.scatter(lat_mesh[::2,::3][x_ci[i,::2,::3]<=CI_level],lon_mesh[::2,::3][x_ci[i,::2,::3]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                        ax.scatter(lat_mesh[::2,::3][x_ci[i,::2,::3]>CI_level],lon_mesh[::2,::3][x_ci[i,::2,::3]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(0,80);ax.set_xlim(-80,80);
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False                    
                ax.coastlines()
                plt.title(month_dict[i+1],fontsize=7);
                subplot_num = subplot_num + 1
            fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.1,hspace=.01);
            cbar_ax = fig.add_axes([0.115, 0.15, 0.8, 0.05]) #rect kwargs [left, bottom, width, height];
            cbar = fig.colorbar(p, cax=cbar_ax,orientation='horizontal', extend='both',pad=0.15);
            cbar.ax.tick_params(labelsize=7); cbar.set_label(label=x.units, size=7)       
        # Central longitude is default 0.
        else:
            lat = x.lat
            lon = x.lon
            data = x
            lat_mesh, lon_mesh  = np.meshgrid(lon,lat)

            fig = plt.figure(figsize=(14,8));
            subplot_num=0
            for i in range(len(month_dict)):
                ax = plt.subplot(3,4,int(1+subplot_num), projection=plot_proj,transform=plot_proj)
                if plot_zoom=='global':
                    if CI_display == 'mask':
                        data = xr.where(x_ci<=CI_level,data,np.nan)
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.scatter(lat_mesh[::4,::6][x_ci[i,::4,::6]<=CI_level],lon_mesh[::4,::6][x_ci[i,::4,::6]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.scatter(lat_mesh[::4,::6][x_ci[i,::4,::6]>CI_level],lon_mesh[::4,::6][x_ci[i,::4,::6]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                    p.axes.set_global();
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True
                elif plot_zoom=='conus':
                    if CI_display == 'mask':
                        data = xr.where(x_ci<=CI_level,data,np.nan)
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]<=CI_level],lon_mesh[:,:][x_ci[i,:,:]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())                 
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]>CI_level],lon_mesh[:,:][x_ci[i,:,:]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(15,60);ax.set_xlim((-150),(-65));
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
                elif plot_zoom=='west_coast':
                    if CI_display == 'mask':
                        data = xr.where(x_ci<=CI_level,data,np.nan)
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]<=CI_level],lon_mesh[:,:][x_ci[i,:,:]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())                     
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]>CI_level],lon_mesh[:,:][x_ci[i,:,:]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(30,55);ax.set_xlim((-140),(-100));
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
                elif plot_zoom=='pacific_ocean':
                    lat = x.lat
                    lon = x.lon
                    data = x
                    data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
                    lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)
                    if CI_display == 'mask':
                        ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                        ax.scatter(lat_mesh[::2,::3][x_ci[i,::2,::3]<=CI_level],lon_mesh[::2,::3][x_ci[i,::2,::3]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                        ax.scatter(lat_mesh[::2,::3][x_ci[i,::2,::3]>CI_level],lon_mesh[::2,::3][x_ci[i,::2,::3]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(0,80);ax.set_xlim(-80,80);
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False                    
                ax.coastlines()
                plt.title(month_dict[i+1],fontsize=12);
                subplot_num = subplot_num + 1
            fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.1,hspace=.01);
            cbar_ax = fig.add_axes([0.115, 0.15, 0.8, 0.05]) #rect kwargs [left, bottom, width, height];
            cbar = fig.colorbar(p, cax=cbar_ax,orientation='horizontal', extend='both',pad=0.15);
            cbar.ax.tick_params(labelsize=7); cbar.set_label(label=x.units, size=7)
    else:
        mask =reorient_netCDF(xr.open_dataset(overlay_mask)['mask'])
        # Filter out negliglble grid boxes
        mask = xr.where(mask>0.001,mask,0)
        # Remove white line if plotting over Pacific Ocean.
        if central_lon==180:
            lat = x.lat
            lon = x.lon
            data = x
            data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
            mask_wrap, lon_wrap = add_cyclic_point(mask,coord=lon)
            lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)

            cm = 1/2.54  # centimeters in inches

            # fig = plt.figure(figsize=(14,8));
            fig = plt.figure(figsize=(18*cm,10*cm));
            subplot_num=0
            for i in range(len(month_dict)):
                ax = plt.subplot(3,4,int(1+subplot_num), projection=plot_proj,transform=plot_proj)
                if plot_zoom=='global':
                    if CI_display == 'mask':
                        ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                        ax.scatter(lat_mesh[::4,::6][x_ci[i,::4,::6]<=CI_level],lon_mesh[::4,::6][x_ci[i,::4,::6]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                        ax.scatter(lat_mesh[::4,::6][x_ci[i,::4,::6]>CI_level],lon_mesh[::4,::6][x_ci[i,::4,::6]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})  
                    p.axes.set_global();
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True
                elif plot_zoom=='conus':
                    if CI_display == 'mask':
                        ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]<=CI_level],lon_mesh[:,:][x_ci[i,:,:]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]>CI_level],lon_mesh[:,:][x_ci[i,:,:]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})  
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(15,60);ax.set_xlim((-150+180),(-65+180));
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
                elif plot_zoom=='west_coast':
                    if CI_display == 'mask':
                        ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]<=CI_level],lon_mesh[:,:][x_ci[i,:,:]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]>CI_level],lon_mesh[:,:][x_ci[i,:,:]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})  
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(30,55);ax.set_xlim((-140+180),(-100+180));
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
                elif plot_zoom=='pacific_ocean':
                    if CI_display == 'mask':
                        ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                        ax.scatter(lat_mesh[::2,::3][x_ci[i,::2,::3]<=CI_level],lon_mesh[::2,::3][x_ci[i,::2,::3]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                        ax.scatter(lat_mesh[::2,::3][x_ci[i,::2,::3]>CI_level],lon_mesh[::2,::3][x_ci[i,::2,::3]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})  
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(0,80);ax.set_xlim(-80,80);
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False                
                ax.coastlines();
                #gl = ax.gridlines(draw_labels=True,alpha=0);
                plt.title(month_dict[i+1],fontsize=7);
                subplot_num = subplot_num + 1
            fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.1,hspace=.01);
            cbar_ax = fig.add_axes([0.115, 0.15, 0.8, 0.05]) #rect kwargs [left, bottom, width, height];
            cbar = fig.colorbar(p, cax=cbar_ax,orientation='horizontal', extend='both',pad=0.15);
            cbar.ax.tick_params(labelsize=7); cbar.set_label(label=x.units, size=7)
        else:
            lat = x.lat
            lon = x.lon
            data = x
            lat_mesh, lon_mesh  = np.meshgrid(lon,lat)

            fig = plt.figure(figsize=(14,8));
            subplot_num=0
            lat_mesh, lon_mesh  = np.meshgrid(x.lon,x.lat)
            for i in range(len(month_dict)):
                ax = plt.subplot(3,4,int(1+subplot_num), projection=plot_proj,transform=plot_proj)
                if plot_zoom=='global':
                    if CI_display == 'mask':
                        data = xr.where(x_ci<=CI_level,data,np.nan)
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        ax.scatter(lat_mesh[::4,::6][x_ci[i,::4,::6]<=CI_level],lon_mesh[::4,::6][x_ci[i,::4,::6]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})                  
                        ax.scatter(lat_mesh[::4,::6][x_ci[i,::4,::6]>CI_level],lon_mesh[::4,::6][x_ci[i,::4,::6]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})  
                    p.axes.set_global();
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True
                elif plot_zoom=='conus':
                    if CI_display == 'mask':
                        data = xr.where(x_ci<=CI_level,data,np.nan)
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]<=CI_level],lon_mesh[:,:][x_ci[i,:,:]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})                     
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]>CI_level],lon_mesh[:,:][x_ci[i,:,:]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})  
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(15,60);ax.set_xlim((-150),(-65));
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
                elif plot_zoom=='west_coast':
                    if CI_display == 'mask':
                        data = xr.where(x_ci<=CI_level,data,np.nan)
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]<=CI_level],lon_mesh[:,:][x_ci[i,:,:]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})                  
                        ax.scatter(lat_mesh[:,:][x_ci[i,:,:]>CI_level],lon_mesh[:,:][x_ci[i,:,:]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon, lat, data[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                        ax.contour(lon,lat,mask[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})  
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(30,55);ax.set_xlim((-140),(-100));
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
                elif plot_zoom=='pacific_ocean':
                    lat = x.lat
                    lon = x.lon
                    data = x
                    data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
                    mask_wrap, lon_wrap = add_cyclic_point(mask,coord=lon)
                    lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)
                    if CI_display == 'mask':
                        ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                    elif CI_display == 'stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                        ax.scatter(lat_mesh[::2,::3][x_ci[i,::2,::3]<=CI_level],lon_mesh[::2,::3][x_ci[i,::2,::3]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    elif CI_display == 'inv_stipple':
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})
                        #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                        ax.scatter(lat_mesh[::2,::3][x_ci[i,::2,::3]>CI_level],lon_mesh[::2,::3][x_ci[i,::2,::3]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
                    else:
                        p = ax.pcolormesh(lon_wrap, lat, data_wrap[i,:,:],vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
                        ax.contour(lon_wrap,lat,mask_wrap[i,:,:], transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,subplot_kws={'projection':plot_proj})  
                    ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(0,80);ax.set_xlim(-80,80);
                    #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
                    #ax.yaxis.set_major_formatter(LatitudeFormatter())
                    #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False                
                ax.coastlines();
                plt.title(month_dict[i+1],fontsize=7);
                subplot_num = subplot_num + 1
            fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.1,hspace=.01);
            cbar_ax = fig.add_axes([0.115, 0.15, 0.8, 0.05]) #rect kwargs [left, bottom, width, height];
            cbar = fig.colorbar(p, cax=cbar_ax,orientation='horizontal', extend='both',pad=0.15);
            cbar.ax.tick_params(labelsize=7); cbar.set_label(label=x.units, size=7)
    return fig




# 6) plot_contourf_monthly_maps: plot maps of global/regional monthly means/anomalies
def plot_contourf_monthly_maps(in_xr,cmin, cmax, cstep, ccmap, regional):
    """
    Function to plot maps of global/regional ensemble mean, seasonal climatologies, and seasonal anomalies
    :param in_xr: xarray w/ dims [month,lat,lon] representing climatological monthly mean/anomaly
    :param cmin: float minimum value for ensemble mean and seasonal climatology plots
    :param cmax: float maximum value for ensemble mean and seasonal climatology plots
    :param ccmap: string colormap pallette for ensemble mean and seasonal climatology plots
    :param regional: boolean to determine zoom of plot (True = CONUS, False = global)
    """

    x = in_xr
    month_dict = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
    contourf_levs = np.linspace(cmin, cmax, int((cmax-cmin)/cstep+cmax))
    fig = plt.figure(figsize=(14,10));
    subplot_num=0
    for i in range(len(month_dict)):
        ax = plt.subplot(3,4,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
        p =x.isel(month=i).plot(levels=contourf_levs,cmap=ccmap,add_colorbar=False)#,\
            #cbar_kwargs={'pad':0.15, 'orientation' : 'horizontal','label':cldliq_850.units,'extend':'both'});
        ax.coastlines()
        if regional==False:
            p.axes.set_global();
        elif regional==True:
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(15,60);ax.set_xlim(-160,-90);
        gl = ax.gridlines(draw_labels=True,alpha=0);
        gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True
        plt.title(month_dict[i+1],fontsize=12);
        subplot_num = subplot_num + 1
    fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.25);
    cbar_ax = fig.add_axes([0.075, 0.1, 0.85, 0.05]) #rect kwargs [left, bottom, width, height];
    fig.colorbar(p, cax=cbar_ax,orientation='horizontal', label=x.units, extend='both',pad=0.15);

    return



# 7) plot_annual_maps: plot maps of global/regional monthly means/anomalies
def plot_annual_maps(in_xr,cmin, cmax, ccmap, plot_zoom,central_lon=0,CI_in='none',CI_level='none',CI_display='none',mean_val='none'):
    """
    Function to plot maps of global/regional ensemble mean and anomalies
    :param in_xr: xarray w/ dims [lat,lon] representing annual mean/anomaly
    :param cmin: float minimum value for ensemble mean and seasonal climatology plots
    :param cmax: float maximum value for ensemble mean and seasonal climatology plots
    :param ccmap: string colormap pallette for ensemble mean and anomaly plots
    :param plot_zoom: string to specify zoom of plot ('global','conus', 'west_coast', 'pacific_ocean')
    :param central_lon: float specifying central longitude for plotting (default=0) If central_lon=180, need to add cyclical point to remove white line
    :param CI_in: xarray w/ same dims as in_xr specifying 1's where the grid cells are significant to the CI and 0's elsewhere. Default is None.
    :param CI_level: float specifying signficance level for plotting.
    :param CI_display: string specifying how to show CI (default='none'). Options include stippling significant pixels, inverted stippling where insignficant pixels are stippled, or masking out insignificant pixels.
    :param mean_val: default is none. If not, specify array(mean, std) to put mean value in top right corner.
    """

    x = in_xr
    x_ci = CI_in
    plot_proj = ccrs.PlateCarree(central_longitude=central_lon)
    

    # Remove white line if plotting over Pacific Ocean.
    if central_lon==180:
        lat = x.lat
        lon = x.lon
        data = x
        data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
        lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)

        fig = plt.figure(figsize=(8,7));
        ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)

        if plot_zoom=='global':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]<=CI_level],lon_mesh[::4,::6][x_ci[::4,::6]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]>CI_level],lon_mesh[::4,::6][x_ci[::4,::6]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            p.axes.set_global();
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True
        elif plot_zoom=='conus':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(15,60);ax.set_xlim((-150+180),(-65+180));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='west_coast':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(30,55);ax.set_xlim((-140+180),(-100+180));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='pacific_ocean':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]<=CI_level],lon_mesh[::2,::3][x_ci[::2,::3]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]>CI_level],lon_mesh[::2,::3][x_ci[::2,::3]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(0,80);ax.set_xlim(-80,80);
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False                    
        ax.coastlines()
        # fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.1,hspace=.01);
        #cbar_ax = fig.add_axes([0.115, 0.15, 0.8, 0.05]) #rect kwargs [left, bottom, width, height];
        fig.colorbar(p,orientation='horizontal', label=x.units, extend='both',pad=0.1);
        if mean_val!='none':
            plt.title(str(round(mean_val[0],2))+ ' '+str(x.units), fontsize=10, loc = 'right');
    # Central longitude is default 0.
    else:
        lat = x.lat
        lon = x.lon
        data = x
        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)

        fig = plt.figure(figsize=(8,6));
        ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)

        if plot_zoom=='global':
            if CI_display == 'mask':
                data = xr.where(x_ci<=CI_level,data,np.nan)
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]<=CI_level],lon_mesh[::4,::6][x_ci[::4,::6]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]>CI_level],lon_mesh[::4,::6][x_ci[::4,::6]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            p.axes.set_global();
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True
        elif plot_zoom=='conus':
            if CI_display == 'mask':
                data = xr.where(x_ci<=CI_level,data,np.nan)
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())                 
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(15,60);ax.set_xlim((-150),(-65));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='west_coast':
            if CI_display == 'mask':
                data = xr.where(x_ci<=CI_level,data,np.nan)
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())                     
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(30,55);ax.set_xlim((-140),(-100));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='pacific_ocean':
            lat = x.lat
            lon = x.lon
            data = x
            data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
            lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]<=CI_level],lon_mesh[::2,::3][x_ci[::2,::3]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]>CI_level],lon_mesh[::2,::3][x_ci[::2,::3]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(0,80);ax.set_xlim(-80,80);
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False                    
        ax.coastlines()
        #fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.1,hspace=.01);
        #cbar_ax = fig.add_axes([0.115, 0.15, 0.8, 0.05]) #rect kwargs [left, bottom, width, height];
        fig.colorbar(p,orientation='horizontal', label=x.units, extend='both',pad=0.1);
        if mean_val!='none':
            plt.title(str(round(mean_val[0],2))+ ' '+str(x.units), fontsize=10, loc = 'right');
    return fig



# 8) count_cumsum: calculate cumulative sum of the number occurences that a condition is met
def count_cumsum(in_xr, dbin, condition):
    """
    Function to calculate cumulative sum of the number occurences that a condition is met.
    Returns bins and cumulative sums. (bins, cumsum)
    :param in_xr: xarray over which you want to calculate the cumulative sum
    :param dbin: float specifying bin width
    :param condition: string specifying condition (<, <=, >, >=, ==)
    """

    bins = np.arange(np.nanmin(in_xr), np.nanmax(in_xr), dbin)
    cumsum = np.zeros(len(bins))

    for i in range(len(bins)):
        if condition == '<':
            cumsum[i] = np.count_nonzero(in_xr < bins[i])
        elif condition == '<=':
            cumsum[i] = np.count_nonzero(in_xr <= bins[i])
        elif condition == '>':
            cumsum[i] = np.count_nonzero(in_xr > bins[i])
        elif condition == '>=':
            cumsum[i] = np.count_nonzero(in_xr >= bins[i])
        elif condition == '==':
            cumsum[i] = np.count_nonzero(in_xr == bins[i])
    
    return bins, cumsum



# 9) plot_panel_maps: plot maps of global/regional monthly means/anomalies by subplot panel
def plot_panel_maps(in_xr,cmin, cmax, ccmap, plot_zoom,central_lon=0,projection = 'PlateCarree',CI_in='none',CI_level='none',CI_display='none', nrow=1, ncol=1, subplot_num=0, mean_val='none',cbar=True):
    """
    Function to plot maps of global/regional ensemble mean and anomalies
    :param in_xr: xarray w/ dims [lat,lon] representing annual mean/anomaly
    :param cmin: float minimum value for ensemble mean and seasonal climatology plots
    :param cmax: float maximum value for ensemble mean and seasonal climatology plots
    :param ccmap: string colormap pallette for ensemble mean and anomaly plots
    :param plot_zoom: string to specify zoom of plot ('global','conus', 'west_coast', 'pacific_ocean')
    :param central_lon: float specifying central longitude for plotting (default=0) If central_lon=180, need to add cyclical point to remove white line
    :param projection: string specifying plot projection. Regional facets only work for PlateCarree. Default is PlateCarree.
:param CI_in: xarray w/ same dims as in_xr specifying 1's where the grid cells are significant to the CI and 0's elsewhere. Default is None.
    :param CI_level: float specifying signficance level for plotting.
    :param CI_display: string specifying how to show CI (default='none'). Options include stippling significant pixels, inverted stippling where insignficant pixels are stippled, or masking out insignificant pixels.
    :param nrow: int specifying number of rows for subplot.
    :param ncol: int specifying number of cols for subplot.
    :param subplot_num: int specifying which subplot panel you are plotting.
    :param mean_val: default is none. If not, specify array(mean, std) to put mean value in top right corner.
    """

    x = in_xr
    x_ci = CI_in
    if projection=='PlateCarree':
        plot_proj = ccrs.PlateCarree(central_longitude=central_lon)
    elif projection=='Robinson':
        plot_proj = ccrs.Robinson(central_longitude=central_lon)
    elif projection=='Mollweide':
        plot_proj = ccrs.Mollweide(central_longitude=central_lon)
    elif projection=='LambertConformal':
        plot_proj = ccrs.LambertConformal(central_longitude=central_lon)
    ax = plt.subplot(nrow,ncol,int(1+subplot_num), projection=plot_proj,transform=plot_proj)

    # Remove white line if plotting over Pacific Ocean.
    if central_lon==180:
        lat = x.lat
        lon = x.lon
        data = x
        data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
        lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)

        # fig = plt.figure(figsize=(8,7));
        # ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)

        if plot_zoom=='global':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]<=CI_level],lon_mesh[::4,::6][x_ci[::4,::6]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]>CI_level],lon_mesh[::4,::6][x_ci[::4,::6]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            p.axes.set_global();
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True
        elif plot_zoom=='conus':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(15,60);ax.set_xlim((-150+180),(-65+180));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='west_coast':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(30,55);ax.set_xlim((-140+180),(-100+180));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='pacific_ocean':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]<=CI_level],lon_mesh[::2,::3][x_ci[::2,::3]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]>CI_level],lon_mesh[::2,::3][x_ci[::2,::3]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            #ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(0,80);ax.set_xlim(-80,80);
            if projection=='LambertConformal':
                ax.set_extent([-270, -90, 15, 80], ccrs.PlateCarree());
            else:
                ax.set_extent([-270, -90, 0, 80], ccrs.PlateCarree());
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--');
            gl.xlines=False;gl.ylines=False;
            gl.ylocator = ticker.FixedLocator([0, 30, 60])
            gl.xlocator = ticker.FixedLocator([-120, 180, 120])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style={'size':7};gl.ylabel_style={'size':7}
            # ax.yaxis.set_major_formatter(LatitudeFormatter());ax.xaxis.set_major_formatter(LongitudeFormatter());
            gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True                    
        ax.coastlines()
        # fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.1,hspace=.01);
        #cbar_ax = fig.add_axes([0.115, 0.15, 0.8, 0.05]) #rect kwargs [left, bottom, width, height];
        if cbar==True:
            cbar=plt.colorbar(p,orientation='horizontal', extend='both',pad=0.1);
            cbar.ax.tick_params(labelsize=7); cbar.set_label(label=x.units, size=7);
        if mean_val!='none':
            plt.title(str(round(mean_val[0],2))+ ' '+str(x.units), fontsize=7, loc = 'right');
    # Central longitude is default 0.
    else:
        lat = x.lat
        lon = x.lon
        data = x
        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)

        # fig = plt.figure(figsize=(8,6));
        # ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)

        if plot_zoom=='global':
            if CI_display == 'mask':
                data = xr.where(x_ci<=CI_level,data,np.nan)
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]<=CI_level],lon_mesh[::4,::6][x_ci[::4,::6]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]>CI_level],lon_mesh[::4,::6][x_ci[::4,::6]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            p.axes.set_global();
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True
        elif plot_zoom=='conus':
            if CI_display == 'mask':
                data = xr.where(x_ci<=CI_level,data,np.nan)
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())                 
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(15,60);ax.set_xlim((-150),(-65));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='west_coast':
            if CI_display == 'mask':
                data = xr.where(x_ci<=CI_level,data,np.nan)
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())                     
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(30,55);ax.set_xlim((-140),(-100));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='pacific_ocean':
            lat = x.lat
            lon = x.lon
            data = x
            data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
            lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]<=CI_level],lon_mesh[::2,::3][x_ci[::2,::3]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]>CI_level],lon_mesh[::2,::3][x_ci[::2,::3]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(0,80);ax.set_xlim(-80,80);
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False                    
        ax.coastlines()
        #fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.1,hspace=.01);
        #cbar_ax = fig.add_axes([0.115, 0.15, 0.8, 0.05]) #rect kwargs [left, bottom, width, height];
        if cbar==True:
            cbar=plt.colorbar(p,orientation='horizontal', extend='both',pad=0.1);
            cbar.ax.tick_params(labelsize=7); cbar.set_label(label=x.units, size=7);
        if mean_val!='none':
            plt.title(str(round(mean_val[0],2))+ ' '+str(x.units), fontsize=7, loc = 'right');
    return ax,p



# 10) calc_weighted_mean_sd: calculate area weighted mean and standard deviation
def calc_weighted_mean_sd(DataArray):
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
    if 'time' in list(DataArray.coords):
        aggregate_weighted_sd = weighted_mean.std(dim='time')
        aggregate_weighted_mean = weighted_mean.mean(dim='time')
    elif 'month' in list(DataArray.coords):
        aggregate_weighted_sd = weighted_mean.std(dim='month')
        aggregate_weighted_mean = weighted_mean.mean(dim='month')        
    return float(aggregate_weighted_mean.values), float(aggregate_weighted_sd.values)



# 11) calc_weighted_mean_tseries: calculate area weighted mean for time series
def calc_weighted_mean_tseries(DataArray):
    '''
    Calculate area-weighted aggregate mean of a variable in an input DataArray
    Adapted from https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html
    Returns an array: the area-weighted mean over time of the variable
        over whatever spatial domain is specified in the input DataArray
    '''
    # create array of weights for each grid cell based on latitude
    weights = np.cos(np.deg2rad(DataArray.lat))
    weights.name = "weights"
    array_weighted = DataArray.weighted(weights)
    weighted_mean = array_weighted.mean(("lon", "lat"))
    return weighted_mean



# 12) plot_west_coast_panel_maps: plot maps of west coast monthly means/anomalies by subplot panel
def plot_west_coast_panel_maps(in_xr,cmin, cmax, ccmap,central_lon=180,projection = 'PlateCarree',colorbar=True,CI_in='none',CI_level='none',CI_display='none', nrow=1, ncol=1, subplot_num=0, mean_val='none'):
    """
    Function to plot maps of global/regional ensemble mean and anomalies
    :param in_xr: xarray w/ dims [lat,lon] representing annual mean/anomaly
    :param cmin: float minimum value for ensemble mean and seasonal climatology plots
    :param cmax: float maximum value for ensemble mean and seasonal climatology plots
    :param ccmap: string colormap pallette for ensemble mean and anomaly plots
    :param central_lon: float specifying central longitude for plotting (default=0) If central_lon=180, need to add cyclical point to remove white line
    :param projection: string specifying plot projection. Regional facets only work for PlateCarree. Default is PlateCarree.
    :param colorbar: Boolean (default=True) to specify including colorbar.
    :param CI_in: xarray w/ same dims as in_xr specifying 1's where the grid cells are significant to the CI and 0's elsewhere. Default is None.
    :param CI_level: float specifying signficance level for plotting.
    :param CI_display: string specifying how to show CI (default='none'). Options include stippling significant pixels, inverted stippling where insignficant pixels are stippled, or masking out insignificant pixels.
    :param nrow: int specifying number of rows for subplot.
    :param ncol: int specifying number of cols for subplot.
    :param subplot_num: int specifying which subplot panel you are plotting.
    :param mean_val: default is none. If not, specify array(mean, std) to put mean value in top right corner.
    """

    x = in_xr
    x_ci = CI_in
    if projection=='PlateCarree':
        plot_proj = ccrs.PlateCarree(central_longitude=central_lon)
    elif projection=='Robinson':
        plot_proj = ccrs.Robinson(central_longitude=central_lon)
    elif projection=='Mollweide':
        plot_proj = ccrs.Mollweide(central_longitude=central_lon)
    elif projection=='LambertConformal':
        plot_proj = ccrs.LambertConformal(central_longitude=central_lon)
    # ax = plt.subplot(nrow,ncol,int(1+subplot_num), projection=plot_proj,transform=plot_proj)
    ax = plt.gca();

    shapename = 'admin_1_states_provinces_lakes'
    states_shp = shpreader.natural_earth(resolution='10m',category='cultural', name=shapename)
    
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))

    # Central lon is 180 (default)
    if central_lon==180:
        lat = x.lat
        lon = x.lon
        data = x
        data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
        lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)
        if CI_display == 'mask':
            data = xr.where(x_ci<=CI_level,data,np.nan)
            p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
        elif CI_display == 'stipple':
            p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
        elif CI_display == 'inv_stipple':
            p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
        else:
            p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
        states = ['Washington','Oregon','California']
        for astate in shpreader.Reader(states_shp).records():
            if astate.attributes['name'] in states:
                ### You want to replace the following code with code that sets the
                ### facecolor as a gradient based on the population density above
                #facecolor = [0.9375, 0.9375, 0.859375]
                edgecolor = 'black'
                # `astate.geometry` is the polygon to plot
                ax.add_geometries([astate.geometry], ccrs.PlateCarree(),
                                    facecolor='none', edgecolor=edgecolor)
            else:
                ### You want to replace the following code with code that sets the
                ### facecolor as a gradient based on the population density above
                #facecolor = [0.9375, 0.9375, 0.859375]
                edgecolor = 'black'
                # `astate.geometry` is the polygon to plot
                ax.add_geometries([astate.geometry], ccrs.PlateCarree(),
                                    facecolor='none', edgecolor=None,alpha=0)
        ax.set_extent([-125, -112, 32, 50], ccrs.PlateCarree());
        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)
        if colorbar==True:
            cbar = plt.colorbar(p,orientation='vertical', label=x.units, extend='both',format=fmt,pad=0.1);
        if mean_val!='none':
            plt.title(str(round(mean_val[0],2))+ ' '+str(x.units), fontsize=10, loc = 'right');
    # Central longitude is 0.
    else:
        lat = x.lat
        lon = x.lon
        data = x
        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
        if CI_display == 'mask':
            data = xr.where(x_ci<=CI_level,data,np.nan)
            p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
        elif CI_display == 'stipple':
            p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
        elif CI_display == 'inv_stipple':
            p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
        else:
            p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
        states = ['Washington','Oregon','California']
        for astate in shpreader.Reader(states_shp).records():
            if astate.attributes['name'] in states:
                ### You want to replace the following code with code that sets the
                ### facecolor as a gradient based on the population density above
                #facecolor = [0.9375, 0.9375, 0.859375]
                edgecolor = 'black'
                # `astate.geometry` is the polygon to plot
                ax.add_geometries([astate.geometry], ccrs.PlateCarree(),
                                    facecolor='none', edgecolor=edgecolor)
            else:
                ### You want to replace the following code with code that sets the
                ### facecolor as a gradient based on the population density above
                #facecolor = [0.9375, 0.9375, 0.859375]
                edgecolor = 'black'
                # `astate.geometry` is the polygon to plot
                ax.add_geometries([astate.geometry], ccrs.PlateCarree(),
                                    facecolor='none', edgecolor=None,alpha=0)
        ax.set_extent([-125, -110, 30, 50], ccrs.PlateCarree());
        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)
        if colorbar==True:
            cbar = plt.colorbar(p,orientation='vertical', label=x.units, extend='both',format=fmt,pad=0.1);
        if mean_val!='none':
            plt.title(str(round(mean_val[0],2))+ ' '+str(x.units), fontsize=10, loc = 'right');

    return ax,p



# 13) weighted_temporal_mean: calculate day-weighted mean
def weighted_temporal_mean(ds):
    """
    weight by days in each month
    adapated from https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds * 1.0

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    # Return the weighted average
    return obs_sum / ones_out



# 14) weighted_temporal_mean_clim: calculate day-weighted mean for monthly climatology
def weighted_temporal_mean_clim(ds):
    """
    weight by days in each month for monthly climatology
    adapated from https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html
    """
    # Determine the month length
    month_dict = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    month_length = xr.DataArray(data=np.array([month_dict.get(key) for key in ds.month.values]),dims='month',coords=dict(month=ds.month))

    # Calculate the weights
    wgts = month_length / month_length.sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.sum(), 1.0)

    # Subset our dataset for our variable
    obs = ds * 1.0

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).sum(dim='month')

    # Calculate the denominator
    ones_out = (ones * wgts).sum(dim='month')

    # Return the weighted average
    return obs_sum / ones_out