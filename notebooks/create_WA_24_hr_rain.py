from matplotlib.gridspec import GridSpec
from netCDF4 import Dataset
import matplotlib
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
import numpy as np
from datetime import datetime, timedelta
import datetime as dt
import xarray as xr
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.colors as mcols
import glob 
import colorcet as cc
import netCDF4
import cmaps
from scipy.interpolate import interp2d
import cartopy
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import matplotlib.gridspec as gridspec
import seaborn as sns
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pyproj import Proj
# from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim
from colorspacious import cspace_converter
import pathlib
from pathlib import Path
import numpy.ma as ma
from numpy import genfromtxt
import pandas as pd
import calendar
from IPython.core.pylabtools import figsize
from scipy import stats
import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

## By Chad Small. This script saves 24 rainfall accumulations in and around California on datetimes when an AR was over California

#bring in data
filtered_df = pd.read_csv('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/text_data/WA_LF_Rain.csv')
filtered_df = filtered_df.drop(columns=['Unnamed: 0'])
filtered_df['Datetime'] = pd.to_datetime(filtered_df['Date'])

#create the rainfall data
for i in range(0,len(filtered_df)):
# for i in range(0,1):
    # dt1 = filtered_df['Datetime'].iloc[i] + timedelta(hours=(6*i)) #this is when you're starting a date and want to see what happens going forward
    dt1 = filtered_df['Datetime'].iloc[i]
    dt_list = [dt1 - dt.timedelta(hours=x) for x in range(1,12,1)][::-1]
    dt_list2 = [dt1 + dt.timedelta(hours=x) for x in range(0,12,1)]#need the center point
    dt_list_tot = dt_list + dt_list2
    rain_ds = []
    fn_hold = []
    moi = str(dt1.month).rjust(2, '0')
    doi = str(dt1.day).rjust(2, '0')
    hoi = str(dt1.hour).rjust(2, '0')
    yoi = str(dt1.year)

    for tttt, this_dt2 in enumerate(dt_list_tot):
        fmt = '/home/orca/data/satellite/precip/gpmdata/imerg/%Y/%m/%d/3B-HHR.MS.MRG.3IMERG.%Y%m%d-S%H0000-*.V06B.HDF5'
        if len(glob.glob(this_dt2.strftime(fmt))) < 1:       
            fmt = '/home/orca/data/satellite/precip/gpmdata/imerg/late/%Y/%m/%d/3B-HHR-L.MS.MRG.3IMERG.%Y%m%d-S%H0000-*.HDF5'
        fn = glob.glob(this_dt2.strftime(fmt))[0]
        fn_hold += [fn]

    for j in fn_hold:
        ncf = netCDF4.Dataset(j, diskless=True, persist=False)
        nch = ncf.groups.get('Grid')
        xds = xr.open_dataset(xr.backends.NetCDF4DataStore(nch))
    #let's bake in a geo slice
        # max_lat =35.366667
        # min_lat=32.535
        # max_lon=-115.708847
        # min_lon=-124

        max_lat =50
        min_lat=45
        max_lon=-116.2 #243.75
        min_lon=-125.5 #234.5

        # max_lon=246.25 #-113.75
        # min_lon=234.5 #-125.5
        LatIndexer, LonIndexer = 'lat', 'lon'
        rain_slice = xds.sel(**{LatIndexer: slice(min_lat, max_lat),
                        LonIndexer: slice(min_lon, max_lon)})
        rain_ds += [rain_slice]

    rain_ds=xr.concat(rain_ds, dim="time")

    data_hold = rain_ds['precipitationCal']
    data_hold=data_hold.where(data_hold.lon + data_hold.lat < 1e+6)
    two_day_rain=data_hold.sum(dim="time",skipna=False)

    #set up the filename
    NNNN=18
    t_len = len(str(filtered_df['AR ID'].iloc[i]))
    hrrr=str(filtered_df['AR ID'].iloc[i])[t_len - NNNN:]
    TT =15
    # str(hrrr)[:TT]
    title = str(hrrr)[:TT]


    #save as nc

    #rain2d = np.array(two_day_rain)
    lon_rain = two_day_rain['lon']
    lat_rain = two_day_rain['lat']
    # interp_fcn = interp2d(mask_lon0, mask_lat0, mask0)
    # mask = interp_fcn(lon_rain, lat_rain)
    # rain2d = np.array(two_day_rain.T)

    # rain2d[mask < 0.1] = np.nan

    # rain2d = rain2d.T
    rain2d =  two_day_rain

    # lon_ary = np.array(two_day_rain['lon'])
    # lat_ary = np.array(two_day_rain['lat'])
    # ny, nx = (rain2d.shape[1], rain2d.shape[0])

    lon_ary = np.array(two_day_rain['lon'])
    lat_ary = np.array(two_day_rain['lat'])
    ny, nx = (rain2d.shape[1], rain2d.shape[0])
    ncout = Dataset('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/Rain_nc/Acc_24hr/'+title+'_'+yoi+moi+doi+hoi+'.nc','w','NETCDF4'); # using netCDF3 for output format 
    ncout.createDimension('lon',nx);
    ncout.createDimension('lat',ny);
    ncout.createDimension('time', 1);
    #ncout.createDimension('time',nyears);
    lonvar = ncout.createVariable('lon','float64',('lon'));lonvar[:] = lon_ary;
    latvar = ncout.createVariable('lat','float64',('lat'));latvar[:] = lat_ary;
    ncout.createVariable('time','d',('time',))
    ncout['time'][:] = (dt1 - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0
    ncout['time'].units = 'hours since 1970-1-1 0:0'
    #timevar = ncout.createVariable('time','float64',('time'));timevar.setncattr('units',unout);timevar[:]=date2num(datesout,unout);
    # myvar = ncout.createVariable('One Day Rain','float64',('time','lon','lat'));myvar.setncattr('units','mm');myvar[:] = rain2d;
    myvar = ncout.createVariable('One Day Rain','float64',('time','lon','lat'));myvar.setncattr('units','mm');myvar[:] = rain2d;