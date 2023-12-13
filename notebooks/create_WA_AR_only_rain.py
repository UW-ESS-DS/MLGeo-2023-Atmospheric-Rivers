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

## This script, by Chad Small, takes the IMERG 24 hour rainfall accumulation and just locates it under the AR envelope over the state of WA
## for ARs that intersect with MJO at the initiation



#bring in the WA state ERA5 mask
IMERG_mask = xr.open_dataset('/home/disk/orca/csmall3/AR_testing_research/state_masks/WA_state_mask.nc')
IMERG_mask['lat'] = IMERG_mask['lat'] #need this to fix the lat flipping problem!

#try with WA start times
start_WA = pd.read_csv('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/text_data/WA_LF_Rain.csv')
start_WA = start_WA.drop(columns=['Unnamed: 0'])

for i in range(0,len(start_WA)):
    ar_oi = xr.open_dataset(start_WA['AR ID'].iloc[i])

    start_WA['Datetime'] = pd.to_datetime(start_WA['Date'])

    #test with time indexer
    fin_time=start_WA['Datetime'].iloc[i]
    TimeIndexer = 'time'
    ar_inst = ar_oi.sel(**{TimeIndexer: slice(fin_time, fin_time)})

    ar_inst = ar_inst['mask_at_end_time']

    #read in the rain data
    moi = str(fin_time.month).rjust(2, '0')
    doi = str(fin_time.day).rjust(2, '0')
    hoi = str(fin_time.hour).rjust(2, '0')
    yoi = str(fin_time.year)


    NNNN=18
    t_len = len(str(start_WA['AR ID'].iloc[i]))
    hrrr=str(start_WA['AR ID'].iloc[i])[t_len - NNNN:]
    TT =15
    title = str(hrrr)[:TT]

    #pull in the right rain array
    #try to mask out the rain at that date
    ar_rain_ds = xr.open_dataset('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/Rain_nc/Acc_24hr/'+title+'_'+yoi+moi+doi+hoi+'.nc')
    ar_rain_ary=ar_rain_ds['One Day Rain']
    ar_rain_ary = ar_rain_ary[0]

    #consolidate the code
    #try interpolating to coarsen
    minx = ar_rain_ary.lon.min().item()
    maxx = ar_rain_ary.lon.max().item()
    miny = ar_rain_ary.lat.min().item()
    maxy = ar_rain_ary.lat.max().item()

    # set up new lat/lon grid
    new_grid_x = np.arange(
        np.ceil(minx / 0.25) * 0.25,
        (np.floor(maxx / 0.25) + 0.5) * 0.25,
        0.25
    )
    new_grid_y = np.arange(
        np.ceil(miny / 0.25) * 0.25,
        (np.floor(maxy / 0.25) + 0.5) * 0.25,
        0.25
    )
    # interpolate using nearest neighbor (can use linear, etc. if desired)
    coarse = ar_rain_ary.interp(lon=new_grid_x, lat=new_grid_y, method="nearest")

    #slice the AR ds
    #maybe slicing the AR might help
    max_lat =coarse['lat'].max().values
    min_lat=coarse['lat'].min().values
    max_lon=coarse['lon'].max().values + 360 #243.75
    min_lon=coarse['lon'].min().values + 360 #234.5

    # max_lon=246.25 #-113.75
    # min_lon=234.5 #-125.5
    LatIndexer, LonIndexer = 'lat', 'lon'
    ar_slice = ar_inst.sel(**{LatIndexer: slice(max_lat, min_lat),
                    LonIndexer: slice(min_lon, max_lon)})

    #change the lon of the ar slice so it works out
    ar_slice.coords['lon'] = (ar_slice.coords['lon'] + 180) % 360 - 180

    #maybe the regular resample will work now
    rain_loc,ar_loc=xr.align(coarse, ar_slice)

    #finally overlap
    test_plt=rain_loc.where(ar_loc,0)

    test_plt = test_plt[:,:,0]

    #set up masking just over AR region

    #change test_plt lons back
    test_plt['lon'] = test_plt['lon'] +360

    #try to filter it out under the WA state mask

    mask_ary = IMERG_mask['Mask']
    a,b=xr.align(test_plt, mask_ary) #this works to fix the alignment problem!

    new_plt=a.where(b,0)#this works to mask perfectlY!

    lon_rain = new_plt['lon']
    lat_rain = new_plt['lat']
    # interp_fcn = interp2d(mask_lon0, mask_lat0, mask0)
    # mask = interp_fcn(lon_rain, lat_rain)
    # rain2d = np.array(two_day_rain.T)

    # rain2d[mask < 0.1] = np.nan

    # rain2d = rain2d.T
    rain2d =  new_plt.values

    # lon_ary = np.array(two_day_rain['lon'])
    # lat_ary = np.array(two_day_rain['lat'])
    # ny, nx = (rain2d.shape[1], rain2d.shape[0])

    lon_ary = np.array(new_plt['lon'])
    lat_ary = np.array(new_plt['lat'])
    ny, nx = (rain2d.shape[1], rain2d.shape[0])
    ncout = Dataset('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/Rain_nc/Acc_24hr_AR_only/'+title+'_'+yoi+moi+doi+hoi+'.nc','w','NETCDF4'); # using netCDF3 for output format 
    ncout.createDimension('lon',nx);
    ncout.createDimension('lat',ny);
    ncout.createDimension('time', 1);
    #ncout.createDimension('time',nyears);
    lonvar = ncout.createVariable('lon','float64',('lon'));lonvar[:] = lon_ary;
    latvar = ncout.createVariable('lat','float64',('lat'));latvar[:] = lat_ary;
    ncout.createVariable('time','d',('time',))
    ncout['time'][:] = (fin_time - dt.datetime(1970,1,1,0,0,0)).total_seconds()/3600.0
    ncout['time'].units = 'hours since 1970-1-1 0:0'
    #timevar = ncout.createVariable('time','float64',('time'));timevar.setncattr('units',unout);timevar[:]=date2num(datesout,unout);
    # myvar = ncout.createVariable('One Day Rain','float64',('time','lon','lat'));myvar.setncattr('units','mm');myvar[:] = rain2d;
    myvar = ncout.createVariable('One Day Rain','float64',('time','lon','lat'));myvar.setncattr('units','mm');myvar[:] = rain2d;