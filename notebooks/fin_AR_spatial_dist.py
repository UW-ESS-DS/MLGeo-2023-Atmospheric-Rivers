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

## This script by Chad Small breaks up each AR system into its individual objects and adds them up in space such that they can be plotted.
## For computational reasons, it's often easier to break these up into multiple netcdf files. In this case, it's 5 files. See lines 46, 47 and 98/99

instance = int(sys.argv[1])
bef_ints = instance - 1

#first let's open the df for Danny's AR data
fewer_ARs = pd.read_csv('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/text_data/era5_matched_ar_events.csv')
fewer_ARs = fewer_ARs.drop(columns=['Unnamed: 0'])

ar_fls = []
for i in range(0, len(fewer_ARs)):
    url=fewer_ARs['event_filename'].iloc[i]
    NNNN=23
    t_len = len(str(url))
    hrrr=str(url)[t_len - NNNN:]
    TT =4
    # str(hrrr)[:TT]
    yoi = str(hrrr)[:TT]

    url=fewer_ARs['event_filename'].iloc[i]
    NNNN=18
    t_len = len(str(url))
    hrrr=str(url)[t_len - NNNN:]
    TT =15
    # str(hrrr)[:TT]
    fl_name = str(hrrr)[:TT]
    

    ar_fls += ['/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/Project_AR_nc/'+str(yoi)+'_'+str(fl_name)+'.nc']


hold = ar_fls

idx_8_1_deg_0 = hold


#turn this off for now because I have the number I need and don't want to duplicate time
#turn back on for full data though
# num_masks = [] #this just gives me the number of masks for now
# for i in idx_8_1_deg_0:
#     ar_oi = xr.open_dataset(i)
#     for j in range(0,len(ar_oi['time'])):
#         ar_mask=ar_oi['mask_at_end_time'][j]
#         num_masks += [ar_mask]

# #save the length
# len_oi=len(num_masks)
# ar_msk_num = pd.DataFrame({'number of masks':len_oi}, index=[0])
# ar_msk_num.to_csv('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/text_data/All_WA_AR_mask_len.csv') 

# len_oi = 87255 #this is only for now because I have that number good to go

len_oi = len(idx_8_1_deg_0)
# I need to break up the AR list into groups of 20 actually 
strt_range = int((len_oi/5)*bef_ints)
end_range = int((len_oi/5)*instance)

ar_masks = []
for i in range(strt_range,end_range):
    ar_oi = xr.open_dataset(idx_8_1_deg_0[i])
    for j in range(0,len(ar_oi['time'])):
        ar_mask=ar_oi['mask_at_end_time'][j]
        ar_masks += [ar_mask]

if len(ar_masks) == 0:
    print('Nothing to concatenate')
    ar_10_sums = []
else:
    ar_bulk=xr.concat(ar_masks, dim="time")
    ar_10_sums=ar_bulk.sum(dim='time', skipna=True)

#save as nc
#save the ar bulk just in case
if len(ar_10_sums) == 0:
    print('nothing to save')
else:
    nc_save=np.array(ar_10_sums)
    lon_ary = np.array(ar_10_sums['lon'])
    lat_ary = np.array(ar_10_sums['lat'])
    # time_ary = np.array(ar_bulk['time'])
    ny, nx = (ar_10_sums.shape[0], ar_10_sums.shape[1])
    ncout = Dataset('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/Space_dist_nc/Masks_WA_pt_'+str(instance)+'.nc','w','NETCDF4'); # using netCDF3 for output format 
    ncout.createDimension('lon',nx);
    ncout.createDimension('lat',ny);
    # ncout.createDimension('time',nyears);
    lonvar = ncout.createVariable('lon','float64',('lon'));lonvar[:] = lon_ary;
    latvar = ncout.createVariable('lat','float64',('lat'));latvar[:] = lat_ary;
    #timevar = ncout.createVariable('time','float64',('time'));timevar.setncattr('units',unout);timevar[:]=date2num(datesout,unout);
    myvar = ncout.createVariable('Total_Mask','float64',('lat','lon'));myvar.setncattr('units','masks');myvar[:] = nc_save;
    ncout.close();
