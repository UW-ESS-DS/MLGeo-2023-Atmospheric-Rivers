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

## By Chad Small. This script locates all of the times an AR makes landfall in WA

#bring the list of ARs
WA_ARs = pd.read_csv('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/text_data/WA_LF_ARs.csv')
WA_ARs = WA_ARs.drop(columns=['Unnamed: 0'])
clean_df=WA_ARs.loc[WA_ARs['Landfall?'] == True]
clean_df=clean_df.drop_duplicates(subset=['AR ID']).reset_index(drop=True)

start_Cali = clean_df

#need the mask too
cali_test =xr.open_dataset('/home/disk/orca/csmall3/AR_testing_research/state_masks/WA_state_mask.nc')
cali_mask = cali_test['Mask']

#test finding how many times it makes landfall in cali

AR_ID = []
AR_Landfall = []
AR_MJO_DT = []

for ni, i in enumerate(start_Cali['AR ID']):
    url = i
    ar_test = xr.open_dataset(url)

    print('{} of {} ARs.'.format(ni, len(start_Cali['AR ID'])))

    # found_a_match = False

    for j in range(0, len(ar_test['mask_at_end_time']['time'])):
        # try:
        mask_ary=ar_test['mask_at_end_time'][j]
        
        #pull the datetime
        times = np.array(mask_ary['time'])
        times=np.array2string(times)
        str_times = times.replace("'", "")
        fin_time = datetime.strptime(str_times.split(".")[0], '%Y-%m-%dT%H:%M:%S')


        #lets' try adding the data frames back to back

        test_add = cali_mask + mask_ary
        htest=np.array((test_add == 2).any())
        bool_match=np.array(htest, dtype=bool)

        # AR_ID += [str(i)]
        # AR_Landfall += [np.array((test_add == 2).any()).astype(str)]
        # AR_MJO_DT += [fin_time.strftime('%Y-%m-%dT%H:%M:%S')]

        #let's try to do the plot inside a if else statements
        if bool_match == True:
            AR_ID += [str(i)]
            AR_Landfall += [np.array((test_add == 2).any()).astype(str)]
            AR_MJO_DT += [fin_time.strftime('%Y-%m-%dT%H:%M:%S')]


            found_a_match = True
            # break
        else: 
            print("Doesn't make landfall")
    
    # if found_a_match:
    #     continue

        # except:
        #     pass

print('Create Dataframe for Output.')
AR_MJO_overlp_df = pd.DataFrame({'AR ID':AR_ID, 
                               'Date':AR_MJO_DT,
                               'Landfall?':AR_Landfall})




#save the filtered DF
AR_MJO_overlp_df.to_csv('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/text_data/WA_LF_Rain.csv') 