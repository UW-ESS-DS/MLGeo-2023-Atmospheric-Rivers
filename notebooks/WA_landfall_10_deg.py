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

## this script by Chad Small should do the following:
## 1. it pulls in all of the 10 Degree AR data from 2014/2015 through 2017/2018
## 2. It filters by duration to the 90th percentile duration 
## 3. It checks if the ARs make landfall in WA

#1. Bring in the AR data
fn_list_1 = sorted(glob.glob('/home/disk/orca/csmall3/AR_testing_research/LPT_ARs/hourly_res/10N_10S_AR_outputs/data/AR/g0_0h/thresh1/systems/2014050100_2015043023/lpt_system_mask_AR.lptid*.nc'))
fn_list_2 = sorted(glob.glob('/home/disk/orca/csmall3/AR_testing_research/LPT_ARs/hourly_res/10N_10S_AR_outputs/data/AR/g0_0h/thresh1/systems/2015050100_2016043023/lpt_system_mask_AR.lptid*.nc'))
fn_list_3 = sorted(glob.glob('/home/disk/orca/csmall3/AR_testing_research/LPT_ARs/hourly_res/10N_10S_AR_outputs/data/AR/g0_0h/thresh1/systems/2016050100_2017043023/lpt_system_mask_AR.lptid*.nc'))
fn_list_4 = sorted(glob.glob('/home/disk/orca/csmall3/AR_testing_research/LPT_ARs/hourly_res/10N_10S_AR_outputs/data/AR/g0_0h/thresh1/systems/2017050100_2018043023/lpt_system_mask_AR.lptid*.nc'))

fn_list = fn_list_1 + fn_list_2 +fn_list_3 +fn_list_4

#2. Make sure it's the right length
new_ls = []

for i in fn_list:
    ar_test = xr.open_dataset(i)
    if ar_test['duration'].values > 576:
        print("AR too long")
    else:
        new_ls += [i]

#3. Figure out if it makes Landfall in WA

#bring in any AR (doesn't matter) to set up the background spatial resolution
start_df = pd.read_csv('/home/disk/orca/csmall3/AR_testing_research/Text_data/hourly_ARs/Deg_5_Match/Pac_LPT_strt_5_Deg_AR_RMM.csv')
start_df = start_df.drop(columns=['Unnamed: 0'])

ar_oi = xr.open_dataset(str(start_df['AR ID'].iloc[0]))

AR_ID = []
AR_Landfall = []
AR_MJO_DT = []

#bring in the mask of WA state
cali_test =xr.open_dataset('/home/disk/orca/csmall3/AR_testing_research/state_masks/WA_state_mask.nc')
cali_mask = cali_test['Mask']

#creating a directory to hold pictures that show where the LF for the ARs was and what that looked like
os.makedirs('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/Figures/Washington_LF_Match/', exist_ok=True)

#start iterating through

#add second loop for the lifetime of the AR
for ni, i in enumerate(new_ls):
    url = i
    ar_test = xr.open_dataset(url)

    print('{} of {} ARs.'.format(ni, len(new_ls)))

    found_a_match = False

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

        AR_ID += [str(i)]
        AR_Landfall += [np.array((test_add == 2).any()).astype(str)]
        AR_MJO_DT += [fin_time.strftime('%Y-%m-%dT%H:%M:%S')]

        #let's try to do the plot inside a if else statements
        if bool_match == True:
            NNNN=18
            t_len = len(str(url))
            hrrr=str(url)[t_len - NNNN:]
            TT =15
            # str(hrrr)[:TT]
            title = str(hrrr)[:TT]




            ## The plots 
            plot_data = cali_mask
            lon_ary = cali_mask['lon']
            lat_ary = cali_mask['lat']

            plot_data2 = mask_ary
            lon_ary2 = mask_ary['lon']
            lat_ary2 = mask_ary['lat']

            colormap=cmaps.MPL_jet
            fig = plt.figure(figsize=[14, 7])
            political_boundaries = NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none')
            states = NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lakes', scale='50m', facecolor='none')

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_title('MJO AR('+title+') Overlap '+str(fin_time), fontsize=14)					   
            ax.add_feature(cartopy.feature.LAND, edgecolor='k', facecolor='none', zorder=10)
            ax.coastlines('50m', linewidth=2, zorder=5)
            # ax.set_extent([50, 180, -50, 50], crs=ccrs.PlateCarree())#set west coast in a second
            css = ax.pcolormesh(lon_ary, lat_ary, plot_data, cmap = 'Greys', transform=ccrs.PlateCarree(),zorder=1, vmin=0, alpha=.5)
            css = ax.pcolormesh(lon_ary2, lat_ary2, plot_data2, cmap = 'Reds', transform=ccrs.PlateCarree(),zorder=1, vmin=0, alpha=.5)

            

            #try adding gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5)
            gl.ypadding = 5
            gl.xpadding = 5
            gl.xlocator = mticker.FixedLocator(np.arange(-180,180,45)[::1])
            gl.ylocator = mticker.FixedLocator(np.arange(-90,90,45)[::1])		
            # fig.show()
            plt.savefig('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/Figures/Washington_LF_Match/'+title+'.png')
            # plt.savefig('./figures/tst_NEW_Deg_0_'+str(year)+'_'+str(nxt_year)+'_all_new_match/'+title+'.png')
            plt.close(fig)

            found_a_match = True
            break
        else: 
            print("Doesn't make landfall")
    
    if found_a_match:
        continue

        # except:
        #     pass



print('Create Dataframe for Output.')
AR_MJO_overlp_df = pd.DataFrame({'AR ID':AR_ID, 
                               'Date':AR_MJO_DT,
                               'Landfall?':AR_Landfall})


#to save the df later
print('Save Dataframe to CSV.')
# AR_MJO_overlp_df.to_csv('./Text_data/tst_NEW_Deg_0_Test_all_output_'+str(year)+'_'+str(nxt_year)+'.csv') 
AR_MJO_overlp_df.to_csv('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/text_data/WA_LF_ARs.csv') 
