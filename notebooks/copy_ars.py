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
import shutil

#bring in ARs
all_df = pd.read_csv('/home/disk/orca/csmall3/HW_stuff/MLGeo_2023/Final_Project/text_data/WA_LF_ARs.csv')
all_df = all_df.drop(columns=['Unnamed: 0'])
clean_df=all_df.loc[all_df['Landfall?'] == True]
clean_df=clean_df.drop_duplicates(subset=['AR ID']).reset_index(drop=True)

#need datetime for this
clean_df['Datetime'] = pd.to_datetime(clean_df['Date'])

for i in range(0, len(clean_df)):
# for i in range(0, 2):
    url=clean_df['AR ID'].iloc[i]
    NNNN=18
    t_len = len(str(url))
    hrrr=str(url)[t_len - NNNN:]
    TT =15
    # str(hrrr)[:TT]
    title = str(hrrr)[:TT]

    shutil.copyfile(clean_df['AR ID'].iloc[i], '/home/disk/orca/csmall3/public_html/outgoing/ML_GEO_ARs/'+str(clean_df['Datetime'].iloc[i].year)+'_'+title+'.nc')