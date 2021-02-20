#%% Imports
import os
import gdown
import pandas as pd
import requests
from zipfile import ZipFile
import exploretransform as et
import requests
import io
import numpy as np

#%% Settings
HOME_DIR = '/Users/bxp151/ml/usedcars'
os.chdir(HOME_DIR)

IMG_DIR = HOME_DIR + '/images'
PROC_DIR = HOME_DIR + '/processed'

COMB_FILE = '/combined.csv'


if not os.path.exists("images"):
    os.mkdir(IMG_DIR)
    
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
#%% Import and explore
