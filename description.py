#%% Imports
import os
import pandas as pd
from random import Random
import numpy as np 

#%% Settings
HOME_DIR = '/Users/bxp151/ml/usedcars'
os.chdir(HOME_DIR)
PROC_DIR = HOME_DIR + '/processed'
DATA_DIR = HOME_DIR + '/data'
IN_FILE = '/latlon.csv.zip'
OUT_FILE = '/vin_description.csv'
OUT_FILE2 = '/vin_description2.csv'
OUT_FILE3 = '/unlabeled_desciptions.csv.zip'
OUT_FILE4 = '/full.csv'
IN_FILE2 = '/seller.csv'
random_seed = 42
    
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
#%% Read file and subset VIN, description
df = pd.read_csv(PROC_DIR + IN_FILE, 
                 usecols = ['vin', 'description', 'image_url'], compression='zip')

# randomly select 1000 row indicies to manually label training set
row_ind = Random(x=random_seed).sample(range(df.shape[0]), 1000)

rs = df.loc[row_ind, :] 
rs.to_csv(DATA_DIR + OUT_FILE, index = False)
#%% Export additional 1000 samples for training model

sample_count = np.bincount(row_ind, minlength=df.shape[0])
unsampled_mask = sample_count == 0 
ms = df.loc[unsampled_mask, :]

ms.loc[:, ('length')] = ms['description'].str.len()

s = ms[ms.length <= 1500]

ms_rowind = Random(x=random_seed).sample(set(s.index), 1000)

ss = s.loc[ms_rowind, :]
ss.to_csv(DATA_DIR + OUT_FILE2, index = False)
#%% Export all descriptions
df[['description']].to_csv(DATA_DIR + OUT_FILE3, index = False, compression='zip')

#%% Import labeled descriptions and append to df, then export

ll = pd.read_csv(PROC_DIR + IN_FILE, compression='zip')
lb = pd.read_csv(DATA_DIR + IN_FILE2)

# drop the description here as it takes up a lot of storage
full = lb.merge(ll, left_index=True, right_index=True).drop('description', axis=1)

full.to_csv(PROC_DIR + OUT_FILE4)
