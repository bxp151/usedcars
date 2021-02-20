#%% Imports
import os
import gdown
import pandas as pd
from zipfile import ZipFile
import requests
import numpy as np
import exploretransform as et

#%% Settings
HOME_DIR = '/Users/bxp151/ml/usedcars'
os.chdir(HOME_DIR)

DATA_DIR = HOME_DIR + '/data'
IMG_DIR = HOME_DIR + '/images'
PROC_DIR = HOME_DIR + '/processed'

DATA_URL = 'https://drive.google.com/uc?export=download&confirm=7B-T&id=1m_Eslv90pmkhh3U7X0JeJ_-QeVJYedG4'
DATA_FILE_ZIP = '/vehicles.csv.zip'
DATA_FILE = '/vehicles.csv'
VIN_FILE = '/vins.csv'
OUT_FILE = '/nhtsa.csv.zip'

    
if not os.path.exists("data"):
    os.mkdir(DATA_DIR)
    
if not os.path.exists("processed"):
    os.mkdir(PROC_DIR)
    

    
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)

#%% Download, unzip, and read source file

def acquire():
    
    gdown.download(DATA_URL, DATA_DIR + DATA_FILE_ZIP, quiet=False) 
    
    with ZipFile(DATA_DIR + DATA_FILE_ZIP) as zip:
        zip.extractall(path=DATA_DIR)
        
    return pd.read_csv(DATA_DIR + DATA_FILE)


df = acquire()   

#%% Explore dataset 

# Only take cars with VINs
df = df.loc[df['VIN'].notnull()]
'''
et.peek(df)
et.explore(df)
'''

#%% Get additional info for VINs from NHTSA API

def vingen(df):
    '''
    Parameters
    ----------
    df : the original dataframe containing VINs

    Returns
    -------
    vinlist : list of lists containing batches of 50 VINs

    '''
    
    allvins = list(df['VIN'].unique())
    vinlist = list()
    
    # sets the number of iterations based on the number of vins
    iters = int(np.ceil(len(allvins) / 50))
    
    for i in range(iters):
        '''
        For loop creates a list of batches of 50 vins to pass to the NHTSA API
        '''
    
        if i == 0:
            start_idx = 0
            end_idx = 50
        
        # puts the vins in the correct format for nhtsa
        vinlist.append(";".join(allvins[start_idx:end_idx]))
        
        
        start_idx = start_idx + 50
        if (end_idx + 50) < len(allvins):
            end_idx = end_idx + 50
        else:
            end_idx = len(allvins) 

    return vinlist
    
vinlist = vingen(df)

for num, batch in enumerate(vinlist):
    # removes the vins.csv file if it exists
    if num == 0:
        if os.path.exists(DATA_DIR + VIN_FILE):
            os.remove(DATA_DIR + VIN_FILE)    
    url = 'https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/'
    post_fields = {'format': 'csv', 'data': batch}
    r = requests.post(url, data=post_fields)
    with open(DATA_DIR + VIN_FILE, 'a+') as writer:
        writer.write(r.text)
    print('batch ' + str(num + 1) + " complete " + 
          str(len(vinlist)-(num+1)) + " batches remaining")



#%% Read in specific columns from NHTSA csv file

# skipping header rows that were written into the file 
skiprows = np.arange(51, len(pd.read_csv(DATA_DIR + VIN_FILE)), 51)

cols = ['vin',  'modelyear', 'make', 'model', 'displacementcc', 
        'fueltypeprimary', 'vehicletype', 'bodyclass']

ht = pd.read_csv(DATA_DIR + VIN_FILE, skiprows=skiprows, usecols=cols)


#%% Explore NHTSA and CL data
# et.peek(ht)
# et.explore(ht)

'''
NHTSA Dataset:
    1. drop missing columns
    2. subset PASSENGER CAR
'''
ht = ht.dropna()
ht = ht.loc[ht['vehicletype'] == 'PASSENGER CAR']


'''
CL dataset:
    1. keep: important columns
    2. Odometer > 0, price > 0 
    3. Sort the dataset by VIN / post date
    4. Take the most recent post for each VIN
'''
colscl = ['VIN', 'posting_date','state','region', 'region_url', 'price','condition',
          'cylinders', 'size', 'type', 'paint_color', 'description',
          'odometer','title_status', 'image_url']

cl = df[colscl]
cl = cl.loc[ (cl['price'] > 0) & \
             (cl['odometer'] > 0) ]
cl.columns = map(str.lower, cl.columns)
cl = cl.sort_values(by=['vin','posting_date'])
cl = cl.groupby('vin').tail(1)


'''
    1. Join sets on 'vin'
    2. subset only clean cars with price >= 1000
    2. write final set to csv

'''

co = ht.merge(cl, on='vin')

co = co.loc[ (co['title_status'] == 'clean') & \
             (co['price'] >= 1000) ]

co.to_csv(PROC_DIR + OUT_FILE, index=False, compression='zip')


