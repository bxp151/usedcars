#%% Imports
import os
import pandas as pd
import requests
import time

#%% Settings
HOME_DIR = '/Users/bxp151/ml/usedcars'
os.chdir(HOME_DIR)
PROC_DIR = HOME_DIR + '/processed'
IN_FILE = '/nhtsa.csv.zip'
LL_FILE = '/llfile.csv'
OUT_FILE = '/latlon.csv.zip'

API_KEY = input ("Enter your google API key: ")
BASE_URL = 'https://maps.googleapis.com/maps/api/geocode/json?address='
    
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)

#%% Create the base for a city, state lookup table

df = pd.read_csv(PROC_DIR + IN_FILE, compression='zip')

# Take the last value from each group
ll = df[['region_url', 'state']].sort_values('region_url').groupby('region_url').tail(1)

# Take only the city name from the URL
ll['city'] = ll['region_url'].apply(lambda x: x.split('//')[1].split('.c')[0])
ll = ll[['region_url','city', 'state']].reset_index(drop=True)
ll['lat'] = 0.0
ll['lon'] = 0.0

#%% Connect to Google Geocode API to get lat/lon for each region

for i in range(ll.shape[0]):
    url = BASE_URL + ll['city'][i] + ',+' + ll['state'][i] + '&key=' + API_KEY 
    r = requests.get(url).json()
    ll.loc[i, 'lat'] = r['results'][0]['geometry']['location']['lat']
    ll.loc[i, 'lon'] = r['results'][0]['geometry']['location']['lng']
    time.sleep(1)

ll = ll.drop(['city', 'state'], axis = 1)
ll.to_csv(PROC_DIR + LL_FILE, index = False)
    
#%% Merge the lat/lon values into the original df and write file

co = df.merge(ll, on='region_url')
co.to_csv(PROC_DIR + OUT_FILE, compression = 'zip', index=False)
