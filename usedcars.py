#%% Imports
import os
import pandas as pd
import exploretransform as et
import numpy as np
import plotly.express as px
from plotly.offline import plot 
from sklearn.model_selection import StratifiedShuffleSplit

#%% Settings
HOME_DIR = '/Users/bxp151/ml/usedcars'
os.chdir(HOME_DIR)

IMG_DIR = HOME_DIR + '/images'
PROC_DIR = HOME_DIR + '/processed'

IN_FILE = '/full.csv'


if not os.path.exists("images"):
    os.mkdir(IMG_DIR)
    
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
#%% Load data

def load_data(file):
    return pd.read_csv(file)
    
df = load_data(PROC_DIR + IN_FILE)
#%% Initial Explore

print('\nRows: ' + str(df.shape[0]) + ' \nColumns: ' +  str(df.shape[1]) )

et.peek(df)
et.explore(df)

'''
delete: vin,vehicletype,state,region,region_url,type,image_url,title_status,
        cylinders,posting_date
change: modelyear, odometer to int
'''

fig1 = px.histogram(df, x='price')
plot(fig1)

df[['make', 'price']].sort_values(by='price', ascending=False).head(400)

'''
Prices are heavily right skewed.  There are some very pricey cars for sale
such as Ferarri, Lamborghini, Rolls Royce
'''
#%% Initial cleanup

def init_clean(df):
    drop = ['vin','vehicletype','state','region','region_url','type',
        'image_url','title_status', 'cylinders', 'posting_date']
    df = df.drop(drop, axis = 1)
    df[['modelyear', 'odometer']] = df[['modelyear', 'odometer']].astype(int)
    return df

df = init_clean(df)

#%% Split data 

def strat_split(df, target, quantiles):
    '''
    Python doesn't have a built in mechanism for stratification in regression
    Specifiy the quantiles and target to create stratified samples
    '''
    
    # calculate the quantiles for the target
    q = df[target].quantile(quantiles)
    
    # create numeric labels for the new category
    l = list(range(1,len(quantiles),1))
    
    # create the new target category
    df["target_cat"] = pd.cut(df[target], bins = q, right=True,
                             labels = l, include_lowest = True) 
    
    traintest = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)

    for train_idx, test_idx in traintest.split(df, df["target_cat"]):
        train= df.iloc[train_idx]
        test = df.iloc[test_idx]

    train, test = train.drop(['target_cat'], axis = 1), test.drop(['target_cat'], axis = 1)  
    
    return train, test

''' Using quantiles to ensure high priced cars are in train and test sets '''
train, test = strat_split(df, target = "price", 
                          quantiles = [0, .25 , .50 , .75, .9975, 1])

''' Verify price histogram look the same between train and test '''
fig2 = px.histogram(train, x='price')
plot(fig2)

fig3 = px.histogram(test, x='price')
plot(fig3)

#%% EDA
et.peek(train)
et.explore(train)

train['age'] = 2020 - train['modelyear']
train.drop('modelyear', axis=1, inplace=True)

''' 
Numeric Features 
    1. most cars engines are under 3 liters
    2. most cars are 10 years or younger
    3. most cars have under 200K miles
'''

train.hist()
fig4 = px.histogram(train, 'odometer')
plot(fig4)
train.describe()
et.skewstats(train)
pd.plotting.scatter_matrix(train, alpha=0.2)

'''
Categorical Features

    1 Top 5 models
        Civic      1333
        Accord     1314
        Camry      1273
        Altima     1084
        Fusion     1063
    
    2 Top 5 manufacturers
        TOYOTA           3842
        FORD             3533
        CHEVROLET        3442
        HONDA            2980
        NISSAN           2606

    3 Most cars are Sedan, Hatchback or Coupe
        Sedan/Saloon                    23676
        Hatchback/Liftback/Notchback    4118
        Coupe                           3132
        
    4 Top 5 colors
        black     5837
        white     5267
        silver    4833
        grey      3040
        blue      3034
    
    5 Most posters claim cars are in excellent condition
        excellent    10121
        good          7764
        like new      1507
        
    6 For all missing values, impute "missing"
    

'''

cat_cols = train.select_dtypes('object').columns

for col in cat_cols[0:4]:
    print(train[col].value_counts())
    
model = train[cat_cols[4]].value_counts()
model.head(10)

for col in cat_cols[5:]:
    print(train[col].value_counts())



