#%% Imports
import os
import pandas as pd
import exploretransform as et
import numpy as np
import plotly.express as px
from plotly.offline import plot 
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, GammaRegressor
import xgboost as xgb
import category_encoders as ce


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
    drop = ['vin','vehicletype','state','region_url','type',
        'image_url','title_status', 'cylinders', 'posting_date']
    df = df.drop(drop, axis = 1)
    df[['modelyear', 'odometer','displacementcc']] =\
        df[['modelyear', 'odometer', 'displacementcc']].astype(int)
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

tr = train.copy()

# et.peek(tr)
# et.explore(tr)

''' 
Initial Feature Engineering:
    1. calculate age based off model year and drop modelyear
    2. Impute 'missing' for condition, paint_color, size(temp)
    3. Temporarity imput 'missing' for size for anlysis
'''
tr['age'] = 2021 - tr['modelyear']
tr.drop('modelyear', axis=1, inplace=True)
tr.loc[tr['condition'].isna(), 'condition'] = 'missing'
tr.loc[tr['paint_color'].isna(), 'paint_color'] = 'missing'
tr.loc[tr['size'].isna(), 'size'] = 'missing'
          


tr.hist()
fig4 = px.histogram(tr, 'odometer')
plot(fig4)
tr.describe()
et.skewstats(tr)
# pd.plotting.scatter_matrix(tr.drop(['lat','lon'], axis=1), alpha=0.2) 
pd.plotting.scatter_matrix(tr, alpha=0.2) 

''' 
Numeric feature notes 
    1. most cars engines are under 3 liters
    2. most cars are 10 years or younger
    3. most cars have under 200K miles
    4. positive relationship between displacement & price
    5. negative relationship bewteen age & price
'''



cat_cols = tr.select_dtypes('object').columns

for col in cat_cols[0:4]:
    print(tr[col].value_counts())
    
model = tr[cat_cols[4]].value_counts()
model.head(10)

for col in cat_cols[5:]:
    print(tr[col].value_counts())
    
'''
Categorical feature notes (univariate)

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
        
'''


# Categorical Box plots
for col in cat_cols:
    plot(px.box(tr, x =col, y = 'price'))

'''
Categorical feature (bivariate)

    1. Yellow and orange seem to be more expensive than other colors, possibly
    because those are more likely to be on exotic cars
    2. There is overlap but dealer seem to be more expensive than private 
    3. Coupes and convertibles make up most of the very high priced cars
    4. There are makes amd models that are much more expensive than average
    5. Cars listed as fair or salvage are significantly different in price 
    (lower)
    6. Size has many missing values and none of the categories seems to be 
    very different - maybe not include

'''

# create dataframe to map average prices and counts        
loc_prices = tr[['lat','lon','region','price']].\
             sort_values(['lat', 'lon', 'region']).\
             groupby(['lat', 'lon', 'region']).\
             agg(avg_price=('price', 'mean'),count=('price','count')).\
             reset_index().sort_values('avg_price', ascending=False)
    
loc_prices[loc_prices['count'] > 300].drop(['lat', 'lon'], axis = 1).head(5)

plot(px.scatter_mapbox(loc_prices, lat = 'lat', lon='lon', 
                       size='count',
                       color='avg_price',
                       color_continuous_scale=px.colors.cyclical.IceFire,
                       mapbox_style='open-street-map',
                       zoom = 2))

'''
Top 5 highest average prices with post count > 300
    1. San Fran
    2. Hawaii
    3. San Diego
    4. Stockton
    5, Anchorage
'''


''' function to perform feature engineering steps for train/test '''
def feat_eng(df):
    df = df.copy()
    df['age'] = 2021 - df['modelyear']
    df.drop(['modelyear','region'], axis=1, inplace=True)
    df.loc[df['condition'].isna(), 'condition'] = 'missing'
    df.loc[df['paint_color'].isna(), 'paint_color'] = 'missing'
    df.loc[df['size'].isna(), 'size'] = 'missing'
    return df


# validate final transformation 
et.explore(feat_eng(train))
et.peek(feat_eng(train))

#%% Pipelines
'''
Linear, GLM, Penalized
-Center/Scale numeric

Random Forest, Adaboost, Gradient Boost, XGBoost

All
-LOO Target encoding

'''

X_train, y_train = feat_eng(train).drop('price', axis = 1), feat_eng(train)['price']

num_cols = X_train.select_dtypes('number').columns
cat_cols = X_train.select_dtypes('object').columns

''' For non-parametric models '''
num_pipe1 = Pipeline([
    ('select', et.ColumnSelect(num_cols))
    ])

cat_pipe1 = Pipeline([
    ('select', et.ColumnSelect(cat_cols)),
    ('loo', ce.LeaveOneOutEncoder(sigma=0.05))
    ])

union1 = FeatureUnion([
    ('num_pipe1', num_pipe1),
    ('cat_pipe', cat_pipe1)
    ])

''' For parametric models '''
num_pipe2 = Pipeline([
    ('select', et.ColumnSelect(num_cols)),
    ('scaler', StandardScaler() )
    ])

cat_pipe2 = Pipeline([
    ('select', et.ColumnSelect(cat_cols)),
    ('loo', ce.LeaveOneOutEncoder(sigma=0.05)),
    ('scaler', StandardScaler() )
    ])

union2 = FeatureUnion([
    ('num_pipe2', num_pipe2),
    ('cat_pipe', cat_pipe2)
    ])


#%% Cross validator

def cv_score(estimator, pipeline):
    return np.abs(np.mean(cross_val_score(\
                   estimator,
                   pipeline.fit_transform(X_train, y_train), 
                   y_train, 
                   cv = 5, 
                   scoring = 'neg_root_mean_squared_error')))


#%% Baseline Models

''' Naive '''
# Predict mean
base00 = np.sqrt(np.sum((y_train - np.mean(y_train))**2)/len(y_train))

''' Parametric '''

# Linear Regression 
base01 = cv_score(LinearRegression(), union2)

# Lasso 
base02 = cv_score(Lasso(), union2)

''' Non-Parametric '''

# Random Forest
base03 = cv_score(RandomForestRegressor(max_features='sqrt', random_state=42), union1)

# Gradient Boost
base04 = cv_score(GradientBoostingRegressor(random_state=42), union1)

# XGBoost
base05 = cv_score(xgb.XGBRegressor(random_state=42), union1)



