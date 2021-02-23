# Predicting passenger vehicle prices


## Objective

The objective is to predict prices for passenger vehicles on craigslist.

The [initial data set](https://www.kaggle.com/austinreese/craigslist-carstrucks-data) includes every used vehicle posted for sale in United States on Craigslist from October 23, 2020 to December 3, 2020.  I utilized [NHTSA's VIN decoder API](https://vpic.nhtsa.dot.gov/api/) to capture verifiable information about each vehicle and combined it with the initial data set.  I then used [Google's Geocoding API](https://developers.google.com/maps/documentation/geocoding/overview) to retrieve latitude and longitude numbers based on the region URLs.  

I'm currently working on building a text classifier using [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) to classify each sale as either private or dealer based on the description.   This classification is typically a signal for price differentiation. 

<br>


## Performance Measure
In order to measure the effectiveness of the model in predicting car prices, we used root mean square error (RMSE).  RMSE is calculated by: 

1. calculating the sum the squared differences between the predicted (model) and observed (test set) values 
2. dividing #1 by the numer of observations
3. taking the square root of #2



</br>


## Key Findings

1. ...
2. ...





## Model Validation
...


## Approach

The overall approach to building the default prediction model:

1. Preprocessing data
	* Initial data exploration
	* Collect decoded VIN information from  NHTSA's API
	* Combine initial data with decoded VIN information
2. Secondary data exploration
3. Split data into Train/Test
4. Build and analyze baseline models
5. Feature engineering
6. Build and analyze final models
7. Final predictions using test set

</br>


## Potential Model Improvements

...

<br>
<br>



## Data Definitions

Variable | Description 
---- | ----------- 
seller | type of seller (dealer/private)
bodyclass | style of body
displacementcc | the size of the engine in cubic centimeters
fueltypeprimary | the fuel type
make | the manufacturer (eg. Honda)
model | the  model (eg. Accord)
modelyear | the model year
posting_date | the date the ad was posted
price (target) | the seller's price 
condition | the car's condition
cylinders | the number of cylinders
size | the size category (eg. compact)
paint_color | the exterior color 
odometer | number of miles
lat | ad region's lattitude 
lon | ad region's longitude
