# Predicting passenger vehicle prices


## Objective

The objective is to predict prices for passenger vehicles on craigslist.

The [initial data set](https://www.kaggle.com/austinreese/craigslist-carstrucks-data) includes every used vehicle posted for sale in United States on Craigslist from October 23, 2020 to December 3, 2020.  I utilized [NHTSA's VIN decoder API](https://vpic.nhtsa.dot.gov/api/) to capture verifiable information about each vehicle and combined it with the initial data set.  I then used [Google's Geocoding API](https://developers.google.com/maps/documentation/geocoding/overview) to retrieve latitude and longitude numbers based on the region URLs.  I also trained two text classifiers using [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) and [Google BERT](https://huggingface.co/transformers/model_doc/bert.html) to classify each sale as either private or dealer based on the seller description.

<br>


## Performance Measure
In order to measure the effectiveness of the model in predicting car prices, we used root mean square error (RMSE).  RMSE is calculated by: 

1. calculating the sum the squared differences between the predicted (model) and observed (test set) values 
2. dividing #1 by the numer of observations
3. taking the square root of #2



</br>


## Key Findings

1. The final model achieved an RMSE of $4100.00 on the test data 
2. The 5 most important features used to predict price were:

	* **bodyclass --**  style of body (e.g. sedan)
	* **odometer --** number of miles 
	* **make --** manufacturer (e.g Honda)
	* **paint_color --** exterior color
	* **seller --** seller type (private, dealer)

![](./images/fig5.png)




## Model Validation
In order to validate the model, we used target shuffling which shows the probability that the model's results occured by chance. 
    
    For 400 repititions:
        
        1. Shuffle the target 
        2. Fit the best model to the shuffled target (shuffled model)
        3. Make predictions using the shuffled model and score using AUC
        4. Plot the distribution of scores 
</br>

Since the best model performed better than every target permutation model, there is a 0 in 400 probability that the model's results occured by chance

![](./images/fig6.png)


## Approach

The overall approach to building the default prediction model:

1. Preprocessing data
	* Initial data exploration
	* Collect decoded VIN information from  NHTSA's API
	* Combine initial data with decoded VIN information
	* Build text classifier to populate "seller" feature (private vs. dealer) based on description feature
2. Secondary data exploration
3. Split data into Train/Test
4. Build and analyze baseline models
5. Feature engineering
6. Build and analyze final models
7. Final predictions using test set

</br>


## Potential Model Improvements

1. An attempt was made to collect vehicle manufacturer suggested retail prices (MSRPs) for each observation.  Unfortunately, there were no freely available sources of that information.  I think that this feature would have improved predictions significantly.  
2. There is oppporutnity to reduce the # of features included in the model since most featuers outside of the top 5 were unimportant


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
seller | seller type (private, dealer)
