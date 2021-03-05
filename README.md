# Predicting passenger vehicle prices


## Objective

The objective is to predict prices for passenger vehicles on craigslist.

The [initial data set](https://www.kaggle.com/austinreese/craigslist-carstrucks-data) includes every used vehicle posted for sale in United States on Craigslist from October 23, 2020 to December 3, 2020.  I utilized [NHTSA's VIN decoder API](https://vpic.nhtsa.dot.gov/api/) to capture verifiable information about each vehicle and combined it with the initial data set.  I then used [Google's Geocoding API](https://developers.google.com/maps/documentation/geocoding/overview) to retrieve latitude and longitude numbers based on the region URLs.  I also trained two text classifiers using [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) and [Google BERT](https://huggingface.co/transformers/model_doc/bert.html) to classify each sale as either private or dealer based on the seller description.

<br>


## Executive summary

The executive summary is located in [REPORT.md](./REPORT.md)

<br>


## Usage Instructions

1. Clone the repository: `git clone https://github.com/bxp151/usedcars.git` 
2. Install the required packages: `pip install -r requirements.txt `
3. From here you skip 4-6 and jump right to step 7
4. **OPTIONAL**: (this step takes 4-5 hours to complete).  This code connects to the NHTSA VIN decoder API to capture vehicle information. The output from this step is `/processed/nhtsa.csv.zip` and is included as part of the repository    
	* Open `nhtsa.py` and set `HOME_DIR` to the directory path where you cloned the repository
	* Run `nhtsa.py`
5. **OPTIONAL**: (this step takes ~20 mins to complete and requires a google API key).  This code connects to the Google Geocoding API to get the lattitue and longitude of each Craigslist server. The output from this step is `/processed/latlon.csv.zip` and is included as part of the repository    
	* Open `latlon.py` and set `HOME_DIR` to the directory path where you cloned the repository
	* Run `latlon.py`
6. **OPTIONAL**: (this step take 4-8 hours to complete and requires a google account). This code creates a feature "seller" based on manually labeling a sample of the training data as either "private" or "dealer" based on the description.  The output from this step is `seller.csv` which is manually downloaded from Google Collab.  This file should be moved in the the `/data` directory.
   * Run `description.py` to see how the sample of training data was taken
	* Take a copy of the [collab notebooks](https://drive.google.com/drive/folders/1zt6PbSovX9jnWfoiHMly1FxGiFGzSazz?usp=sharing) 
	* Run the distilbert first then the bert notebook
7. Finally, run `usedcars.py`
