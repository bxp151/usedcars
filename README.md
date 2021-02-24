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
3. **OPTIONAL**: Run preprocessing code (this step takes 4-5 hours to complete).  This code connects to the NHTSA VIN decoder API to capture vehicle information. The output from this step is `/processed/combined.csv.zip` and is included as part of the repository    
	* Open `nhtsa.py` and set `HOME_DIR` to the directory path where you cloned the repository
	* Run `nhtsa.py`
	
4. The following steps are **in-progress** and will be updated as they are completed
	* populate lat/lon for reigons using API
	* classification model for descriptions (private / dealer)
	
