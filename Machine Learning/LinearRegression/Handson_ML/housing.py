
import os
import tarfile
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join('C:\\handson_ML\\virtualenv','datasets', 'housing')
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# =============================================================================
# #Fetching data from github url
# =============================================================================
def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exits_ok = True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()
  
# =============================================================================
# #Loading housing data into pandas
# =============================================================================
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()

# =============================================================================
# #Check the info, ocean_proximity value counts and describe
# =============================================================================
def check_info_ocean_proximity_and_describe(data):
    print("************************************")
    print("Housing data Information \n %s"%(data.info()))
    print("************************************")
    print("Ocean Proximity Value Counts \n %s"%data['ocean_proximity'].value_counts())
    print("************************************")
    print("Describe housing data \n %s"%data.describe())

#uncomment below lines to run the code
# =============================================================================
# check_info_ocean_proximity_and_describe(housing)
# =============================================================================


# =============================================================================
# #Plot data
# =============================================================================
def plot_data(data):
    data.hist(bins = 50, figsize = (20,15))
    plt.show()
    
#uncomment below lines to run the code
# =============================================================================
# plot_data(housing)
# =============================================================================
    
    
# =============================================================================
# #Create test data using random.permutation. It will generate a random
# #different data sets on each run
# =============================================================================
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#uncomment below lines to run the code
# =============================================================================
# train_set, test_set = split_train_test(housing, 0.2)
# print("Train Set with ration 0.8 \n%s"%train_set)
# print("Test Set with ration 0.2 \n%s"%test_set)
# =============================================================================


# =============================================================================
# To have a stable train/test split even after updating the dataset, a common
# solution is to use each instance's identifier to decide whether or not it 
# should go in the test set. we can acheive this by creating a unique row identifer
# to the housing data. This can be done by creating test set with hash is lower than
# or equal to 20% of the maximum has value.
# =============================================================================
def split_train_test_by_id(data,  test_ratio, id_column):
    data = data.reset_index() #add one more index column to dataset(housing)
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

#crc is cyclic redundancy check, this function, i will explain later
def test_set_check(identifier, test_ratio):
    from zlib import crc32
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32

#uncomment below lines to run the code
# =============================================================================
# housing = load_housing_data()
# train_set, test_set = split_train_test_by_id(housing, 0.2, "index")
# print("Train Set with ration 0.8 \n%s"%train_set)
# print("Test Set with ration 0.2 \n%s"%test_set)
# =============================================================================

# =============================================================================
# #Another way to create unique identifier with latitude and longitude and it 
# #should guaranteed to be stable for a few million years,
# =============================================================================
def test_by_latitude_longitude(data, test_ratio, id_column):
    data[id_column] = data['longitude'] * 1000 + data['latitude']
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]    

#If you want to run above code, uncomment below lines
# =============================================================================
# housing = load_housing_data()
# train_set, test_set = test_by_latitude_longitude(housing, 0.2, "id")
# print("Train Set with ration 0.8 \n%s"%train_set)
# print("Test Set with ration 0.2 \n%s"%test_set)
# =============================================================================


# =============================================================================
# Scikit-Learn package also provides few functions to split datasets into multiple
# subsets in various ways.
# Below is one of the example using scikit learn to split train and test data
# =============================================================================
from sklearn.model_selection import train_test_split

def split_train_test_by_sklearn(data, test_ratio):
    return train_test_split(data, test_size = test_ratio, random_state=42)

#If you want to run above code, uncomment below lines
# =============================================================================
# housing = load_housing_data()
# train_set, test_set = split_train_test_by_sklearn(housing, 0.2)
# print("Train Set with ration 0.8 \n%s"%train_set)
# print("Test Set with ration 0.2 \n%s"%test_set)8
# =============================================================================


# =============================================================================
# Stratified Sampling
# The above test sets are random data sets and not categorical.
# In Stratfied Sampling we divide data into various categories, since median income
# is a continous numerical attribute, we first need to create an income category
# attribute. pd.cut() function to create an income category attribute with five categories
# labelled from 1 to 5, category 1 ranges from 0 to 1.5, cat 2 ranges from 1.5 to 3 and so on.
# =============================================================================
    
def create_income_category_attribute(data):
    
    data['income_cat'] = pd.cut(data['median_income'],
                                bins=[0., 1.5, 3., 4.5, 6., np.inf],
                                labels = [1, 2, 3, 4, 5])
    data['income_cat'].hist()
    return data
    
#If you want to run above code, uncomment below lines
# =============================================================================
# housing = load_housing_data()
# create_income_category_attribute(housing)
# =============================================================================

# =============================================================================
# Now we are ready to stratified sampling based on income category. for this we can 
# use sklearn StratifiedSplit class.
# =============================================================================

def stratified_split_train_test(data):
    
    data = create_income_category_attribute(data)
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=42)
    for train_index, test_index in split.split(data, data['income_cat']):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
        
    return strat_train_set, strat_test_set

housing = load_housing_data()
train_set, test_set = stratified_split_train_test(housing)

print(train_set['income_cat'].value_counts() / len(train_set))
print(test_set['income_cat'].value_counts() / len(test_set))


