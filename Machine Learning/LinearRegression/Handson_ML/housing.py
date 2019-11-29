
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
