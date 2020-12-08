# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Basic information
# %% [markdown]
# <li>The owner has been a host since August 2010
# <li>The location is lon:151.274506, lat:33.889087
# <li>The current review score rating 95.0
# <li>Number of reviews 53
# <li>Minimum nights 4
# <li>The house can accomodate 10 people.
# <li>The owner currently charges a cleaning fee of 370
# <li>The house has 3 bathrooms, 5 bedrooms, 7 beds.
# <li>The house is available for 255 of the next 365 days
# <li>The client is verified, and they are a superhost.
# <li>The cancelation policy is strict with a 14 days grace period.
# <li>The host requires a security deposit of $1,500
# 
# Data came from July 2018
# 

# %%
# creation of Sample Customer Data later called SCD, hold as dict
from dateutil import parser

SCD_dict = {}
SCD_dict["city"] = "Bondi Beach" 
SCD_dict["longitude"] = 151.274506 
SCD_dict["latitude"] = -33.88907 
SCD_dict["review_score_rating"] = 95 
SCD_dict["number_of_reviews"] = 53 
SCD_dict["minimum_nights"] = 4 
SCD_dict["accommodates"] = 10 
SCD_dict["bathrooms"] = 3 
SCD_dict["bedrooms"] = 5
SCD_dict["beds"] = 7
SCD_dict["security_deposit"] = 1500
SCD_dict["cleaning_fee"] = 370
SCD_dict["property_Type"] = "House"
SCD_dict["room_type"] = "Entire home/apt"
SCD_dict["availability_365"] = 255
SCD_dict["host_identity_verified"] = "t" 
SCD_dict["host_is_superhost"] = "t"
SCD_dict["cancellation_policy"] = "strict_14_with_grace_period"
SCD_dict["host_since"] = parser.parser("01-08-2010") 



# %%
## This is simply a bit of importing logic that you don't have ..
## .. to concern yourself with for now. 
import pandas as pd 
import numpy as np 
import os 
from pathlib import Path
import re 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 

github_p = "https://raw.githubusercontent.com/Finance-781/FinML/master/Lecture%202%20-%20End-to-End%20ML%20Project%20/Practice/"

my_file = Path("datasets/sydney_airbnb.csv") # Defines path
if my_file.is_file():              # See if file exists
    print("Local file found")      
    df = pd.read_csv('datasets/sydney_airbnb.csv')
else:
    print("Be patient: loading from github (2 minutes)")
    df = pd.read_csv(github_p+'datasets/sydney_airbnb.csv')
    print("Done")


# %%
df.columns


# %%
chosen = ["price","city","longitude","latitude","review_scores_rating","number_of_reviews","minimum_nights","security_deposit","cleaning_fee",
        "accommodates","bathrooms","bedrooms","beds","property_type","room_type","availability_365" ,"host_identity_verified", 
        "host_is_superhost","host_since","cancellation_policy"] 
df = df[chosen]


# %%
df.head()


# %%
df.columns


# %%
df.shape 


# %%
df.info()


# %%
# Some price values are held as strings, I dont want that, i will convert those to floats using regular expression and some pandas built in function

price_list = ["price", "cleaning_fee", "security_deposit"] #columns holding such values

for col in price_list:
    df[col] = df[col].fillna("0")
    df[col] = df[col].apply(lambda x: float(re.compile('[^0-9eE.]').sub('',x)) if len(x) > 0 else 0)


# %%
# changing date from "host since" to datetime wchich also is hold as a string value
df['host_since'] = pd.to_datetime(df['host_since'])


# %%
df.info()

# %% [markdown]
# ## Data exploration using box plots to check for outliers

# %%
sns.boxplot(y = df['price'])

# %% [markdown]
# ## Rule of thumb for skewness
# %% [markdown]
# <li>If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.
# <li>If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed.
# <li>If the skewness is less than -1 or greater than 1, the data are highly skewed.

# %%
# Using skew to check data distribution
df['price'].skew()

# %% [markdown]
# Well it seems we are skewed (bad pun intended)
# %% [markdown]
# ## Rule fo thumb for kurtosis
# %% [markdown]
# For kurtosis, the general guideline is that if the number is greater than +1, the distribution is too peaked. Likewise, a kurtosis of less than –1 indicates a distribution that is too flat. Distributions exhibiting skewness and/or kurtosis that exceed these guidelines are considered nonnormal

# %%
#Using kurtosis to check peakedness of a distribution
df['price'].kurtosis()


# %%
print(df['price'].quantile(0.995)) #99.5% percentile value
print(df['price'].mean())
print(df['price'].median())


# %%
#Lets keep all the data rows under the 99.5% value of 1600
df = df[df["price"] < df["price"].quantile(0.995)].reset_index(drop = True)


# %%
df.isnull().sum()


# %%
df.info()


# %%
df['cancellation_policy'].value_counts()


# %%
df['city'].value_counts()


# %%
df['property_type'].value_counts()


# %%
df['room_type'].value_counts()


# %%
df.describe()


# %%
df.columns


# %%
try:
    # We select all rows and all columns after the 6th
    df.iloc[:,6:].hist(bins=50, figsize=(20,15))
    plt.show()
except AttributeError:
    pass


# %%
df['minimum_nights'].hist(bins=50,figsize=(10,15))


# %%
df["city"].value_counts().head(10)


# %%
plt.figure(figsize=(16,10))
ax = sns.countplot(x="city", data=df)


# %%
#too much cities, lets  round it up to top 20 locations (Sydney)
list_of_20 = list(df['city'].value_counts().head(10).index)
df = df[df["city"].isin(list_of_20)].reset_index(drop=True)


# %%
df["property_type"].value_counts()


# %%
property_items_counts = df.groupby(['property_type']).size() #amount  of each property

rare_property = list(property_items_counts.loc[property_items_counts <= 10].index.values)

df = df[~df['property_type'].isin(rare_property)].reset_index(drop=True) #droping rare occuring properties, using ~bitwise NOT operator, neat trick


# %%
df['property_type'].value_counts()

# %% [markdown]
# ## Splitting and training model

# %%
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit


# %%
train_sest, test_set = train_test_split(df,test_size=0.2,random_state=42)


# %%
test_set.head()


# %%
# changing  values to numeric 
df["host_identity_verified"] = df["host_identity_verified"].apply(lambda x: 1 if x == "t" else 0)
df["host_is_superhost"] = df["host_is_superhost"].apply(lambda x: 1 if x == "t" else 0)


# %%
df["host_is_superhost"].sample(5)


# %%
# converting categorical data to numeric using sklearn's label encoder
le = preprocessing.LabelEncoder()

for  col in ["city"]:
    df[col +"_code"] = le.fit_transform(df[col])


# %%
df[['city',"city_code"]].sample(10)


# %%
#stratifying the data 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["city_code"]):
    del df["city_code"]
    strat_train_Set = df.loc[train_index]
    strat_test_Set = df.loc[test_index]


# %%
strat_test_Set.groupby("city")['price'].mean()

# %% [markdown]
# ## Some more exploration and Visualization

# %%
travel = strat_train_Set.copy()


# %%
travel.plot(kind = 'scatter', x = 'longitude', y = 'latitude')


# %%
travel.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.1)

# %% [markdown]
# This help to map the main area of intereset

# %%
#removing locations outside area of intereset
travel_co = travel[(travel['longitude']>151.16)&(travel['latitude']<-33.75)].reset_index(drop = True)

travel_co = travel_co[travel_co['latitude']>-33.95].reset_index(drop = True)
#Only locations under 600 bucks
travel_co = travel_co[travel_co['price']<600].reset_index(drop = True)


# %%
travel_co.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.5,
s = travel_co['number_of_reviews']/2, label = "Reviews", figsize = (10,7),
c = "price", cmap = plt.get_cmap('jet'), colorbar = True,
sharex = False)
plt.legend()


# %%
travel_co.shape

# %% [markdown]
# ## Correlation Matrix

# %%
corr_matrix = travel.corr()

plt.figure(figsize = (10,10))
cmap = sns.diverging_palette(220,10, as_cmap = True)

sns.heatmap(corr_matrix, xticklabels = corr_matrix.columns.values,
yticklabels = corr_matrix.columns.values, cmap = cmap, vmax = 1, center = 0, square = True,
linewidths = .5, cbar_kws = {'shrink' : .82})
plt.title('Correlation Matrix Heatmap')


# %%
corr_matrix["price"].sort_values(ascending=False)


# %%



