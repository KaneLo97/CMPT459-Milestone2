import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from sklearn import datasets
import scipy

df = pd.read_json("train.json.zip")
df.head()


station = pd.read_csv("nyc-transit-data.csv")
station = station.filter(['Station Name', 'Station Latitude', 'Station Longitude'])
station.drop_duplicates(inplace=True)

mat = scipy.spatial.distance.cdist(df[['latitude','longitude']], 
                              station[['Station Latitude','Station Longitude']], metric='euclidean')


print(len(mat[0]))
print(len(mat))
print(min(mat[0]))

min_distance_to_station = []

for listing in mat:
    min_distance_to_station.append(min(listing))

# print(min_distance_to_station)

df['distance_to_closest'] = min_distance_to_station
# print(df)
