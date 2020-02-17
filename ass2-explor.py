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

# Clean up code for hospital data
point_of_intrest = pd.read_csv("Point_Of_Interest.csv")
point_of_intrest =  point_of_intrest[point_of_intrest['NAME'].str.contains("HOSPITAL")]
point_of_intrest['the_geom'] = point_of_intrest['the_geom'].str.split(' ')
point_of_intrest['latitude'] = point_of_intrest['the_geom'].apply(lambda x: x[1])
point_of_intrest['longitude'] = point_of_intrest['the_geom'].apply(lambda x: x[0])
# print(point_of_intrest['latitude'])
# print(point_of_intrest['longitude'])

# Get distance to hospitals

mat_hospital = scipy.spatial.distance.cdist(df[['latitude','longitude']], 
                              point_of_intrest[['latitude','longitude']], metric='euclidean')

min_distance_to_hospital = []
for listing in mat_hospital:
    min_distance_to_hospital.append(min(listing))

df['closest_hospital'] = min_distance_to_hospital

# Get distance to subway stations
mat = scipy.spatial.distance.cdist(df[['latitude','longitude']], 
                              station[['Station Latitude','Station Longitude']], metric='euclidean')


# print(len(mat[0]))
# print(len(mat))
# print(min(mat[0]))

min_distance_to_station = []

for listing in mat:
    min_distance_to_station.append(min(listing))

# print(min_distance_to_station)

df['closest_station'] = min_distance_to_station
print(df)
