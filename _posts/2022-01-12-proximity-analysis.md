---
title: 'Proximity Analysis'
tags: [kaggle, geospatial analysis, maps]
layout: post
mathjax: true
categories: [Geospatial Analysis]
permalink: /blog/:title/
published: true
---
{% assign counter = 1 %}
{% assign counter2 = 1 %}
{% assign link = "https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/" %}
{% assign date = page.date | date: "%Y-%m-%d" %}
{% assign filename = page.title | remove: " -" | replace: " ", "-" | downcase %}



### Introduction 

You are part of a crisis response team, and you want to identify how hospitals have been responding to crash collisions in New York City.

<center>
<img src="https://i.imgur.com/wamd0n7.png" width="450"><br/>
</center>

Before you get started, run the code cell below to set everything up.


```python
import math
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon

import folium
from folium import Choropleth, Marker
from folium.plugins import HeatMap, MarkerCluster
```

You'll use the `embed_map()` function to visualize your maps.


```python
def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')
```


### 1) Visualize the collision data.

Run the code cell below to load a GeoDataFrame `collisions` tracking major motor vehicle collisions in 2013-2018.


```python
collisions = gpd.read_file("NYPD_Motor_Vehicle_Collisions.shp")
collisions.head()
```




<div class="table-wrapper" markdown="block">

|   |       DATE |  TIME |   BOROUGH | ZIP CODE |  LATITUDE |  LONGITUDE |                LOCATION |      ON STREET |    CROSS STRE |         OFF STREET | ... |  CONTRIBU_2 | CONTRIBU_3 | CONTRIBU_4 | UNIQUE KEY |                          VEHICLE TY |                          VEHICLE _1 |                          VEHICLE _2 | VEHICLE _3 | VEHICLE _4 |                       geometry |   |
|--:|-----------:|------:|----------:|---------:|----------:|-----------:|------------------------:|---------------:|--------------:|-------------------:|----:|------------:|-----------:|-----------:|-----------:|------------------------------------:|------------------------------------:|------------------------------------:|-----------:|-----------:|-------------------------------:|---|
| 0 | 07/30/2019 |  0:00 |     BRONX |    10464 | 40.841100 | -73.784960 |    (40.8411, -73.78496) |           None |          None |   121 PILOT STREET | ... | Unspecified |       None |       None |    4180045 |                               Sedan | Station Wagon/Sport Utility Vehicle | Station Wagon/Sport Utility Vehicle |       None |       None | POINT (1043750.211 245785.815) |   |
| 1 | 07/30/2019 |  0:10 |    QUEENS |    11423 | 40.710827 | -73.770660 |  (40.710827, -73.77066) | JAMAICA AVENUE |    188 STREET |               None | ... |        None |       None |       None |    4180007 |                               Sedan |                               Sedan |                                None |       None |       None | POINT (1047831.185 198333.171) |   |
| 2 | 07/30/2019 |  0:25 |      None |     None | 40.880318 | -73.841286 | (40.880318, -73.841286) |    BOSTON ROAD |          None |               None | ... |        None |       None |       None |    4179575 |                               Sedan | Station Wagon/Sport Utility Vehicle |                                None |       None |       None | POINT (1028139.293 260041.178) |   |
| 3 | 07/30/2019 |  0:35 | MANHATTAN |    10036 | 40.756744 | -73.984590 |  (40.756744, -73.98459) |           None |          None | 155 WEST 44 STREET | ... |        None |       None |       None |    4179544 |                           Box Truck | Station Wagon/Sport Utility Vehicle |                                None |       None |       None |  POINT (988519.261 214979.320) |   |
| 4 | 07/30/2019 | 10:00 |  BROOKLYN |    11223 | 40.600090 | -73.965910 |   (40.60009, -73.96591) |       AVENUE T | OCEAN PARKWAY |               None | ... |        None |       None |       None |    4180660 | Station Wagon/Sport Utility Vehicle |                                Bike |                                None |       None |       None |  POINT (993716.669 157907.212) |   |

</div>



Use the "LATITUDE" and "LONGITUDE" columns to create an interactive map to visualize the collision data.  What type of map do you think is most effective?


```python
m_1 = folium.Map(location=[40.7, -74], zoom_start=11) 

# Your code here: Visualize the collision data
# Visualize the collision data
HeatMap(data=collisions[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m_1)

# Show the map
embed_map(m_1, "q_1.html")
```





<iframe
    width="100%"
    height="500px"
    src="https://sourestdeeds.github.io/maps/q2_1.html"
    frameborder="0"
    allowfullscreen

></iframe>




### 2) Understand hospital coverage.

Run the next code cell to load the hospital data.


```python
hospitals = gpd.read_file("nyu_2451_34494.shp")
hospitals.head()
```




<div class="table-wrapper" markdown="block">

|   |             id |                                              name |                address |   zip | factype |  facname | capacity | capname | bcode |    xcoord |   ycoord |  latitude |  longitude |                       geometry |
|--:|---------------:|--------------------------------------------------:|-----------------------:|------:|--------:|---------:|---------:|--------:|------:|----------:|---------:|----------:|-----------:|-------------------------------:|
| 0 | 317000001H1178 | BRONX-LEBANON HOSPITAL CENTER - CONCOURSE DIVI... |   1650 Grand Concourse | 10457 |    3102 | Hospital |      415 |    Beds | 36005 | 1008872.0 | 246596.0 | 40.843490 | -73.911010 | POINT (1008872.000 246596.000) |
| 1 | 317000001H1164 |   BRONX-LEBANON HOSPITAL CENTER - FULTON DIVISION |        1276 Fulton Ave | 10456 |    3102 | Hospital |      164 |    Beds | 36005 | 1011044.0 | 242204.0 | 40.831429 | -73.903178 | POINT (1011044.000 242204.000) |
| 2 | 317000011H1175 |                              CALVARY HOSPITAL INC | 1740-70 Eastchester Rd | 10461 |    3102 | Hospital |      225 |    Beds | 36005 | 1027505.0 | 248287.0 | 40.848060 | -73.843656 | POINT (1027505.000 248287.000) |
| 3 | 317000002H1165 |                             JACOBI MEDICAL CENTER |       1400 Pelham Pkwy | 10461 |    3102 | Hospital |      457 |    Beds | 36005 | 1027042.0 | 251065.0 | 40.855687 | -73.845311 | POINT (1027042.000 251065.000) |
| 4 | 317000008H1172 |            LINCOLN MEDICAL & MENTAL HEALTH CENTER |           234 E 149 St | 10451 |    3102 | Hospital |      362 |    Beds | 36005 | 1005154.0 | 236853.0 | 40.816758 | -73.924478 | POINT (1005154.000 236853.000) |

</div>



Use the "latitude" and "longitude" columns to visualize the hospital locations. 


```python
m_2 = folium.Map(location=[40.7, -74], zoom_start=11) 

# Visualize the hospital locations
for idx, row in hospitals.iterrows():
    Marker([row['latitude'], row['longitude']], popup=row['name']).add_to(m_2)
        
# Show the map
embed_map(m_2, "q_2.html")
```





<iframe
    width="100%"
    height="500px"
    src="https://sourestdeeds.github.io/maps/q2_2.html"
    frameborder="0"
    allowfullscreen

></iframe>




### 3) When was the closest hospital more than 10 kilometers away?

Create a DataFrame `outside_range` containing all rows from `collisions` with crashes that occurred more than 10 kilometers from the closest hospital.

Note that both `hospitals` and `collisions` have EPSG 2263 as the coordinate reference system, and EPSG 2263 has units of meters.


```python
# Your code here
coverage = gpd.GeoDataFrame(geometry=hospitals.geometry).buffer(10000)
my_union = coverage.geometry.unary_union
outside_range = collisions.loc[~collisions["geometry"].apply(lambda x: my_union.contains(x))]
```

The next code cell calculates the percentage of collisions that occurred more than 10 kilometers away from the closest hospital.


```python
percentage = round(100*len(outside_range)/len(collisions), 2)
print("Percentage of collisions more than 10 km away from the closest hospital: {}%".format(percentage))
```

    Percentage of collisions more than 10 km away from the closest hospital: 15.12%


### 4) Make a recommender.

When collisions occur in distant locations, it becomes even more vital that injured persons are transported to the nearest available hospital.

With this in mind, you decide to create a recommender that:
- takes the location of the crash (in EPSG 2263) as input,
- finds the closest hospital (where distance calculations are done in EPSG 2263), and 
- returns the name of the closest hospital. 


```python
def best_hospital(collision_location):
    idx_min = hospitals.geometry.distance(collision_location).idxmin()
    my_hospital = hospitals.iloc[idx_min]
    name = my_hospital["name"]
    return name

# Test your function: this should suggest CALVARY HOSPITAL INC
print(best_hospital(outside_range.geometry.iloc[0]))
```

    CALVARY HOSPITAL INC


### 5) Which hospital is under the highest demand?

Considering only collisions in the `outside_range` DataFrame, which hospital is most recommended?  

Your answer should be a Python string that exactly matches the name of the hospital returned by the function you created in **4)**.


```python
# Your code here
highest_demand = outside_range.geometry.apply(best_hospital).value_counts().idxmax()
```

### 6) Where should the city construct new hospitals?

Run the next code cell (without changes) to visualize hospital locations, in addition to collisions that occurred more than 10 kilometers away from the closest hospital. 


```python
m_6 = folium.Map(location=[40.7, -74], zoom_start=11) 

coverage = gpd.GeoDataFrame(geometry=hospitals.geometry).buffer(10000)
folium.GeoJson(coverage.geometry.to_crs(epsg=4326)).add_to(m_6)
HeatMap(data=outside_range[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m_6)
folium.LatLngPopup().add_to(m_6)

embed_map(m_6, 'm_6.html')
```





<iframe
    width="100%"
    height="500px"
    src="https://sourestdeeds.github.io/maps/m2_6.html"
    frameborder="0"
    allowfullscreen

></iframe>




Click anywhere on the map to see a pop-up with the corresponding location in latitude and longitude.

The city of New York reaches out to you for help with deciding locations for two brand new hospitals.  They specifically want your help with identifying locations to bring the calculated percentage from step **3)** to less than ten percent.  Using the map (and without worrying about zoning laws or what potential buildings would have to be removed in order to build the hospitals), can you identify two locations that would help the city accomplish this goal?  

Put the proposed latitude and longitude for hospital 1 in `lat_1` and `long_1`, respectively.  (Likewise for hospital 2.)

Then, run the rest of the cell as-is to see the effect of the new hospitals.  Your answer will be marked correct, if the two new hospitals bring the percentage to less than ten percent.


```python
# Proposed location of hospital 1
lat_1 = 40.6714
long_1 = -73.8492

# Proposed location of hospital 2
lat_2 = 40.6702
long_2 = -73.7612


# Do not modify the code below this line
new_df = pd.DataFrame(
    {'Latitude': [lat_1, lat_2],
     'Longitude': [long_1, long_2]})
new_gdf = gpd.GeoDataFrame(new_df, geometry=gpd.points_from_xy(new_df.Longitude, new_df.Latitude))
new_gdf.crs = {'init' :'epsg:4326'}
new_gdf = new_gdf.to_crs(epsg=2263)
# get new percentage
new_coverage = gpd.GeoDataFrame(geometry=new_gdf.geometry).buffer(10000)
new_my_union = new_coverage.geometry.unary_union
new_outside_range = outside_range.loc[~outside_range["geometry"].apply(lambda x: new_my_union.contains(x))]
new_percentage = round(100*len(new_outside_range)/len(collisions), 2)
print("(NEW) Percentage of collisions more than 10 km away from the closest hospital: {}%".format(new_percentage))
# make the map
m = folium.Map(location=[40.7, -74], zoom_start=11) 
folium.GeoJson(coverage.geometry.to_crs(epsg=4326)).add_to(m)
folium.GeoJson(new_coverage.geometry.to_crs(epsg=4326)).add_to(m)
for idx, row in new_gdf.iterrows():
    Marker([row['Latitude'], row['Longitude']]).add_to(m)
HeatMap(data=new_outside_range[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m)
folium.LatLngPopup().add_to(m)
display(embed_map(m, 'q_6.html'))
```

    /usr/local/anaconda3/lib/python3.9/site-packages/pyproj/crs/crs.py:131: FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method. When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6
      in_crs_string = _prepare_from_proj_string(in_crs_string)


    (NEW) Percentage of collisions more than 10 km away from the closest hospital: 9.12%




<iframe
    width="100%"
    height="500px"
    src="https://sourestdeeds.github.io/maps/q2_6.html"
    frameborder="0"
    allowfullscreen

></iframe>

