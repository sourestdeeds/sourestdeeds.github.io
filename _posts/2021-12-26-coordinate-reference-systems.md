---
title: 'Coordinate Reference Systems'
tags: [kaggle, geospatial analysis, geopandas, coordinate reference systems]
layout: post
mathjax: true
categories: [Geospatial Analysis]
---
{% assign counter = 1 %}
{% assign link = "https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/" %}
{% assign date = page.date | date: "%Y-%m-%d" %}
{% assign filename = page.title | remove: " -" | replace: " ", "-" | downcase %}

The maps you create in this course portray the surface of the earth in two dimensions. But, as you know, the world is actually a three-dimensional globe. So we have to use a method called a **map projection** to render it as a flat surface.

Map projections can't be 100% accurate. Each projection distorts the surface of the Earth in some way, while retaining some useful property. For instance,

- The *equal-area* projections (like "Lambert Cylindrical Equal Area", or "Africa Albers Equal Area Conic") preserve area. This is a good choice, if you'd like to calculate the area of a country or city, for example.
- The *equidistant* projections (like "Azimuthal Equidistant projection") preserve distance. This would be a good choice for calculating flight distance.

<br>
<picture>
<source type="image/webp" srcset="{{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp">
<source type="image/png" srcset="{{ link }}{{ date }}-{{ filename }}/{{ counter }}.png">
<img src="{{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center" class=lightbox/>
</picture>
{% assign counter = counter | plus: 1 %} 
<br>

We use a coordinate reference system (CRS) to show how the projected points correspond to real locations on Earth. In this tutorial, you'll learn more about coordinate reference systems, along with how to use them in GeoPandas.

```python
import geopandas as gpd
import pandas as pd
```

### Setting the CRS

When we create a GeoDataFrame from a shapefile, the CRS is already imported for us.

```python
# Load a GeoDataFrame containing regions in Ghana
regions = gpd.read_file("../input/geospatial-learn-course-data/ghana/ghana/Regions/Map_of_Regions_in_Ghana.shp")
print(regions.crs)
```

    epsg:32630

How do you interpret that?

Coordinate reference systems are referenced by [European Petroleum Survey Group](http://www.epsg.org/) (EPSG) codes.

This GeoDataFrame uses [EPSG 32630](https://epsg.io/32630), which is more commonly called the "Mercator" projection. This projection preserves angles (making it useful for sea navigation) and slightly distorts area.

However, when creating a GeoDataFrame from a CSV file, we have to set the CRS. [EPSG 4326](https://epsg.io/4326) corresponds to coordinates in latitude and longitude.

```python
# Create a DataFrame with health facilities in Ghana
facilities_df = pd.read_csv("../input/geospatial-learn-course-data/ghana/ghana/health_facilities.csv")

# Convert the DataFrame to a GeoDataFrame
facilities = gpd.GeoDataFrame(facilities_df, geometry=gpd.points_from_xy(facilities_df.Longitude, facilities_df.Latitude))

# Set the coordinate reference system (CRS) to EPSG 4326
facilities.crs = {'init': 'epsg:4326'}

# View the first five rows of the GeoDataFrame
facilities.head()
```

<div class="table-wrapper" markdown="block">

|   | Region  | District         | FacilityName            | Type          | Town         | Ownership  | Latitude | Longitude | geometry                 |
|---|---------|------------------|-------------------------|---------------|--------------|------------|----------|-----------|--------------------------|
| 0 | Ashanti | Offinso North    | A.M.E Zion Clinic       | Clinic        | Afrancho     | CHAG       | 7.40801  | -1.96317  | POINT (-1.96317 7.40801) |
| 1 | Ashanti | Bekwai Municipal | Abenkyiman Clinic       | Clinic        | Anwiankwanta | Private    | 6.46312  | -1.58592  | POINT (-1.58592 6.46312) |
| 2 | Ashanti | Adansi North     | Aboabo Health Centre    | Health Centre | Aboabo No 2  | Government | 6.22393  | -1.34982  | POINT (-1.34982 6.22393) |
| 3 | Ashanti | Afigya-Kwabre    | Aboabogya Health Centre | Health Centre | Aboabogya    | Government | 6.84177  | -1.61098  | POINT (-1.61098 6.84177) |
| 4 | Ashanti | Kwabre           | Aboaso Health Centre    | Health Centre | Aboaso       | Government | 6.84177  | -1.61098  | POINT (-1.61098 6.84177) |

</div>


In the code cell above, to create a GeoDataFrame from a CSV file, we needed to use both Pandas and GeoPandas:

- We begin by creating a DataFrame containing columns with latitude and longitude coordinates.
- To convert it to a GeoDataFrame, we use <span style="font-family:monospace;">gpd.GeoDataFrame()</span>.
- The <span style="font-family:monospace;">gpd.points_from_xy()</span> function creates <span style="font-family:monospace;">Point</span> objects from the latitude and longitude columns.

### Re-projecting

Re-projecting refers to the process of changing the CRS. This is done in GeoPandas with the <span style="font-family:monospace;">to_crs()</span> method.

When plotting multiple GeoDataFrames, it's important that they all use the same CRS. In the code cell below, we change the CRS of the <span style="font-family:monospace;">facilities</span> GeoDataFrame to match the CRS of <span style="font-family:monospace;">regions</span> before plotting it.

```python
# Create a map
ax = regions.plot(figsize=(8,8), color='whitesmoke', linestyle=':', edgecolor='black')
facilities.to_crs(epsg=32630).plot(markersize=1, ax=ax)
```

<br>
<picture>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
[![png]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.png#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.png)
</picture>
{% assign counter = counter | plus: 1 %} 
<br>

The <span style="font-family:monospace;">to_crs()</span> method modifies only the "geometry" column: all other columns are left as-is.

```python
# The "Latitude" and "Longitude" columns are unchanged
facilities.to_crs(epsg=32630).head()
```

<div class="table-wrapper" markdown="block">

|   | Region  | District         | FacilityName            | Type          | Town         | Ownership  | Latitude | Longitude | geometry                      |
|---|---------|------------------|-------------------------|---------------|--------------|------------|----------|-----------|-------------------------------|
| 0 | Ashanti | Offinso North    | A.M.E Zion Clinic       | Clinic        | Afrancho     | CHAG       | 7.40801  | -1.96317  | POINT (614422.662 818986.851) |
| 1 | Ashanti | Bekwai Municipal | Abenkyiman Clinic       | Clinic        | Anwiankwanta | Private    | 6.46312  | -1.58592  | POINT (656373.863 714616.547) |
| 2 | Ashanti | Adansi North     | Aboabo Health Centre    | Health Centre | Aboabo No 2  | Government | 6.22393  | -1.34982  | POINT (682573.395 688243.477) |
| 3 | Ashanti | Afigya-Kwabre    | Aboabogya Health Centre | Health Centre | Aboabogya    | Government | 6.84177  | -1.61098  | POINT (653484.490 756478.812) |
| 4 | Ashanti | Kwabre           | Aboaso Health Centre    | Health Centre | Aboaso       | Government | 6.84177  | -1.61098  | POINT (653484.490 756478.812) |

</div>

In case the EPSG code is not available in GeoPandas, we can change the CRS with what's known as the "proj4 string" of the CRS. For instance, the proj4 string to convert to latitude/longitude coordinates is as follows:

    +proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs

```python
# Change the CRS to EPSG 4326
regions.to_crs("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs").head()
```

<div class="table-wrapper" markdown="block">

|   | Region        | geometry                                          |
|---|---------------|---------------------------------------------------|
| 0 | Ashanti       | POLYGON ((-1.30985 7.62302, -1.30786 7.62198, ... |
| 1 | Brong Ahafo   | POLYGON ((-2.54567 8.76089, -2.54473 8.76071, ... |
| 2 | Central       | POLYGON ((-2.06723 6.29473, -2.06658 6.29420, ... |
| 3 | Eastern       | POLYGON ((-0.21751 7.21009, -0.21747 7.20993, ... |
| 4 | Greater Accra | POLYGON ((0.23456 6.10986, 0.23484 6.10974, 0.... |

</div>

### Attributes of geometric objects

As you learned in the first tutorial, for an arbitrary GeoDataFrame, the type in the "geometry" column depends on what we are trying to show: for instance, we might use:

- A Point for the epicenter of an earthquake.
- A LineString for a street.
- A Polygon to show country boundaries.

All three types of geometric objects have built-in attributes that you can use to quickly analyze the dataset. For instance, you can get the x- and y-coordinates of a Point from the $x$ and $y$ attributes, respectively.

```python
# Get the x-coordinate of each point
facilities.geometry.head().x
```

    0   -1.96317
    1   -1.58592
    2   -1.34982
    3   -1.61098
    4   -1.61098
    dtype: float64


And, you can get the length of a LineString from the length attribute.

Or, you can get the area of a Polygon from the area attribute.

```python
# Calculate the area (in square meters) of each polygon in the GeoDataFrame 
regions.loc[:, "AREA"] = regions.geometry.area / 10**6

print("Area of Ghana: {} square kilometers".format(regions.AREA.sum()))
print("CRS:", regions.crs)
regions.head()
```

    Area of Ghana: 239584.5760055668 square kilometers
    CRS: epsg:32630

<div class="table-wrapper" markdown="block">

|   | Region        | geometry                                          | AREA         |
|---|---------------|---------------------------------------------------|--------------|
| 0 | Ashanti       | POLYGON ((686446.075 842986.894, 686666.193 84... | 24379.017777 |
| 1 | Brong Ahafo   | POLYGON ((549970.457 968447.094, 550073.003 96... | 40098.168231 |
| 2 | Central       | POLYGON ((603176.584 695877.238, 603248.424 69... | 9665.626760  |
| 3 | Eastern       | POLYGON ((807307.254 797910.553, 807311.908 79... | 18987.625847 |
| 4 | Greater Accra | POLYGON ((858081.638 676424.913, 858113.115 67... | 3706.511145  |

</div>


In the code cell above, since the CRS of the *regions* GeoDataFrame is set to EPSG 32630 (a "Mercator" projection), the area calculation is slightly less accurate than if we had used an equal-area projection like "Africa Albers Equal Area Conic".

But this yields the area of Ghana as approximately 239585 square kilometers, which is not too far off from the [correct](https://www.google.com/search?q=area+of+ghana) answer.


### Example

You are a bird conservation expert and want to understand migration patterns of purple martins. In your research, you discover that these birds typically spend the summer breeding season in the eastern United States, and then migrate to South America for the winter. But since this bird is under threat of endangerment, you'd like to take a closer look at the locations that these birds are more likely to visit.

<br>
<picture>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
[![png]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.png#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.png)
</picture>
{% assign counter = counter | plus: 1 %} 
<br>

There are several [protected areas](https://www.iucn.org/theme/protected-areas/about) in South America, which operate under special regulations to ensure that species that migrate (or live) there have the best opportunity to thrive. You'd like to know if purple martins tend to visit these areas. To answer this question, you'll use some recently collected data that tracks the year-round location of eleven different birds.

Before you get started, run the code cell below to set everything up.

```python
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString
```

Run the next code cell (without changes) to load the GPS data into a pandas DataFrame birds_df.

```python
# Load the data and print the first 5 rows
birds_df = pd.read_csv("../input/geospatial-learn-course-data/purple_martin.csv", parse_dates=['timestamp'])
print("There are {} different birds in the dataset.".format(birds_df["tag-local-identifier"].nunique()))
birds_df.head()
```

There are 11 birds in the dataset, where each bird is identified by a unique value in the "tag-local-identifier" column. Each bird has several measurements, collected at different times of the year.

Use the next code cell to create a GeoDataFrame birds.

- Birds should have all of the columns from <span style="font-family:monospace;">birds_df</span>, along with a "geometry" column that contains Point objects with (longitude, latitude) locations.
- Set the CRS of birds to <span style="font-family:monospace;">{'init': 'epsg:4326'}</span>.

```python
# Create the GeoDataFrame
birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df["location-long"], birds_df["location-lat"]))

# Set the CRS to {'init': 'epsg:4326'}
birds.crs = {'init' :'epsg:4326'}
```

### Plot the Data

Next, we load in the 'naturalearth_lowres' dataset from GeoPandas, and set americas to a GeoDataFrame containing the boundaries of all countries in the Americas (both North and South America). Run the next code cell without changes.

```python
# Load a GeoDataFrame with country boundaries in North/South America, print the first 5 rows
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
americas = world.loc[world['continent'].isin(['North America', 'South America'])]
americas.head()
```

<div class="table-wrapper" markdown="block">

|    |   pop_est |     continent |                     name | iso_a3 | gdp_md_est |                                          geometry |
|---:|----------:|--------------:|-------------------------:|-------:|-----------:|--------------------------------------------------:|
|  3 |  35623680 | North America |                   Canada |    CAN |  1674000.0 | MULTIPOLYGON (((-122.84000 49.00000, -122.9742... |
|  4 | 326625791 | North America | United States of America |    USA | 18560000.0 | MULTIPOLYGON (((-122.84000 49.00000, -120.0000... |
|  9 |  44293293 | South America |                Argentina |    ARG |   879400.0 | MULTIPOLYGON (((-68.63401 -52.63637, -68.25000... |
| 10 |  17789267 | South America |                    Chile |    CHL |   436100.0 | MULTIPOLYGON (((-68.63401 -52.63637, -68.63335... |
| 16 |  10646714 | North America |                    Haiti |    HTI |    19340.0 | POLYGON ((-71.71236 19.71446, -71.62487 19.169... |

</div>

Use the next code cell to create a single plot that shows both: (1) the country boundaries in the americas GeoDataFrame, and (2) all of the points in the <span style="font-family:monospace;">birds_gdf</span> GeoDataFrame.

Don't worry about any special styling here; just create a preliminary plot, as a quick sanity check that all of the data was loaded properly. In particular, you don't have to worry about color-coding the points to differentiate between birds, and you don't have to differentiate starting points from ending points. We'll do that in the next part of the exercise.

```python
# Your code here
ax = americas.plot(figsize=(10,10), color='white', linestyle=':', edgecolor='gray')
birds.plot(ax=ax, markersize=10)
```

<br>
<picture>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
[![png]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.png#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.png)
</picture>
{% assign counter = counter | plus: 1 %} 
<br>

### Where does each bird start and end its journey? (Part 1)

Now, we're ready to look more closely at each bird's path. Run the next code cell to create two GeoDataFrames:

- <span style="font-family:monospace;">path_gdf</span> contains LineString objects that show the path of each bird. It uses the <span style="font-family:monospace;">LineString()</span> method to create a LineString object from a list of Point objects.
- <span style="font-family:monospace;">start_gdf</span> contains the starting points for each bird.

```python
# GeoDataFrame showing path for each bird
path_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: LineString(x)).reset_index()
path_gdf = gpd.GeoDataFrame(path_df, geometry=path_df.geometry)
path_gdf.crs = {'init' :'epsg:4326'}

# GeoDataFrame showing starting point for each bird
start_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[0]).reset_index()
start_gdf = gpd.GeoDataFrame(start_df, geometry=start_df.geometry)
start_gdf.crs = {'init' :'epsg:4326'}

# Show first five rows of GeoDataFrame
start_gdf.head()
```

<div class="table-wrapper" markdown="block">

|   | tag-local-identifier |                   geometry |
|--:|---------------------:|---------------------------:|
| 0 |                30048 | POINT (-90.12992 20.73242) |
| 1 |                30054 | POINT (-93.60861 46.50563) |
| 2 |                30198 | POINT (-80.31036 25.92545) |
| 3 |                30263 | POINT (-76.78146 42.99209) |
| 4 |                30275 | POINT (-76.78213 42.99207) |

</div>

Use the next code cell to create a GeoDataFrame <span style="font-family:monospace;">end_gdf</span> containing the final location of each bird.

The format should be identical to that of <span style="font-family:monospace;">start_gdf</span>, with two columns ("tag-local-identifier" and "geometry"), where the "geometry" column contains Point objects.
Set the CRS of end_gdf to <span style="font-family:monospace;">{'init': 'epsg:4326'}</span>.

```python
end_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[-1]).reset_index()
end_gdf = gpd.GeoDataFrame(end_df, geometry=end_df.geometry)
end_gdf.crs = {'init': 'epsg:4326'}
```

###  Where does each bird start and end its journey? (Part 2)

Use the GeoDataFrames from the question above (<span style="font-family:monospace;">path_gdf</span>, <span style="font-family:monospace;">start_gdf</span>, and <span style="font-family:monospace;">end_gdf</span>) to visualize the paths of all birds on a single map. You may also want to use the americas GeoDataFrame.

```python
ax = americas.plot(figsize=(10, 10), color='white', linestyle=':', edgecolor='gray')

start_gdf.plot(ax=ax, color='red',  markersize=30)
path_gdf.plot(ax=ax, cmap='tab20b', linestyle='-', linewidth=1, zorder=1)
end_gdf.plot(ax=ax, color='black', markersize=30)
```

<br>
<picture>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
[![png]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.png#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.png)
</picture>
{% assign counter = counter | plus: 1 %} 
<br>

### Where are the protected areas in South America? (Part 1)

It looks like all of the birds end up somewhere in South America. But are they going to protected areas?

In the next code cell, you'll create a GeoDataFrame protected_areas containing the locations of all of the protected areas in South America. The corresponding shapefile is located at filepath protected_filepath.

```python
# Path of the shapefile to load
protected_filepath = "../input/geospatial-learn-course-data/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile-polygons.shp"

# Your code here
protected_areas = gpd.read_file(protected_filepath)
```

### Where are the protected areas in South America? (Part 2)

Create a plot that uses the protected_areas GeoDataFrame to show the locations of the protected areas in South America. (You'll notice that some protected areas are on land, while others are in marine waters.)

```python
# Country boundaries in South America
south_america = americas.loc[americas['continent']=='South America']

# Your code here: plot protected areas in South America
# Plot protected areas in South America
ax = south_america.plot(figsize=(10,10), color='white', edgecolor='gray')
protected_areas.plot(ax=ax, alpha=0.4)
```

### What percentage of South America is protected?

You're interested in determining what percentage of South America is protected, so that you know how much of South America is suitable for the birds.

As a first step, you calculate the total area of all protected lands in South America (not including marine area). To do this, you use the "REP_AREA" and "REP_M_AREA" columns, which contain the total area and total marine area, respectively, in square kilometers.

Run the code cell below without changes.

```python
P_Area = sum(protected_areas['REP_AREA']-protected_areas['REP_M_AREA'])
print("South America has {} square kilometers of protected areas.".format(P_Area))
```

Then, to finish the calculation, you'll use the south_america GeoDataFrame.

```python
south_america.head()
```

<div class="table-wrapper" markdown="block">

|    |   pop_est |     continent |         name | iso_a3 | gdp_md_est |                                          geometry |
|---:|----------:|--------------:|-------------:|-------:|-----------:|--------------------------------------------------:|
|  9 |  44293293 | South America |    Argentina |    ARG |   879400.0 | MULTIPOLYGON (((-68.63401 -52.63637, -68.25000... |
| 10 |  17789267 | South America |        Chile |    CHL |   436100.0 | MULTIPOLYGON (((-68.63401 -52.63637, -68.63335... |
| 20 |      2931 | South America | Falkland Is. |    FLK |      281.8 | POLYGON ((-61.20000 -51.85000, -60.00000 -51.2... |
| 28 |   3360148 | South America |      Uruguay |    URY |    73250.0 | POLYGON ((-57.62513 -30.21629, -56.97603 -30.1... |
| 29 | 207353391 | South America |       Brazil |    BRA |  3081000.0 | POLYGON ((-53.37366 -33.76838, -53.65054 -33.2... |

</div>

Calculate the total area of South America by following these steps:

- Calculate the area of each country using the area attribute of each polygon (with EPSG 3035 as the CRS), and add up the results. The calculated area will be in units of square meters.
- Convert your answer to have units of square kilometeters.

```python
# Your code here: Calculate the total area of South America (in square kilometers)
totalArea = sum(south_america.geometry.to_crs(epsg=3035).area) / 10**6
```

Run the code cell below to calculate the percentage of South America that is protected.

```python
# What percentage of South America is protected?
percentage_protected = P_Area/totalArea
print('Approximately {}% of South America is protected.'.format(round(percentage_protected*100, 2)))
```

    Approximately 30.39% of South America is protected.

### Where are the birds in South America?

So, are the birds in protected areas?

Create a plot that shows for all birds, all of the locations where they were discovered in South America. Also plot the locations of all protected areas in South America.

To exclude protected areas that are purely marine areas (with no land component), you can use the "MARINE" column (and plot only the rows in <span style="font-family:monospace;">protected_areas[protected_areas['MARINE']!='2']</span>, instead of every row in the <span style="font-family:monospace;">protected_areas</span> GeoDataFrame).

```python
ax = south_america.plot(figsize=(10,10), color='white', edgecolor='gray')
protected_areas[protected_areas['MARINE']!='2'].plot(ax=ax, alpha=0.4, zorder=1)
birds[birds.geometry.y < 0].plot(ax=ax, color='red', alpha=0.6, markersize=10, zorder=2)
```

<br>
<picture>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
[![png]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.png#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.png)
</picture>
{% assign counter = counter | plus: 1 %} 
<br>