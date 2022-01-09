---
title: 'Discover Pi Mensae'
tags: [jupyter, project, pi mensae, TESS, astrophysics]
layout: post
mathjax: true
categories: Astrophysics
permalink: /blog/:title/
---


Data from the TESS mission are [available from the data archive at MAST](https://archive.stsci.edu/prepds/tess-data-alerts/). This tutorial demonstrates how the [Lightkurve Python package](http://lightkurve.keplerscience.org) can be used to read in these data and create your own TESS light curves with different aperture masks.

Below is a quick tutorial on how to get started using *Lightkurve* and TESS data. We'll use the nearby, bright target Pi Mensae (ID 261136679), around which the mission team recently discovered a short period planet candidate on a 6.27 day orbit. See the [pre-print paper by Huang et al (2018)](https://arxiv.org/abs/1809.05967) for more details.

TESS data is stored in a binary file format which is documented in the [TESS Science Data Products Description Document](https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf). *Lightkurve* provides a [TessTargetPixelFile](https://docs.lightkurve.org/reference/api/lightkurve.TessTargetPixelFile.html?highlight=tesstargetpixelfile) class which allows you to interact with the data easily.




```python
import lightkurve as lk
```


```python
search_result = lk.search_targetpixelfile('Pi Mensae', mission='TESS', sector=1)
```


```python
search_result
```




SearchResult containing 2 data products.

<table id="table140206256874304">
<thead><tr><th>#</th><th>mission</th><th>year</th><th>author</th><th>exptime</th><th>target_name</th><th>distance</th></tr></thead>
<thead><tr><th></th><th></th><th></th><th></th><th>s</th><th></th><th>arcsec</th></tr></thead>
<tr><td>0</td><td>TESS Sector 01</td><td>2018</td><td><a href='https://heasarc.gsfc.nasa.gov/docs/tess/pipeline.html'>SPOC</a></td><td>120</td><td>261136679</td><td>0.0</td></tr>
<tr><td>1</td><td>TESS Sector 01</td><td>2018</td><td><a href='https://archive.stsci.edu/hlsp/tess-spoc'>TESS-SPOC</a></td><td>1800</td><td>261136679</td><td>0.0</td></tr>
</table>




```python
tpf = search_result.download(quality_bitmask='default')
```



```python
tpf
```




    TessTargetPixelFile(TICID: 261136679)



*TessTargetPixelFile*'s have many helpful methods and attributes. For example, you can access basic meta data on the target easily:


```python
tpf.mission
```




    'TESS'




```python
tpf.targetid  # TESS Input Catalog (TIC) Identifier
```




    261136679




```python
tpf.sector  # TESS Observation Sector
```




    1




```python
tpf.camera  # TESS Camera Number
```




    4




```python
tpf.ccd  # TESS CCD Number
```




    2



We might want to plot the data, we can do this with the **plot()** method. You can add the keyword *aperture_mask* to plot an aperture on top of the image. In this case we've used the *pipeline_mask* which is stored in the original .fits file, but you can use any aperture you like.


```python
%matplotlib inline
tpf.plot(aperture_mask=tpf.pipeline_mask);
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_15_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_15_0.png)
    


If you want to access the original fits file that generated the data you can use the *hdu* attribute of the tpf. This will return an *astropy.io.fits* object, for example


```python
tpf.hdu
```




    [<astropy.io.fits.hdu.image.PrimaryHDU object at 0x7f848032e9d0>, <astropy.io.fits.hdu.table.BinTableHDU object at 0x7f848032ea00>, <astropy.io.fits.hdu.image.ImageHDU object at 0x7f845023c5e0>, <astropy.io.fits.hdu.table.BinTableHDU object at 0x7f845023c9d0>]



You can access each extension and the data inside it in the same way you'd use [astropy.io.fits](https://docs.astropy.org/en/stable/io/fits/). If you want to access data held in the TPF, such as the time of the observations, you can do that easily by using


```python
tpf.time
```




    <Time object: scale='tdb' format='btjd' value=[1325.2969605  1325.29834936 1325.29973823 ... 1353.17428819 1353.17567704
     1353.1770659 ]>



This returns the time in units of days counted since [Julian Day](https://en.wikipedia.org/wiki/Julian_day) 2457000.  

You can access the corresponding flux values using


```python
tpf.flux
```



Flux is a *numpy.ndarray* with a shape of (TIME x PIXELS x PIXELS). If you want to access just the first frame you can use


```python
tpf.flux[0]
```


These values are in units electrons per second.

## Building Light Curves from TPFs

We can use the [to_lightcurve()](https://docs.lightkurve.org/reference/api/lightkurve.KeplerTargetPixelFile.to_lightcurve.html?highlight=to_lightcurve) method to turn this TPF into a light curve using *Simple Aperture Photometry*. This will put an aperture on the target, and sum up the flux in all the pixels inside the aperture. 

The default for **to_lightcurve()** is to use the mask generated by the TESS pipeline.


```python
lc = tpf.to_lightcurve()
```

Now we can use the plot function to take a look at the data.


```python
lc.errorbar();
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_28_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_28_0.png)
    


This looks pretty good, but maybe we can improve things by creating a new aperture.


```python
aperture_mask = tpf.create_threshold_mask(threshold=10)

# Plot that aperture
tpf.plot(aperture_mask=aperture_mask);
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_30_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_30_0.png)
    



```python
lc = tpf.to_lightcurve(aperture_mask=aperture_mask)
```


```python
lc.errorbar();
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_32_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_32_0.png)
    


There's a long term trend in this dataset, which we can remove with a simple smoothing filter. You can use the [lc.flatten()](https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.flatten.html?highlight=flatten#lightkurve.LightCurve.flatten) method to apply and divide the [Savitzky-Golay smoothing filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter). Here we'll use a *window_length* of 1001 cadences, which is roughly a 5% of the full length of the light curve. 


```python
# Number of cadences in the full light curve
print(lc.time.shape)
```

    (18279,)



```python
flat_lc = lc.flatten(window_length=1001)
flat_lc.errorbar();
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_35_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_35_0.png)
    


The light curve looks much flatter. Unfortunately there is a portion of the light curve that is very noisy, due to a jitter in the TESS spacecraft. We can remove this simply by masking the light curve. First we'll select the times that had the jitter.


```python
# Flag the times that are good quality
mask = (flat_lc.time.value < 1346) | (flat_lc.time.value > 1350)
```

Then we can just clip those times out.


```python
masked_lc = flat_lc[mask]
masked_lc.errorbar();
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_39_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_39_0.png)
    


We can use *Lightkurve* to plot these two light curves over each other to see the difference. 


```python
# First define the `matplotlib.pyplot.axes`
ax = flat_lc.errorbar()

# Pass that axis to the next plot
masked_lc.errorbar(ax=ax, label='masked');
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_41_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_41_0.png)
    


This looks much better. Now we might want to clip out some outliers from the light curve. We can do that with a simple lightkurve function [remove_outliers()](https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.remove_outliers.html?highlight=remove_outliers).


```python
clipped_lc = masked_lc.remove_outliers(sigma=6)
clipped_lc.errorbar();
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_43_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_43_0.png)
    


It's a little hard to see these data because of the plotting style. Let's use a scatter plot instead. We can do this with the [lc.scatter()](https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.scatter.html?highlight=scatter#lightkurve.LightCurve.scatter) method. This method works in the same way that [matplotlib.pyplot.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html) works, and takes in the same keyword arguments.


```python
clipped_lc.scatter(s=0.1);
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_45_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_45_0.png)
    


We can also add errorbars using the [lc.errorbar()](https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.errorbar.html?highlight=errorbar#lightkurve.LightCurve.errorbar) method.


```python
ax = clipped_lc.scatter(s=0.1)
clipped_lc.errorbar(ax=ax, alpha=0.2);  # alpha determines the transparency
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_47_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_47_0.png)
    


Finally let's use *lightkurve* to fold the data at the exoplanet orbital period and see if we can see the transit.


```python
folded_lc = clipped_lc.fold(period=6.27, epoch_time=1325.504)
folded_lc.errorbar();
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_49_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_49_0.png)
    


It looks like there's something there, but it's hard to see. Let's bin the light curve to reduce the number of points, but also reduce the uncertainty of those points.


```python
import astropy.units as u
binned_lc = folded_lc.bin(time_bin_size=5*u.minute)
binned_lc.errorbar();
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_51_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_51_0.png)
    


And now we can see the transit of Pi Men c! 

Note that you can actually do all these steps in just a few lines:


```python
lc = tpf.to_lightcurve(aperture_mask=aperture_mask).flatten(window_length=1001)
lc = lc[(lc.time.value < 1346) | (lc.time.value > 1350)]
lc.remove_outliers(sigma=6).fold(period=6.27, epoch_time=1325.504).bin(time_bin_size=5*u.minute).errorbar();
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_53_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_53_0.png)
    


## Comparing two apertures

In the above tutorial we used our own aperture instead of the pipeline aperture. Let's compare the results from using these two different apertures.


```python
# Use the default
lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask).flatten(window_length=1001)
lc = lc[(lc.time.value < 1346) | (lc.time.value > 1350)].remove_outliers(6).fold(period=6.27, epoch_time=1325.504).bin(5*u.minute)

# Use a custom aperture
custom_lc = tpf.to_lightcurve(aperture_mask=aperture_mask).flatten(window_length=1001)
custom_lc = custom_lc[(custom_lc.time.value < 1346) | (custom_lc.time.value > 1350)].remove_outliers(6).fold(period=6.27, epoch_time=1325.504).bin(5*u.minute)
```


```python
ax = lc.errorbar(label='Default aperture')
custom_lc.errorbar(ax=ax, label='Custom aperture');
```


    
[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_56_0.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-11-10-exoplanets-recover-first-tess-candidate/output_56_0.png)
    


The importance of using different aperture masks is clearly visible in the figure above.  Note however that the data archive at MAST also contains lightcurve products which have more advanced systematics removal methods applied.  We will explore those in a future tutorial!
