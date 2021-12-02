---
# {% include_relative _posts/2021-11-26-library-catalog-subject.md %}
# {% include image.html url="/images/my-cat.jpg" description="My cat, Robert Downey Jr." %}
# ![png](https://raw.githubusercontent.com/sourestdeeds/firefly/main/firefly/data/WASP-100%20b%20density.png)
# By default, content added below the "---" mark will appear in the home page
# between the top bar and the list of recent posts.
# To change the home page layout, edit the _layouts/home.html file.
# See: https://jekyllrb.com/docs/themes/#overriding-theme-defaults
#
layout: home
mainnav: false
accordion: 
  - title: Firefly 
    content: <a href="https://github.com/sourestdeeds/firefly">Firefly</a> is a self-contained python package for use with <a href="https://github.com/joshjchayes/TransitFit">TransitFit</a> to fit TESS lightcurves, capable of fully automating the data retrieval required. The lightcurve below represents 206 transits phase folded around the transit midpoint (bottom of the bucket!) for WASP-100 b. <br><br> TransitFit (<a href="https://arxiv.org/abs/2103.12139">Hayes et al., 2021</a>) is capable of using information about the host and planet parameters, alongside the observation filters to couple stellar limb-darkening coefficients across wavelengths. It was primarily designed for use with transmission spectroscopy studies, and employs transit observations at various wavelengths from different telescopes to simultaneously fit transit parameters using nested sampling retrieval. <br><br> [![png](https://raw.githubusercontent.com/sourestdeeds/firefly/main/firefly/data/WASP-100%20b%20density.png)](https://raw.githubusercontent.com/sourestdeeds/firefly/main/firefly/data/WASP-100%20b%20density.png)<center><b>Figure 1:</b> The observed transits of WASP-100 b phase folded.</center> <br> The Transiting Exoplanet Survey Satellite (<a href="https://www.spiedigitallibrary.org/journals/Journal-of-Astronomical-Telescopes-Instruments-and-Systems/volume-1/issue-01/014003/Transiting-Exoplanet-Survey-Satellite/10.1117/1.JATIS.1.1.014003.full?SSO=1">Ricker et al., 2014</a>) (TESS) is an all-sky transit survey, whose primary goal is to detect Earth-sized planets orbiting bright stars, allowing follow-up observations to determine planet masses and atmospheric compositions. TESS has an 85% sky coverage, of which each sector is continuously observed for 4 weeks. For higher ecliptic lattitudes, the sectors overlap creating photometric time series for durations up to a year. The upper and lower ecliptic poles are called the continuous viewing zones (CVZ), and are constantly observed in a yearly rotation between the two poles regardless of sector. Such multi-sector photometry allows for a steady stream of transits to be observed, which lends itself well to probe for transit timing variations (TTV’s). Increasing the accuracy of known parameters through the use of lightcurve fitting programs also benefits from a consistent single source of observations, as the systematic variance between sectors is minimal. TESS aims for 50 ppm photometric precision on stars with a TESS magnitude of 9-15. <br><br> [![gif](https://sourestdeeds.github.io/photos/TESS_Staring_Contest.gif)](https://sourestdeeds.github.io/photos/TESS_Staring_Contest.gif)<br><br><center><b>Figure 2:</b> TESS capturing the flicker of a star indicating a transit.</center>
  - title: Serenity
    content: Classified. 
---

> “Not everything that can be counted counts. Not everything that counts can be counted.”

{% include accordion.html %}
{% include search-lunr.html %}
