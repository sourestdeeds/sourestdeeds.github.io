---
# {% include_relative _posts/2021-11-26-library-catalog-subject.md %}
# {% include image.html url="/images/my-cat.jpg" description="My cat, Robert Downey Jr." %}
# {% include follow-buttons.html %}
# {% include search-lunr.html %}
# ![png](https://raw.githubusercontent.com/sourestdeeds/firefly/main/firefly/data/WASP-100%20b%20density.png)
# By default, content added below the "---" mark will appear in the home page
# between the top bar and the list of recent posts.
# To change the home page layout, edit the _layouts/home.html file.
# See: https://jekyllrb.com/docs/themes/#overriding-theme-defaults
#
layout: home
mainnav: false
accordion: 
  - title: <svg xmlns="http://www.w3.org/2000/svg" style="justify-content:center;" width="24" height="24" viewBox="0 0 24 24"><path d="M19.957 4.035c-.345-.024-.682-.035-1.012-.035-7.167 0-11.248 5.464-12.732 9.861l3.939 3.938c4.524-1.619 9.848-5.549 9.848-12.639 0-.367-.014-.741-.043-1.125zm-9.398 11.815l-2.402-2.402c1.018-2.383 3.91-7.455 10.166-7.767-.21 4.812-3.368 8.276-7.764 10.169zm4.559 1.282c-.456.311-.908.592-1.356.842-.156.742-.552 1.535-1.126 2.21-.001-.48-.135-.964-.369-1.449-.413.187-.805.348-1.179.49.551 1.424-.01 2.476-.763 3.462 1.08-.081 2.214-.61 3.106-1.504.965-.962 1.64-2.352 1.687-4.051zm-9.849-5.392c-.482-.232-.965-.364-1.443-.367.669-.567 1.453-.961 2.188-1.121.262-.461.556-.915.865-1.361-1.699.046-3.09.723-4.054 1.686-.893.893-1.421 2.027-1.503 3.106.986-.753 2.039-1.313 3.463-.762.145-.391.305-.785.484-1.181zm6.448.553c-.326-.325-.326-.853 0-1.178.325-.326.853-.326 1.178 0 .326.326.326.854 0 1.179-.326.325-.853.325-1.178-.001zm4.124-4.125c-.65-.65-1.706-.65-2.356 0-.651.651-.651 1.707 0 2.357.65.651 1.706.651 2.357 0 .65-.65.65-1.706-.001-2.357zm-1.591 1.592c-.228-.228-.228-.598 0-.825.227-.228.598-.228.826 0 .227.227.226.597 0 .825-.228.227-.598.227-.826 0zm-12.609 10.555l-.755-.755 4.341-4.323.755.755-4.341 4.323zm4.148 1.547l-.755-.755 3.03-3.054.756.755-3.031 3.054zm-5.034 2.138l-.755-.755 5.373-5.364.756.755-5.374 5.364zm21.083-14.291c-.188.618-.673 1.102-1.291 1.291.618.188 1.103.672 1.291 1.291.189-.619.673-1.103 1.291-1.291-.618-.188-1.102-.672-1.291-1.291zm-14.655-6.504c-.247.81-.881 1.443-1.69 1.69.81.247 1.443.881 1.69 1.69.248-.809.881-1.443 1.69-1.69-.81-.247-1.442-.88-1.69-1.69zm-1.827-3.205c-.199.649-.706 1.157-1.356 1.355.65.199 1.157.707 1.356 1.355.198-.649.706-1.157 1.354-1.355-.648-.198-1.155-.706-1.354-1.355zm5.387 0c-.316 1.035-1.127 1.846-2.163 2.163 1.036.316 1.847 1.126 2.163 2.163.316-1.036 1.127-1.846 2.162-2.163-1.035-.317-1.845-1.128-2.162-2.163zm11.095 13.64c-.316 1.036-1.127 1.846-2.163 2.163 1.036.316 1.847 1.162 2.163 2.197.316-1.036 1.127-1.881 2.162-2.197-1.035-.317-1.846-1.127-2.162-2.163z"/></svg> Firefly
    content: <a href="https://github.com/sourestdeeds/firefly">Firefly</a> is a self-contained python EDA pipeline which uses <a href="https://github.com/joshjchayes/TransitFit">TransitFit</a> to fit [TESS](https://youtu.be/Q4KjvPIbgMI){:.lightbox} [lightcurves](https://youtu.be/vLh9KWns9gE){:.lightbox} (time-series data), capable of fully automating the data retrieval required. The lightcurve below represents 206 transits phase folded around the transit midpoint (bottom of the bucket!) for WASP-100 b. <br><br> TransitFit (<a href="https://arxiv.org/abs/2103.12139">Hayes et al., 2021</a>) is capable of using information about the host and planet parameters, alongside the observation filters to couple stellar [limb-darkening](https://www.youtube.com/watch?v=ur0fATmsVoc&ab_channel=minutephysics){:.lightbox} coefficients across wavelengths. It was primarily designed for use with transmission spectroscopy studies, and employs transit observations at various wavelengths from different telescopes to simultaneously fit transit parameters using [nested sampling](https://github.com/joshspeagle/dynesty) retrieval. <br><br> [![png](https://raw.githubusercontent.com/sourestdeeds/firefly/main/firefly/data/WASP-100%20b%20density.png)](https://raw.githubusercontent.com/sourestdeeds/firefly/main/firefly/data/WASP-100%20b%20density.png)<center><b>Figure 1:</b> The observed transits of WASP-100 b phase folded.</center> <br> [The Transiting Exoplanet Survey Satellite](https://youtu.be/k_wmsk2OyuY){:.lightbox} (<a href="https://www.spiedigitallibrary.org/journals/Journal-of-Astronomical-Telescopes-Instruments-and-Systems/volume-1/issue-01/014003/Transiting-Exoplanet-Survey-Satellite/10.1117/1.JATIS.1.1.014003.full?SSO=1">Ricker et al., 2014</a>) (TESS) is an all-sky transit survey, whose primary goal is to detect Earth-sized planets orbiting bright stars, allowing follow-up observations to determine planet masses and atmospheric compositions. TESS has an 85% sky coverage, of which each sector is continuously observed for 4 weeks. For higher ecliptic lattitudes, the sectors overlap creating photometric time series for durations up to a year. The upper and lower ecliptic poles are called the [continuous viewing zones](https://tess.mit.edu/wp-content/uploads/sky_coverage.png) (CVZ), and are constantly observed in a yearly rotation between the two poles regardless of sector. Such multi-sector photometry allows for a steady stream of transits to be observed, which lends itself well to probe for [transit timing variations](https://www.youtube.com/watch?v=rqQ1xKsNIQE&ab_channel=NASAVideo){:.lightbox} (TTV’s). Increasing the accuracy of known parameters through the use of lightcurve fitting programs also benefits from a consistent single source of observations, as the systematic variance between sectors is minimal. TESS aims for 50 ppm photometric precision on stars with a TESS magnitude of 9-15. <br><br> [![gif](https://sourestdeeds.github.io/photos/TESS_Staring_Contest.gif)](https://sourestdeeds.github.io/photos/TESS_Staring_Contest.gif)<br><br><center><b>Figure 2:</b> TESS capturing the flicker of a star indicating a transit.</center>
  - title: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M11 6.999c2.395.731 4.27 2.607 4.999 5.001.733-2.395 2.608-4.269 5.001-5-2.393-.731-4.268-2.605-5.001-5-.729 2.394-2.604 4.268-4.999 4.999zm7 7c1.437.438 2.562 1.564 2.999 3.001.44-1.437 1.565-2.562 3.001-3-1.436-.439-2.561-1.563-3.001-3-.437 1.436-1.562 2.561-2.999 2.999zm-6 5.501c1.198.365 2.135 1.303 2.499 2.5.366-1.198 1.304-2.135 2.501-2.5-1.197-.366-2.134-1.302-2.501-2.5-.364 1.197-1.301 2.134-2.499 2.5zm-6-8.272c.522.658 1.118 1.253 1.775 1.775-.657.522-1.252 1.117-1.773 1.774-.522-.658-1.118-1.253-1.776-1.776.658-.521 1.252-1.116 1.774-1.773zm-.001-4.228c-.875 2.873-3.128 5.125-5.999 6.001 2.876.88 5.124 3.128 6.004 6.004.875-2.874 3.128-5.124 5.996-6.004-2.868-.874-5.121-3.127-6.001-6.001z"/></svg> Serenity
    content: Classified. 
---

> “If it ain't broke, fix it until it is.”

{% include accordion.html %}

<script
  type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.3.1/jquery.min.js"
  src = "https://cdnjs.cloudflare.com/ajax/libs/lunr.js/2.3.9/lunr.min.js"
></script>

<input
  placeholder="Search&hellip;"
  type="search"
  id="search"
  class="search-input"
/>
<div id="results" class="all-posts results"></div>