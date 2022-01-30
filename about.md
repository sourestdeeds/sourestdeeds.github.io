---
layout: page
mainnav: true
title: About
mathjax: false
permalink: /about/
---

<div id="adobe-dc-view" style="width: 100%;"></div>
<script src="https://documentcloud.adobe.com/view-sdk/main.js"></script>
<script type="text/javascript">
	document.addEventListener("adobe_dc_view_sdk.ready", function(){ 
		var adobeDCView = new AdobeDC.View({clientId: "75c0126e67ed437d8268ece13f6e2b7f", divId: "adobe-dc-view"});
		adobeDCView.previewFile({
			content:{location: {url: "https://sourestdeeds.github.io/pdf/stephen-charles-cv.pdf"}},
			metaData:{fileName: "stephen-charles-cv.pdf"}
		}, {embedMode: "IN_LINE"});
	});
</script>

### Certificates

{% include image-gallery-rect.html folder="certificates/datacamp" %}

{% include image-gallery-rect.html folder="certificates/kaggle" %}

### Published Papers

- Author:
- Co-Author:
	- [Hayes, J J C, E Kerins, J S Morgan et al. (2021)](https://arxiv.org/pdf/2103.12139.pdf) “TransitFit: an exoplanet transit fitting package
for multi-telescope datasets and its application to WASP-127 b, WASP-91 b, and WASP-126 b”, *arXiv*, 1–14.

### Journey

Before I started academic study I used to run a game server for *Ultima Online*, primarily written in <span style="font-family:monospace;">C++</span>. As such, the server was highly customisable, and with the source code from another server I painfully merged two incompatible SVN’s together to create one. Just by comparison alone I was able to get used to the logic and constructs that formed the language. Browsing through the many lines of code and merging was a very slow process, but it allowed me understand slightly different methods of doing things. It was almost like learning through reverse engineering. I like to follow the concept of learning by doing. Of note, I took interest in the AI constructs and pathing algorithms adopted, to try and create realistic npcs to fill the world.

My undergraduate studies primarily focused on *Mathematics* in its purest form, to which I supplemented this by taking extra modules from *Physics* to practice application. In my last year I took a formal <span style="font-family:monospace;">C++</span> course to take what I had learned from my game server days and apply it to what I had learned through undergraduate study. Suddenly very nasty equations were efficiently solved! I think data analysis in this aspect fascinates me. My dissertation project involved the inter-facial perturbations of a spherical bubble, to which an approximate formula was derived. As with most fluid dynamics problems, the solution requires numerical analysis. I was able to apply my previous knowledge and reverse engineering from before using the many libraries and pseudo code available.

My most recent foray into applying programming language is in the field of *Astrophysics*. This required extreme care in the handling and use of very large data sets, extracting the precise information needed and filtering such data through a pipeline. Specifically, I worked in exoplanet research, fitting [light curves](https://youtu.be/vLh9KWns9gE){:.lightbox} (time-series data) to improve on the data sets already proven to exist. The hope is that with a greater sensitivity, more interesting and varied data will provide sightings of [exomoons](https://youtu.be/3Ma1xLz1Asw){:.lightbox}. In this field I have written a a self-contained *python* EDA pipeline called [firefly](https://github.com/sourestdeeds/firefly), which uses a transit fitting program called [TransitFit](https://github.com/joshjchayes/TransitFit). Its capable of fitting [TESS](https://youtu.be/Q4KjvPIbgMI){:.lightbox} lightcurves with a [nested sampling](https://github.com/joshspeagle/dynesty) routine, using a *Bayesian* machine learning approach. In the future I hope to expand the functionality by allowing simultaneous fitting of multiple space based and ground based telescopes.

### Instagram

{% include instagram.html username="sourestdeeds" %}
