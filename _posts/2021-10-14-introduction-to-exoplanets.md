---
title: 'Introduction to Exoplanets'
tags: [exoplanet, TESS, astrophysics]
layout: post
mathjax: true
categories: Astrophysics
permalink: /blog/:title/
---
{% assign counter = 1 %}
{% assign link = "https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/" %}
{% assign date = page.date | date: "%Y-%m-%d" %}
{% assign filename = page.title | remove: " -" | replace: " ", "-" | downcase %}

Historical records have demonstrated that the classification of the uniqueness of Earth and and the composition of the solar system were controversial. In the 3rd century BC, Epicurus (341-270 B.C.) stated that 

> “There are infinite worlds both like and unlike this world of ours. For the atoms being infinite in number, as was already proven, (...) there nowhere exists an obstacle to the infinite number of worlds.”

This idea was not shared by Aristotle (384-322 B.C.) who said that 

> “There cannot be more worlds than one.”

In 1609, Gallileo Galilei (1564-1642) first observed other planets within our solar system. Copernicus (1473-1543) later confirmed that planetary motion is confined by its orbit around a central star.

Unlike the solar system discoveries, exoplanets are difficult to observe directly. The light reflected from a planet is much fainter than that of its host. When observed from large distances, the exoplanet appears in close proximity to its host, and its light, already faint, is diluted by the glare from its host. Exoplanets induce influence on their parent stars by which various methods of detection were conceived, and avoid the difficulties associated with contrast.

The first confirmed detections were made by Wolszczan and Frail (1992), who monitored perturbations in pulsar timing variations. PSR 1257+12 b and c are \\( \simeq 2-3 M_{Earth} \\) and orbit at a similar distance to Mercury. The first discovery of an exoplanet around a sun-like analogue was made by monitoring variations in the Radial Velocity of a star (Mayor and Queloz, 1995). 51 Pegasi b is a hot Jupiter \\( \simeq \tfrac{1}{2} M_{Jupiter} \\) and orbits very close to its parent, \\( \simeq 8 \\) times closer than Mercury. These initial discoveries (Mayor and Queloz, 1995; [Marcy and Butler, 1996](https://iopscience.iop.org/article/10.1086/310096)) highly challenged our understanding of planetary formation, and even recent discoveries require proposed changes and additions to existing models. Gas giants were thought to form beyond the “snow line”, which suggested that they were expected to be found in a similar orbital distance to Jupiter ([Pollack, 1996](https://dx.doi.org/doi.org/10.1006/icar.1996.0190)). This prompted suggestions of inward planetary migration after formation, caused by the interaction with host disks (Lin et al., 1996).

In recent events, further challenges to accepted theory have profoundly altered our perception of ancient planet formation. A 10 Billion year old super earth was confirmed by the Transiting Exoplanet Survey Satellite, suggesting the possibility of habitable worlds shortly after the milky way formed. Such a system, twice as old as ours, demonstrates that rocky planets have existed for the majority of the universes lifetime. TOI-561 ([Lacedelli et al., 2020](https://academic.oup.com/mnras/article/501/3/4148/6027695); [Weiss et al., 2021](https://iopscience.iop.org/article/10.3847/1538-3881/abd409)) hosts 3 confirmed planets so far, in a region called the galactic thick disk which hosts stars with low metallicity. Such regions were thought to be incapable of hosting planets.

Currently, exoplanet discovery boasts the lofty figure of 4364 confirmed planets listed in the NASA exoplanet archive ([Akeson et al., 2013](https://iopscience.iop.org/article/10.1086/672273)) as of March 2021. The progress since 1995 is driven by improvements in instrumentation and observing techniques such as CCD’s, high-resolution spectroscopy, computer based image processing, and diverse ranges of exoplanet detection methods.
<br>
<br>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
<center><b>Figure 1:</b> A TESS exposure demonstating the field of view.</center>
{% assign counter = counter | plus: 1 %} 

### References

- Wolszczan, A and D A Frail (1992) “A planetary system around the millisecond pulsar PSR1257+12,” *Nature*, 359, 710–713.
- Mayor, Michel and Didier Queloz (1995) “A Jupiter-mass companion to a solar-type star,” *Nature*, 378, 703–706.
- Marcy, Geoffrey W. and R. Paul Butler (1996) “A Planetary Companion to 70 Virginis,” *The Astrophysical Journal*, 464 (2), L147–L151, [10.1086/310096](https://iopscience.iop.org/article/10.1086/310096).
- Pollack, J B (1996) “Formation of the Giant Planets by Concurrent Accretion of Solids and Gas,” *Icarus*, 124 (1), 62–85, [doi.org/10.1006/icar.1996.0190](https://dx.doi.org/doi.org/10.1006/icar.1996.0190).
- Lin, D N C, P Bodenheimer, and D C Richardson (1996) “Orbital migration of the planetary companion of 51 Pegasi to its present location,” 380 (April), 606–607.
- Lacedelli, Gaia, L. Malavolta, L. Borsato et al. (2020) “An unusually low density ultra- short period super-Earth and three mini-Neptunes around the old star TOI-561,” *arXiv*, 20 (November), 1–20, [10.1093/mnras/staa3728](https://academic.oup.com/mnras/article/501/3/4148/6027695).
- Weiss, Lauren M., Fei Dai, Daniel Huber et al. (2021) “The TESS-Keck Survey. II. An Ultra-short-period Rocky Planet and Its Siblings Transiting the Galactic Thick-disk Star TOI-561,” *The Astronomical Journal*, 161 (2), 56, [10.38471538-3881/abd409](https://iopscience.iop.org/article/10.3847/1538-3881/abd409).
- Akeson, R. L., X. Chen, D. Ciardi et al. (2013) “The NASA Exoplanet Archive: Data and Tools for Exoplanet Research,” *Publications of the Astronomical Society of the Pacific*, 125 (930), 989–999, [10.1086/672273](https://iopscience.iop.org/article/10.1086/672273).