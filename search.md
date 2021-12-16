---
layout: page
mainnav: true
title: Search
permalink: /search/
active: search
---
{% include breadcrumbs.html %}

<ul class="nav navbar-nav">
        {% if page.active == "about" %}
            <li class="active"><a href="#">About</a></li>
        {% else %}
            <li><a href="/">About</a></li>
        {% endif %}
        {% if page.active == "photography" %}
            <li class="active"><a href="#">Photography</a></li>
        {% else %}
            <li><a href="/">Photography</a></li>
        {% endif %}
        {% if page.active == "search" %}
            <li class="active"><a href="#">Search</a></li>
        {% else %}
            <li><a href="/about">Search</a></li>
        {% endif %}
</ul>

{% include search-lunr.html %}
<script src="/js/text-glitch.js"></script>