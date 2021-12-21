---
layout: page
title: Spearnet Archive
permalink: /spearnet/
mainnav: false
datatable: true
---
{% include breadcrumbs.html %}

<div class="datatable-begin"></div>
<table class="display" style="font-size:8px">
  {% for row in site.data.spear_ttv %}
    {% if forloop.first %}
    <thead>
    <tr>
      {% for pair in row %}
        <th>{{ pair[0] }}</th>
      {% endfor %}
    </tr>
    </thead>
    {% endif %}

    {% tablerow pair in row %}
      {{ pair[1] }}
    {% endtablerow %}
  {% endfor %}
</table>
<div class="datatable-end"></div>
