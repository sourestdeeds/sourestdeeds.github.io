---
layout: page
title: Spearnet Archive
permalink: /spearnet/
mainnav: false
datatable: true
---
{% include breadcrumbs.html %}


<table class="display" style="font-size:10px;">
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
