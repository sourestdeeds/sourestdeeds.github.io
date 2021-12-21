---
layout: page
title: Spearnet Archive
permalink: /spearnet/
mainnav: false
datatable: true
---
{% include breadcrumbs.html %}

<script>
$(document).ready(function(){

    $('table.display').DataTable( {
        paging: true,
        stateSave: true,
        searching: true,
        scrollX: 5
    }
        );
});
</script>

### Coupled 

<table class="display" style="font-size:12px;">
  {% for row in site.data.spear_coupled %}
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

### Uncoupled

<table class="display" style="font-size:12px;">
  {% for row in site.data.spear_uncoupled %}
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

### Coupled TTV

<table class="display" style="font-size:12px;">
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



