---
layout: page
title: Firefly Targets
permalink: /tess-viable/
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
        dom: 'Bfrtip',
        buttons: [
           'copy', 'csv', 'excel',
        ],
        scrollX: 5,
        deferRender: true
    }
        );
    });
</script>

### NASA-TESS Viable

This table contains a list of all confirmed exoplanets which have been observed by TESS.

<table class="display" style="font-size:12px;">
  {% for row in site.data.nasa_tess_viable %}
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
