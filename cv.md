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
layout: page
mainnav: false
mathjax: false
permalink: /cv/
title: CV
---
{% include breadcrumbs.html %}


<div id="adobe-dc-view" style="width: 100%;"></div>
<script src="https://documentcloud.adobe.com/view-sdk/main.js"></script>
<script type="text/javascript">
	document.addEventListener("adobe_dc_view_sdk.ready", function(){ 
		var adobeDCView = new AdobeDC.View({clientId: "75c0126e67ed437d8268ece13f6e2b7f", divId: "adobe-dc-view"});
		adobeDCView.previewFile({
			content:{location: {url: "https://sourestdeeds.github.io/pdf/stephen-charles-cv.pdf"}},
			metaData:{fileName: "stephen-charles-cv.pdf"}
		}, {embedMode: "LIGHT_BOX", showPageControls:False, showPrintPDF:True, showDownloadPDF:True});
	});
</script>
