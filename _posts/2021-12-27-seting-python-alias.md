---
title: 'Python Setup'
tags: [kaggle, default python]
layout: post
mathjax: true
categories: [Python Snippets]
---
{% assign counter = 1 %}
{% assign link = "https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/" %}
{% assign date = page.date | date: "%Y-%m-%d" %}
{% assign filename = page.title | remove: " -" | replace: " ", "-" | downcase %}

First, look where it is installed:

    ls -l /usr/local/bin/python*

note the line which ends with python3.9 without anything (as m for example) and type

    ln -s -f /usr/local/bin/python3.9 /usr/local/bin/python

where '/usr/local/bin/python3.9' is what you have copied from above.

type in a new session

    python --version


Download pip by running the following command:

    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

The [curl command](https://phoenixnap.com/kb/curl-command) allows you to specify a direct download link. Use the -o option to set the name of the downloaded file.
Install the downloaded package by running:

    python get-pip.py