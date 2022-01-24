---
title: 'Model Cards'
tags: [kaggle, AI ethics]
layout: post
mathjax: true
categories: [AI Ethics]
permalink: /blog/:title/
---
{% assign counter = 1 %}
{% assign counter2 = 1 %}
{% assign link = "https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/" %}
{% assign date = page.date | date: "%Y-%m-%d" %}
{% assign filename = page.title | remove: " -" | replace: " ", "-" | downcase %}

A **model card** is a short document that provides key information about a machine learning model. Model cards increase transparency by communicating information about trained models to broad audiences.

<br>
[![jpeg]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.jpeg#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.jpeg)
{% assign counter = counter | plus: 1 %} 
<br>

### Model cards

Though AI systems are playing increasingly important roles in every industry, few people understand how these systems work. AI researchers are exploring many ways to communicate key information about models to inform people who use AI systems, people who are affected by AI systems and others.

Model cards - introduced in a [2019 paper](https://arxiv.org/abs/1810.03993) - are one way for teams to communicate key information about their AI system to a broad audience. This information generally includes intended uses for the model, how the model works, and how the model performs in different situations.

You can think of model cards as similar to the nutritional labels that you find on packaged foods.

### Examples of model cards

Before we continue, it might be useful to briefly skim some examples of model cards.

- [Salesforce's model cards](https://blog.einstein.ai/model-cards-for-ai-model-transparency/)
- [Open AI’s model card for GPT-3](https://github.com/openai/gpt-3/blob/master/model-card.md)
- [Google Cloud's example model cards](https://modelcards.withgoogle.com/face-detection)

### Who is the audience of your model card?

A model card should strike a balance between being easy-to-understand and communicating important technical information. When writing a model card, you should consider your audience: the groups of people who are most likely to read your model card. These groups will vary according to the AI system’s purpose.

For example, a model card for an AI system that helps medical professionals interpret x-rays to better diagnose musculoskeletal injuries is likely to be read by medical professionals, scientists, patients, researchers, policymakers and developers of similar AI systems. The model card may therefore assume some knowledge of health care and of AI systems.

### What sections should a model card contain?

Per the original paper, a model card should have the following nine sections. Note that different organizations may add, subtract or rearrange model card sections according to their needs (and you may have noticed this in some of the examples above).

As you read about the different sections, you're encouraged to review the two example model cards from the original paper. Before proceeding, open each of these model card examples in a new window:

- [Model Card - Smiling Detection in Images](https://github.com/Kaggle/learntools/blob/master/notebooks/ethics/pdfs/smiling_in_images_model_card.pdf)
- [Model Card - Toxicity in Text](https://github.com/Kaggle/learntools/blob/master/notebooks/ethics/pdfs/toxicity_in_text_model_card.pdf)


### 1. Model Details

- Include background information, such as developer and model version.

### 2. Intended Use

- What use cases are in scope?
    - Who are your intended users?
    - What use cases are out of scope?

### 3. Factors

- What factors affect the impact of the model? For example, the smiling detection model's results vary by demographic factors like age, gender or ethnicity, environmental factors like lighting or rain and instrumentation like camera type.

### 4. Metrics

What metrics are you using to measure the performance of the model? Why did you pick those metrics?

- For **classification systems** – in which the output is a class label – potential error types include false positive rate, false negative rate, false discovery rate, and false omission rate. The relative importance of each of these depends on the use case.
- For **score-based analyses** – in which the output is a score or price – consider reporting model performance across groups.

### 5. Evaluation Data

- Which datasets did you use to evaluate model performance? Provide the datasets if you can.
- Why did you choose these datasets for evaluation?
- Are the datasets representative of typical use cases, anticipated test cases and/or challenging cases?

### 6. Training Data

- Which data was the model trained on?

### 7. Quantitative Analyses

- How did the model perform on the metrics you chose? Break down performance by important factors and their intersections. For example, in the smiling detection example, performance is broken down by age (eg, young, old), gender (eg, female, male), and then both (eg, old-female, old-male, young-female, young-male).

### 8. Ethical Considerations

- Describe ethical considerations related to the model, such as sensitive data used to train the model, whether the model has implications for human life, health, or safety, how risk was mitigated, and what harms may be present in model usage.

### 9. Caveats and Recommendations

- Add anything important that you have not covered elsewhere in the model card.

### How can you use model cards in your organization?

The use of detailed model cards can often be challenging because an organization may not want to reveal its processes, proprietary data or trade secrets. In such cases, the developer team should think about how model cards can be useful and empowering, without including sensitive information.

Some teams use other formats - such as [FactSheets](https://aifs360.mybluemix.net/) - to collect and log ML model information.

### Scenario A

You are the creator of the S*imple Zoom* video editing tool, which uses AI to automatically zoom the video camera in on a presenter as they walk across a room during a presentation. You are launching the Simple Zoom tool and releasing a model card along with the tool, in the interest of transparency.

### 1) Audience

Which audiences should you write the model card for? 

> Model cards should be written for the groups that are most likely to read it. For Simple Zoom, such groups probably include people using the tool to record videos, organizations seeking to adopt the tool, IT and audio-visual teams and agencies, computer vision researchers, policymakers and developers of similar AI systems. Given how broad this group is, your team can only assume a basic knowledge of video recording terms throughout the model card.

### Scenario B

You are the product manager for *Presenter Pro*, a popular video and audio recording product for people delivering talks and presentations. As a new feature based on customer demand, your team has been planning to add the AI-powered ability for a single video camera to automatically track a presenter, focusing on them as they walk across the room or stage, zooming in and out automatically and continuously adjusting lighting and focus within the video frame.

You are hoping to incorporate a different company’s AI tool (called Simple Zoom) into your product (Presenter Pro). To determine whether *Simple Zoom* is a good fit for Presenter Pro, you are reviewing Simple Zoom’s model card.

### 2) Intended Use

The **Intended Use** section of the model card includes the following bullets:

- *Simple Zoom* is intended to be used for automatic zoom, focus, and lighting adjustment in the real-time video recording of individual presenters by a single camera
- *Simple Zoom* is not suitable for presentations in which there is more than one presenter or for presentations in which the presenter is partially or fully hidden at any time

As a member of the team evaluating Simple Zoom for potential integration into Presenter Pro, you are aware that Presenter Pro only supports one presenter per video.

However, you are also aware that in some Presenter Pro customers use large props in their presentation videos. Given the information in the Intended Use section of the model card, what problem do you foresee for these customers if you integrate Simple Zoom into Presenter Pro? What are some ways in which you could address this issue?

> Since Simple Zoom is not suitable for presentations in which the presenter is partially or fully hidden at any time, it might not work well in a presentation in which the presenter uses a large object, because the object could partially or fully hide the presenter. There are many potential ways to address this issue. For example, your team could reach out to the Simple Zoom team to assess the potential risks and harms of using Simple Zoom with props. As another example, your team could eventually add a message in the Presenter Pro user interface explaining that the Simple Zoom feature should not be used in presentations that use props.

### 3) Factors, Evaluation Data, Metrics, and Quantitative Analyses

We'll continue with **Scenario B**, where you are the product manager for Presenter Pro. Four more sections of the model card for Simple Zoom are described below.

**Factors**: The model card lists the following factors as potentially relevant to model performance:

- Group Factors
    - Self-reported gender
    - Skin tone
    - Self-reported age
- Other Factors
    - Camera angle
    - Presenter distance from camera
    - Camera type
    - Lighting

**Evaluation Data**: To generate the performance metrics reported in the **Quantitative Analysis** section (discussed below), the *Simple Zoom* team used an evaluation data set of 500 presentation videos, each between two and five minutes long. The videos included both regular and unusual presentation and recording scenarios and included presenters from various demographic backgrounds.

**Metrics**: Since *Simple Zoom* model performance is subjective (involving questions like whether a zoom is of appropriate speed or smoothness; or whether a lighting adjustment is well-executed), the Simple Zoom team tested the tool’s performance by asking a diverse viewer group to view _Simple Zoom_’s output videos (using the evaluation dataset’s 500 videos as inputs). Each viewer was asked to rate the quality of video editing for each video on a scale of 1 to 10, and each video’s average rating was used as a proxy for _Simple Zoom_’s performance on that video.

**Quantitative Analyses**: The quantitative analyses section of the model card includes a brief summary of performance results. According to the summary, the model generally performs equally well across all the listed demographic groups (gender, skin tone and age).

The quantitative analyses section also includes interactive graphs, which allow you to view the performance of the *Simple Zoom* tool by each factor and by intersections of ‘Group’ and ‘Other’ factors.

As a member of the team evaluating *Simple Zoom* for potential integration into *Presenter Pro*, what are some questions you might be interested in answering and exploring via the interactive graphs?

> There are many possible answers to this question. For example, you may want to check that the model’s equal performance across demographic groups (gender, skin tone and age) remains equal across different camera angles, distances from camera, camera types and lighting conditions. As another example, you may want to know how well the model performs from the specific camera angles that Production Pro customers most commonly use.