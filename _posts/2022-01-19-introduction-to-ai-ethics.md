---
title: 'Human-Centered Design for AI'
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

<br>
[![webp]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp#center)]({{ link }}{{ date }}-{{ filename }}/{{ counter }}.webp)
{% assign counter = counter | plus: 1 %} 
<br>


AI increasingly has an impact on everything from social media to healthcare. AI is used to make credit card decisions, to conduct video surveillance in airports, and to inform military operations. These technologies have the potential to harm or help the people that they serve. By applying an ethical lens, we can work toward identifying the harms that these technologies can cause to people and we can design and build them to reduce these harms - or decide not to build them.

Before selecting data and training models, it is important to carefully consider the human needs an AI system should serve - and if it should be built at all.

**Human-centered design (HCD)** is an approach to designing systems that serve people’s needs.

### Approach

HCD involves people in every step of the design process. Your team should adopt an HCD approach to AI as early as possible - ideally, from when you begin to entertain the possibility of building an AI system.

The following six steps are intended to help you get started with applying HCD to the design of AI systems. That said, what HCD means for you will depend on your industry, your resources, your organization and the people you seek to serve.

### 1. Understand people’s needs to define the problem

Working with people to understand the pain points in their current journeys can help find unaddressed needs. This can be done by observing people as they navigate existing tools, conducting interviews, assembling focus groups, reading user feedback and other methods. Your entire team – including data scientists and engineers – should be involved in this step, so that every team member gains an understanding of the people they hope to serve. Your team should include and involve people with diverse perspectives and backgrounds, along race, gender, and other characteristics. Sharpen your problem definition and brainstorm creative and inclusive solutions together.

> A company wants to address the problem of dosage errors for immunosuppressant drugs given to patients after liver transplants. The company starts by observing physicians, nurses and other hospital staff throughout the liver transplant process. It also interviews them about the current dosage determination process - which relies on published guidelines and human judgment - and shares video clips from the interviews with the entire development team. The company also reviews research studies and assembles focus groups of former patients and their families. All team members participate in a freewheeling brainstorming session for potential solutions.


### 2. Ask if AI adds value to any potential solution

Once you are clear about which need you are addressing and how, consider whether AI adds value.

- Would people generally agree that what you are trying to achieve is a good outcome?
- Would non-AI systems - such as rule-based solutions, which are easier to create, audit and maintain - be significantly less effective than an AI system?
- Is the task that you are using AI for one that people would find boring, repetitive or otherwise difficult to concentrate on?
- Have AI solutions proven to be better than other solutions for similar use cases in the past?

If you answered no to any of these questions, an AI solution may not be necessary or appropriate.

> A disaster response agency is working with first responders to reduce the time it takes to rescue people from disasters, like floods. The time- and labor-intensive human review of drone and satellite photos to find stranded people increases rescue time. Everybody agrees that speeding up photo review would be a good outcome, since faster rescues could save more lives. The agency determines that an AI image recognition system would likely be more effective than a non-AI automated system for this task. It is also aware that AI-based image recognition tools have been applied successfully to review aerial footage in other industries, like agriculture. The agency therefore decides to further explore the possibility of an AI-based solution.

### 3. Consider the potential harms that the AI system could cause

Weigh the benefits of using AI against the potential harms, throughout the design pipeline: from collecting and labeling data, to training a model, to deploying the AI system. Consider the impact on users and on society. Your privacy team can help uncover hidden privacy issues and determine whether privacy-preserving techniques like [differential privacy](https://developers.googleblog.com/2019/09/enabling-developers-and-organizations.html) or [federated learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) may be appropriate. Take steps to reduce harms, including by embedding people - and therefore human judgment - more effectively in data selection, in model training and in the operation of the system. If you estimate that the harms are likely to outweigh the benefits, do not build the system.

> An online education company wants to use an AI system to ‘read’ and automatically assign scores to student essays, while redirecting company staff to double-check random essays and to review essays that the AI system has trouble with. The system would enable the company to quickly get scores back to students. The company creates a harms review committee, which recommends that the system not be built. Some of the major harms flagged by the committee include: the potential for the AI system to pick up bias against certain patterns of language from training data and amplify it (harming people in the groups that use those patterns of language), to encourage students to ‘game’ the algorithm rather than improve their essays and to reduce the classroom role of education experts while increasing the role of technology experts.

### 4. Prototype, starting with non-AI solutions

Develop a non-AI prototype of your AI system quickly to see how people interact with it. This makes prototyping easier, faster and less expensive. It also gives you early information about what users expect from your system and how to make their interactions more rewarding and meaningful.

Design your prototype’s user interface to make it easy for people to learn how your system works, to toggle settings and to provide feedback.

The people giving feedback should have diverse backgrounds – including along race, gender, expertise and other characteristics. They should also understand and consent to what they are helping with and how.

> A movie streaming startup wants to use AI to recommend movies to users, based on their stated preferences and viewing history. The team first invites a diverse group of users to share their stated preferences and viewing history with a movie enthusiast, who then recommends movies that the users might like. Based on these conversations and on feedback about which recommended movies users enjoyed, the team changes its approach to how movies are categorized. Getting feedback from a diverse group of users early and iterating often allows the team to improve its product early, rather than making expensive corrections later.

### 5. Provide ways for people to challenge the system

People who use your AI system once it is live should be able to challenge its recommendations or easily opt out of using it. Put systems and tools in place to accept, monitor and address challenges.

Talk to users and think from the perspective of a user: if you are curious or dissatisfied with the system’s recommendations, would you want to challenge it by:

- Requesting an explanation of how it arrived at its recommendation?
- Requesting a change in the information you input?
- Turning off certain features?
- Reaching out to the product team on social media?
- Taking some other action?

> An online video conferencing company uses AI to automatically blur the background during video calls. The company has successfully tested its product with a diverse group of people from different ethnicities. Still, it knows that there could be instances in which the video may not properly focus on a person’s face. So, it makes the background blurring feature optional and adds a button for customers to report issues. The company also creates a customer service team to monitor social media and other online forums for user complaints.

### 6. Build in safety measures

Safety measures protect users against harm. They seek to limit unintended behavior and accidents, by ensuring that a system reliably delivers high-quality outcomes. This can only be achieved through extensive and continuous evaluation and testing. Design processes around your AI system to continuously monitor performance, delivery of intended benefits, reduction of harms, fairness metrics and any changes in how people are *actually* using it.

The kind of safety measures your system needs depends on its purpose and on the types of harms it could cause. Start by reviewing the list of safety measures built into similar non-AI products or services. Then, review your earlier analysis of the potential harms of using AI in your system (see Step 3).

Human oversight of your AI system is crucial:

- Create a human ‘red team’ to play the role of a person trying to manipulate your system into unintended behavior. Then, strengthen your system against any such manipulation.
- Determine how people in your organization can best monitor the system’s safety once it is live.
- Explore ways for your AI system to quickly alert a human when it is faced with a challenging case.
- Create ways for users and others to flag potential safety issues.

> To bolster the safety of its product, a company that develops a widely-used AI-enabled voice assistant creates a permanent internal ‘red team’ to play the role of bad actors that want to manipulate the voice assistant. The red team develops adversarial inputs to fool the voice assistant. The company then uses ‘adversarial training’ to guard the product against similar adversarial inputs, improving its safety.

### Learn more

To dive deeper into the application of HCD to AI, check out these resources:

Lex Fridman’s [introductory lecture](https://www.youtube.com/watch?v=bmjamLZ3v8A) on Human-Centered Artificial Intelligence
Google’s People + AI Research (PAIR) [Guidebook](https://pair.withgoogle.com/guidebook/)
Stanford Human-Centered Artificial Intelligence (HAI) [research](https://hai.stanford.edu/research)

## Examples

### 1) Reducing plastic waste

A Cambodian organization wants to help reduce the significant amounts of plastic waste that pollute the Mekong River System. Which of the following would be an appropriate way to start? 

- Watch the people currently addressing the problem as they navigate existing tools and processes.
- Conduct individual interviews with the people currently addressing the problem.
- Assemble focus groups that consist of people currently addressing the problem.
After you have answered the question, view the official solution by running the code cell below.

> All of them!

### 2) Detecting breast cancer

Pathologists try to detect breast cancer by examining cells on tissue slides under microscopes. This tiring and repetitive work requires an expert eye. Your team wants to create a technology solution that helps pathologists with this task in real-time, using a camera. However, due to the complexity of the work, your team has not found rule-based systems to be capable of adding value to the review of images.

Would AI add value to a potential solution? Why or why not?

> Yes, it would. People would generally agree that the goal is desirable, especially since the AI system will be working with pathologists rather than in their place. AI can help people with repetitive tasks and AI systems have proven effective in similar medical image recognition use cases. That said, it is important to follow current industry best practices and to be thorough in the rest of the design process, including in analyzing harms and in considering how medical practitioners will actually interact with the product in a medical setting.

### 3) Flagging suspicious activity

A bank is using AI to flag suspicious international money transfers for potential money laundering, anti-terrorist financing or sanctions concerns. Though the system has proven more effective than the bank’s current processes, it still frequently flags legitimate transactions for review.

What are some potential harms that the system could cause, and how can the bank reduce the impacts of these potential harms?

> One potential harm is that the AI system could be biased against certain groups, flagging, delaying or denying their legitimate transactions at higher rates than those of other groups. The bank can reduce these harms by selecting data carefully, identifying and mitigating potential bias (see Lessons 3 and 4), not operationalizing the system until potential bias is addressed and ensuring appropriate and continuous human oversight of the system once it is operational.

### 4) Prototyping a chatbot

During an ongoing pandemic outbreak, a country’s public health agency is facing a large volume of phone calls and e-mails from people looking for health information. The agency has determined that an AI-powered interactive chatbot that answers pandemic-related questions would help people get the specific information they want quickly, while reducing the burden on the agency’s employees. How should the agency start prototyping the chatbot?

- Build out the AI solution to the best of its ability before testing it with a diverse group of potential users.
- Build a non-AI prototype quickly and start testing it with a diverse group of potential users.

> The correct answer is: Build a non-AI prototype quickly and start testing it with a diverse group of potential users. Iterating on a non-AI prototype is easier, faster and less expensive than iterating on an AI prototype. Iterating on a non-AI prototype also provides early information on user expectations, interactions and needs. This information should inform the eventual design of AI prototypes.

### 5) Detecting misinformation

A social media platform is planning to deploy a new AI system to flag and remove social media messages containing misinformation. Though the system has proven effective in tests, it sometimes flags non-objectionable content as misinformation.

What are some ways in which the social media platform could allow someone whose message has been flagged to contest the misinformation designation?

> The social media company should ask customers how they would want to challenge a determination. It could be by easily accessing a challenge form on which a user can describe why their message does not contain misinformation, requesting further review by a human reviewer, requesting an explanation of why the content was flagged or a combination of these and other means.

### 6) Improving autonomous vehicles

What are some of the ways to improve the safety of autonomous vehicles? (You might pick more than one option.)

- Incorporate the safety features of regular vehicles.
- Test the system in a variety of environments.
- Hire an internal ‘red team’ to play the role of bad actors seeking to manipulate the autonomous driving system. Strengthen the system against the team’s attacks on an ongoing basis.

> All of these are great ways to improve safety.


