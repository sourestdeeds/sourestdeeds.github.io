---
title: 'The Sliding Window'
tags: [kaggle, sliding window, keras, deep learning, neural network]
layout: post
mathjax: true
categories: [Computer Vision]
published: true
---

In the previous two lessons, we learned about the three operations that carry out feature extraction from an image:

- *filter*with a **convolution** layer.
- *detect* with **ReLU** activation.
- *condense* with a **maximum pooling** layer.

The convolution and pooling operations share a common feature: they are both performed over a **sliding window**. With convolution, this "window" is given by the dimensions of the kernel, the parameter *kernel_size*. With pooling, it is the pooling window, given by *pool_size*.

<br>
[![gif](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-20-sliding-window/1.gif#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-20-sliding-window/1.gif)<br> 

There are two additional parameters affecting both convolution and pooling layers -- these are the *strides* of the window and whether to use *padding* at the image edges. The *strides* parameter says how far the window should move at each step, and the *padding* parameter describes how we handle the pixels at the edges of the input.

With these two parameters, defining the two layers becomes:

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  activation='relu'),
    layers.MaxPool2D(pool_size=2,
                     strides=1,
                     padding='same')
    # More layers follow
])
```

### Stride

The distance the window moves at each step is called the stride. We need to specify the stride in both dimensions of the image: one for moving left to right and one for moving top to bottom. This animation shows strides $=(2, 2)$, a movement of 2 pixels each step

<br>
[![gif](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-20-sliding-window/2.gif#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-20-sliding-window/2.gif)<br> 

What effect does the stride have? Whenever the stride in either direction is greater than 1, the sliding window will skip over some of the pixels in the input at each step.

Because we want high-quality features to use for classification, convolutional layers will most often have strides$=(1, 1)$. Increasing the stride means that we miss out on potentially valuble information in our summary. Maximum pooling layers, however, will almost always have stride values greater than 1, like $(2, 2)$ or $(3, 3)$, but not larger than the window itself.

Finally, note that when the value of the strides is the same number in both directions, you only need to set that number; for instance, instead of strides$=(2, 2)$, you could use strides$=2$ for the parameter setting.

### Padding

When performing the sliding window computation, there is a question as to what to do at the boundaries of the input. Staying entirely inside the input image means the window will never sit squarely over these boundary pixels like it does for every other pixel in the input. Since we aren't treating all the pixels exactly the same, could there be a problem?

What the convolution does with these boundary values is determined by its padding parameter. In TensorFlow, you have two choices: either *padding* = 'same' or *padding* = 'valid'. There are trade-offs with each.

When we set *padding* = 'valid', the convolution window will stay entirely inside the input. The drawback is that the output shrinks (loses pixels), and shrinks more for larger kernels. This will limit the number of layers the network can contain, especially when inputs are small in size.

The alternative is to use *padding* = 'same'. The trick here is to **pad** the input with 0's around its borders, using just enough 0's to make the size of the output the same as the size of the input. This can have the effect however of diluting the influence of pixels at the borders. The animation below shows a sliding window with 'same' padding.

<br>
[![gif](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-20-sliding-window/3.gif#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-20-sliding-window/3.gif)<br> 















<br>
[![jpeg](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-20-sliding-window/1.jpeg#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-20-sliding-window/1.jpeg)<br> 


