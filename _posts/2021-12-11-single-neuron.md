---
title: 'A Single Neuron'
tags: [kaggle, single neuron, keras, deep learning, neural network]
layout: post
mathjax: true
categories: [Deep Learning]
published: true
---

Using Keras and Tensorflow we'll learn how to:

- Create a fully-connected neural network architecture.
- Apply neural nets to two classic ML problems: **regression** and **classification**.
- Train neural nets with **stochastic gradient descent**.
- Improve performance with **dropout**, **batch normalization**, and other techniques.

### What is Deep Learning?

Some of the most impressive advances in artificial intelligence in recent years have been in the field of *deep learning*. Natural language translation, image recognition, and game playing are all tasks where deep learning models have neared or even exceeded human-level performance.

So what is deep learning? **Deep learning** is an approach to machine learning characterized by deep stacks of computations. This depth of computation is what has enabled deep learning models to disentangle the kinds of complex and hierarchical patterns found in the most challenging real-world datasets.

Through their power and scalability **neural networks** have become the defining model of deep learning. Neural networks are composed of neurons, where each neuron individually performs only a simple computation. The power of a neural network comes instead from the complexity of the connections these neurons can form.

### The Linear Unit

So let's begin with the fundamental component of a neural network: the individual neuron. As a diagram, a **neuron** (or **unit**) with one input looks like:

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-11-single-neuron/1.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-11-single-neuron/1.png)<center><b>Figure 1:</b> The Linear Unit: 𝑦=𝑤𝑥+𝑏</center><br> 

The input is \\( x \\). Its connection to the neuron has a weight which is \\( w \\). Whenever a value flows through a connection, you multiply the value by the connection's weight. For the input \\( x \\), what reaches the neuron is \\( w \times x \\). A neural network "learns" by modifying its weights.

The \\( b \\) is a special kind of weight we call the bias. The bias doesn't have any input data associated with it; instead, we put a 1 in the diagram so that the value that reaches the neuron is just \\( b \\) (since \\( 1 \times b = b \\)). The bias enables the neuron to modify the output independently of its inputs.

The \\( y \\) is the value the neuron ultimately outputs. To get the output, the neuron sums up all the values it receives through its connections. This neuron's activation is \\( y = w \times x + b \\).


### Example - The Linear Unit as a Model

Though individual neurons will usually only function as part of a larger network, it's often useful to start with a single neuron model as a baseline. Single neuron models are linear models.

Let's think about how this might work on a dataset like [80 Cereals](https://www.kaggle.com/crawford/80-cereals). Training a model with 'sugars' (grams of sugars per serving) as input and 'calories' (calories per serving) as output, we might find the bias is \\( b=90 \\) and the weight is \\( w=2.5 \\). We could estimate the calorie content of a cereal with 5 grams of sugar per serving like this:

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-11-single-neuron/2.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-11-single-neuron/2.png)<center><b>Figure 2:</b> Computing with the linear unit.</center><br> 

And, checking against our formula, we have calories \\( =2.5\times5+90=102.5 \\), just like we expect.

### Multiple Inputs

The *80 Cereals* dataset has many more features than just 'sugars'. What if we wanted to expand our model to include things like fiber or protein content? That's easy enough. We can just add more input connections to the neuron, one for each additional feature. To find the output, we would multiply each input to its connection weight and then add them all together.

[![png](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-11-single-neuron/3.png#center)](https://raw.githubusercontent.com/sourestdeeds/sourestdeeds.github.io/main/_posts/2021-12-11-single-neuron/3.png)<center><b>Figure 3:</b> A linear unit with three inputs.</center><br> 

The formula for this neuron would be \\( y = w_0 x_0 + w_1 x_1 + w_2 x_2 + b \\). A linear unit with two inputs will fit a plane, and a unit with more inputs than that will fit a hyperplane.

### Linear Units in Keras

The easiest way to create a model in Keras is through `keras.Sequential`{:.language-python .highlight}, which creates a neural network as a stack of *layers*. We can create models like those above using a dense layer.

We could define a linear model accepting three input features ('sugars', 'fiber', and 'protein') and producing a single output ('calories') like so:

```python
from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
```

```
User settings:

   KMP_AFFINITY=granularity=fine,verbose,compact,1,0
   KMP_BLOCKTIME=0
   KMP_SETTINGS=1
   KMP_WARNINGS=0

Effective settings:

   KMP_ABORT_DELAY=0
   KMP_ADAPTIVE_LOCK_PROPS='1,1024'
   KMP_ALIGN_ALLOC=64
   KMP_ALL_THREADPRIVATE=128
   KMP_ATOMIC_MODE=2
   KMP_BLOCKTIME=0
   KMP_CPUINFO_FILE: value is not defined
   KMP_DETERMINISTIC_REDUCTION=false
   KMP_DEVICE_THREAD_LIMIT=2147483647
   KMP_DISP_NUM_BUFFERS=7
   KMP_DUPLICATE_LIB_OK=false
   KMP_ENABLE_TASK_THROTTLING=true
   KMP_FORCE_REDUCTION: value is not defined
   KMP_FOREIGN_THREADS_THREADPRIVATE=true
   KMP_FORKJOIN_BARRIER='2,2'
   KMP_FORKJOIN_BARRIER_PATTERN='hyper,hyper'
   KMP_GTID_MODE=3
   KMP_HANDLE_SIGNALS=false
   KMP_HOT_TEAMS_MAX_LEVEL=1
   KMP_HOT_TEAMS_MODE=0
   KMP_INIT_AT_FORK=true
   KMP_LIBRARY=throughput
   KMP_LOCK_KIND=queuing
   KMP_MALLOC_POOL_INCR=1M
   KMP_NUM_LOCKS_IN_BLOCK=1
   KMP_PLAIN_BARRIER='2,2'
   KMP_PLAIN_BARRIER_PATTERN='hyper,hyper'
   KMP_REDUCTION_BARRIER='1,1'
   KMP_REDUCTION_BARRIER_PATTERN='hyper,hyper'
   KMP_SCHEDULE='static,balanced;guided,iterative'
   KMP_SETTINGS=true
   KMP_SPIN_BACKOFF_PARAMS='4096,100'
   KMP_STACKOFFSET=64
   KMP_STACKPAD=0
   KMP_STACKSIZE=8M
   KMP_STORAGE_MAP=false
   KMP_TASKING=2
   KMP_TASKLOOP_MIN_TASKS=0
   KMP_TASK_STEALING_CONSTRAINT=1
   KMP_TEAMS_THREAD_LIMIT=4
   KMP_TOPOLOGY_METHOD=all
   KMP_USE_YIELD=1
   KMP_VERSION=false
   KMP_WARNINGS=false
   OMP_AFFINITY_FORMAT='OMP: pid %P tid %i thread %n bound to OS proc set {%A}'
   OMP_ALLOCATOR=omp_default_mem_alloc
   OMP_CANCELLATION=false
   OMP_DEFAULT_DEVICE=0
   OMP_DISPLAY_AFFINITY=false
   OMP_DISPLAY_ENV=false
   OMP_DYNAMIC=false
   OMP_MAX_ACTIVE_LEVELS=1
   OMP_MAX_TASK_PRIORITY=0
   OMP_NESTED: deprecated; max-active-levels-var=1
   OMP_NUM_THREADS: value is not defined
   OMP_PLACES: value is not defined
   OMP_PROC_BIND='intel'
   OMP_SCHEDULE='static'
   OMP_STACKSIZE=8M
   OMP_TARGET_OFFLOAD=DEFAULT
   OMP_THREAD_LIMIT=2147483647
   OMP_WAIT_POLICY=PASSIVE
   KMP_AFFINITY='verbose,warnings,respect,granularity=fine,compact,1,0'

2021-11-09 00:11:17.133284: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
```

With the first argument, units, we define how many outputs we want. In this case we are just predicting 'calories', so we'll use units=1.

With the second argument, input_shape, we tell Keras the dimensions of the inputs. Setting `input_shape=[3]`{:.language-python .highlight} ensures the model will accept three features as input ('sugars', 'fiber', and 'protein').

> Why is *input_shape* a Python list?
> The data we'll use in this course will be tabular data, like in a Pandas dataframe. We'll have one input for each feature in the dataset. The features are arranged by column, so we'll always have `input_shape=[num_columns]`{:.language-python .highlight}. The reason Keras uses a list here is to permit use of more complex datasets. Image data, for instance, might need three dimensions: `[height, width, channels]`{:.language-python .highlight}.


