---
layout: post
title: "Paying attention to Self-Attention heads"
categories: GenAI
---

Yet another attention
---------------------

In this article, we will go through a bunch of numpy code to understand the working
of self-attention. There are a lot of tutorials out there where the attention
mechanism is explained with illustrations. I personally liked two of them

- [Understanding and coding Self attention mechanism of large langugage models from Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)

- [The illustrated transformer](http://jalammar.github.io/illustrated-transformer/)


As we work through the numpy code, I have shared some thoughts and expanded on
some details. I will pass on the burden of deciding about the triviality of these
observations to the readers. A list of those observational topics are.

1. Projecting the embedded space into a feature subspace and subsequent learning of
these spaces during training.

2. Scaled self-attention

Projection matrices
-----------------------

List of words are the basic input units to any natural language processing task.
A standard text preprocessing looks like the below figure.

![Text Preprocessing Pipeline](/assets/textpipeline.png)

Without going into the details of each of these blocks, let us pretend we took the output of
embedding block and proceed.

{% highlight python %}
import numpy as np

SEQ_LEN   = 5
EMBD_LEN  = 10

# input matrix
x = np.random.normal(size=(SEQ_LEN,EMBD_LEN))

{% endhighlight %}

The input is a matrix of size (5 x 10), where each row represents the embedding of a word token,
a total of 5 word tokens.

{% highlight python }
# dimensions of q,k and v
q_d = k_d = 20 # dimension of query and key weights
v_d = 25       # dimension of value matrix weight

# weight matrices
wq = np.random.normal(size=(q_d, EMBD_LEN))
wk = np.random.normal(size=(k_d, EMBD_LEN))
wv = np.random.normal(size=(v_d, EMBD_LEN))

# projection operation
wqp = np.matmul(wq,x.T).T
wkp = np.matmul(wk,x.T).T
wvp = np.matmul(wv,x.T).T

{% endhighlight %}


Three weight matrices, wq,wk and wv are created. These matrices are used to project
our input matrix x to create three new matrices, wqp, wkp and wvp. The input shapes
of all the participating matrices are given below,
