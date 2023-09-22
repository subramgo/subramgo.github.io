---
layout: post
title: "Paying attention to Self-Attention heads"
categories: GenAI
---

In this blog, we will go through a bunch of numpy code to understand the working
of self-attention. There are a lot of tutorials out there where the attention
mechanism is explained with illustrations. I personally liked two of them

* [Understanding and coding Self attention mechanism of large langugage models from Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)

* [The illustrated transformer](http://jalammar.github.io/illustrated-transformer/)

List of words are the basic input units to any natural language processing task.
A standard text preprocessing looks like the below figure.

![Text Preprocessing Pipeline](/assets/textpipeline.png)

Without going into the details of each of these blocks, let us take the output of
embedding block and proceed.

{% highlight python %}
import numpy as np

SEQ_LEN   = 5
EMBD_LEN  = 10

# input matrix
x = np.random.normal(size=(SEQ_LEN,EMBD_LEN))

{% endhighlight %}
