---
layout: post
title: "Paying attention to Self-Attention heads"
categories: GenAI
---

Why another self-attention post
---------------------

There are a lot of tutorials out there explaining
attention mechanism with illustrations.

Two of my favorites are

- [Understanding and coding Self attention mechanism of large language models from Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)

- [The illustrated transformer](http://jalammar.github.io/illustrated-transformer/)


While reading the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) and the articles
mentioned above, I was curious about these topics:

1. Projecting the embedded space into a feature subspace and subsequent
learning of these spaces during training.

2. Scaled self-attention, why scale the value.

3. Finally the Big O of computing pairwise score. Space and time complexity. 

I will pass on the burden of deciding about the triviality of these
observations to the readers.


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

{% highlight python %}
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

{% highlight python %}
x, input shape (5, 10)
wq, q weight matrix shape (20, 10)
wk, k weight matrix shape (20, 10)
wv, v weight matrix shape (25, 10)

wqp, q weight matrix shape (5, 20)
wkp, k weight matrix shape (5, 20)
wvp, v weight matrix shape (5, 25)
{% endhighlight %}

The initial input matrix x, with dimension 5 x 10 is now projected into three
new matrices. In wqp and wkp the embedded space of dimension 10 is projected into a
feature space of dimension 20. In wvp its projected into a space of dimension 25.

The matrices wq,wk and wv serve as a parameter to the final neural network and will
be adjusted based on gradients during back probagation. This will allow the network
to learn the new feature space.

I always had this question about the importance of word embeddings for downstream
natural language processing tasks. Had recently put a survey in linked in.
[Clicke here for the survey](https://www.linkedin.com/posts/gopi-subramanian-39bba651_building-custom-llms-activity-7096593842291818496-sdNe?utm_source=share&utm_medium=member_desktop)
Say we are building a classifier on a corpus of
engineering notes written by a technician servicing an aircraft. The lingo and word tokens
are going to reflect the domain. Previously trained embeddings may not have a corresponding
word vectors. In those scenarios do we resort to make an embedding space for that vocabulary before
we build our classifier? I don't have a clear answer. However, by projecting some of these
embeddings into another subspace, we can hope that transformers or other dense network ingesting these
project matrices will learn them during the training process.In worst case, the bytepair encoding or sentencepair encoding may split some of these words in a way we don't want. The downstream learning may become
completely useless.

## Calculate the self-attention

Finally we calculate the self-attention. In a nutshell we find the similarity between
the items in our matrices. We have projected 5 words from embedded space to a new feature space, we find the similarity between these five word tokens.

{% beginhighlight python %}

score = np.matmul(wqp, wkp.T)
scaled_score = score / np.sqrt(wkd)


{% endhighlight%}

The resultant matrix, score is a 5 x 5 matrix.

The dot product can get very large, hence the scaling. From the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf), the foot note explains the need for scaling.

"To illustrate why the dot products get large, assume that the components of q and k are independent random
variables with mean 0 and variance 1. Their dot product will have a mean of 0 and variance of dk"

More importantly the we need to keep the dot product controlled because of the subsequent application of softmax.

{% beginhighlight python %}
def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1,keepdims=True)

scaled_softmax_score = softmax(scaled_score)

{% endhighlight python %}

Without scaling, applying softmax on very large value can lead to arithmetic computation problem. Exponent of large numbers can result in very large values.

Finally the attention context vector is created as follows.

{% beginhighlight python %}

context_vector = np.sum(np.matmul(scaled_softmax_score, wvp),axis=0)

{% endhighlight%}
