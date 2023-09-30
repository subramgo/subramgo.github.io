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


After reading the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) and the articles
mentioned above, I was curious about these topics:

1. Projecting the embedded space into a feature subspace and subsequent
learning of these spaces during training.

2. Scaled self-attention, why scale the value.

3. Finally the Big O of computing pairwise score. Space and time complexity.

I will pass on the burden of deciding about the triviality of these
observations to the readers.


Background
--------

It is easy to understand *attention* as a feature selection problem.
Not all input features seen by a model are equally important for the task at hand. Prudency dictates we make the model pay more attention to important features than considering all the features presented to it.

1. For each feature, compute the importance score. The higher the score, the more important
the feature.
2. Use these scores in your downstream task to guide the model.

Some of the existing techniques follow the above steps:

- Lasso regression, l1 regularization ensures that some of the model coefficients
are pushed towards zero.

- Maxpooling layers in CNN selects' the cell with maximum values and ignore the rest of cell values.

From the above example, we may conclude the need for these importance scores is
to either keep or discard a feature. However, these scores can enrich
the information the feature carries.

In sequential features, in addition to individual elements, the relationship
among the elements carries a lot of information. This relationship is commonly
called the context.

- In the sentence "premature optimization is the root cause
of all evil," the famous Donald Knuth quote, the word evil is better understood
from the context of "premature optimization."

- In a time series sales data, the sales of this quarter is related to the previous quarter,
the last year same quarter and similar other time based relationships.

- In image data, the neighboring pixels are related.

- In video data, pixel from the current frames can be enriched with the information from previous frames.

Self Attention is a way of enriching the features, by adding context
information.

Self Attention bare bones
-----------------------

We will focus on the scaled self-attention proposed in the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf).

In the context of self attention the context information captured using the importance score is used to enrich the input.

Its easy to comprehend (for me!) the context information when the inputs are a list of words provided for any downstream natural language processing task. List of words are the basic input units to any natural language processing task.

In this sentence "John plays ukulele occasionally", we expect that for the token "John", the most related word token should be "plays". This what is called as context. So we need to enrich the representation for the token "John" with its relation to the token "plays".

A standard text preprocessing looks like the below figure.

![Text Preprocessing Pipeline](/assets/textpipeline.png)

Without going into the details of each of these blocks, let us pretend we took the output of embedding block.

{% highlight python %}

import numpy as np

SEQ_LEN   = 5
EMBD_LEN  = 10

x = np.random.normal(size=(SEQ_LEN,EMBD_LEN))

{% endhighlight %}


The input is a matrix of size (5 x 10), where each row represents the embedding of a word token, a total of 5 word tokens.

### Why embeddings

In 2003 [Yoshua Bengio](https://scholar.google.com/citations?user=kukA0LcAAAAJ&hl=en) and his team introduced embeddings in the paper, A neural probabilistic language model. Journal of Machine Learning Research, 3:1137-1155, 2003. Since then there are several follow up research and development occurred in the embedding space.

For our purpose to continue with the rest of the article, word embeddings are
a representation of words in a vector space. Thus we can leverage linear
algebra techniques to find similarity between words.

I personally like the 2013 google paper [Efficient estimation of word representations in vector space](https://arxiv.org/pdf/1301.3781.pdf). This paper introduces CBOW and Skip-ngram architecture
to learn word representations from large corpus.

### Pair-wise scoring

**For the initiated, to keep things simple, I have ignored scaling and
softmax normalization. Will take it up in the following sections.**


{% highlight python %}
pairwise_scoring = np.matmul(x,x.T)
{% endhighlight %}

Each word is a vector.We use the dot product similarity to find the
pair-wise scoring between the words. An illustration of pair wise calculation[1^].

![Pairwise Scoring](/assets/scoring.png)


The resultant pairwise_scoring is a 5 x 5 matrix which encodes the similarity between these words.



### Enrich the input with pairwise scoring

We enrich the input matrix x with this pairwise scoring

{% highlight python %}
enriched_x = np.matmul(pairwise_scoring,x)
{% endhighlight %}

![Enrich input](/assets/enrich.png)

Enriching is taking the sum of word token vectors, where each vector is
weighted by pairwise score. An non vectorized implementation should make it clear.


{% highlight python %}
enriched_x_ = np.zeros(shape=x.shape)
for i, scores in enumerate(pairwise_scoring):
    weighted_sum = np.zeros(x[0,:].shape)
    for j,score in enumerate(scores):
        weighted_sum+= x[j,:] * score
    enriched_x[i] = weighted_sum
{% endhighlight %}

This is the underlying idea behind self-attention.
Since the words are represented in the vector space, dot product of the
vectors provides the similarity scores between words. If two words have occurred together enough times in the corpus used to train the embedding model, the dot product will say they are similar. This score is further used to enrich the input.


Self attention expanded
------------
<table>
  <tr>
<td> <img src="/assets/attention.png"  alt="1" width = 150 height = 150 ></td>

<td><img src="/assets/barebone.jpg" alt="2" width = 150 height = 150>
    </td>
   </tr>
</table>

In the table above, the first column has  the picture from Attention is all you need paper. The second column depicts our barebone attention explained
in the previous section.

1. Our barebone had a single input X, our word embeddings matrix. The scaled dot attention has three inputs, Q,K,V.

2. What are the additional boxes, scale, mask and softmax in scaled dot-product attention.

Let us address the first question. All blogs and papers refer to input as Q,K and V. This comes from the search engine / recommendation terminology.

In the case of self attention, Q, K and V are transformations of the input matrix. In the following code, wqp, wkp and wvp are equivalent to Q, K and V matrices depicted in the diagram.


{% highlight python %}
q_d = k_d = 20 # dimension of query and key weights
v_d = 25       # dimension of value matrix weight

wq = np.random.normal(size=(q_d, EMBD_LEN))
wk = np.random.normal(size=(k_d, EMBD_LEN))
wv = np.random.normal(size=(v_d, EMBD_LEN))

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

### Why do the transformation

Why not use the input matrix x directly as we did in the bare bone attention example ?
While doing the pair-wise scoring on the input matrix, you notice that the diagonal
values of the resultant matrix is all 1. This would mean that we are instructing the word
to attend more to itself.

The matrices wq,wk and wv serve as a parameter to the final neural network and will
be adjusted based on gradients during back probagation. This will allow the network
to learn the new feature space. I always had this question about the importance of word embeddings for downstream
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

{% highlight python %}

score = np.matmul(wqp, wkp.T)
scaled_score = score / np.sqrt(wkd)


{% endhighlight %}

The resultant matrix, score is a 5 x 5 matrix.

The dot product can get very large, hence the scaling. From the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf), the foot note explains the need for scaling.

"To illustrate why the dot products get large, assume that the components of q and k are independent random
variables with mean 0 and variance 1. Their dot product will have a mean of 0 and variance of dk"

More importantly the we need to keep the dot product controlled because of the subsequent application of softmax.

{% highlight python %}
def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1,keepdims=True)

scaled_softmax_score = softmax(scaled_score)

{% endhighlight  %}

Without scaling, applying softmax on very large value can lead to arithmetic computation problem. Exponent of large numbers can result in very large values.

Finally the attention context vector is created as follows.

{% highlight python %}

context_vector = np.sum(np.matmul(scaled_softmax_score, wvp),axis=0)

{% endhighlight %}


[1^] The resultant scoring matrix was filled with random numbers. Apologies to those meticulous readers who actually multiplied these matrices.
