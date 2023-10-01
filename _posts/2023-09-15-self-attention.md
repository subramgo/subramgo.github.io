---
layout: post
title: "Paying attention to Self-Attention heads"
categories: GenAI
---

Why another self-attention post
---------------------

There are a lot of tutorials out there explaining
self attention mechanism with illustrations.

Two of my favorites are

- [Understanding and coding Self attention mechanism of large language models from Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)

- [The illustrated transformer](http://jalammar.github.io/illustrated-transformer/)


After reading the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) and the articles
mentioned above, I was curious about:

1. Projecting the embedded space into a feature subspace and subsequent
learning of these spaces during training.

2. Scaled self-attention, why scale the value.

3. Finally the Big O of computing pairwise score. Space and time complexity.

I will pass on the burden of deciding about the triviality of these
observations to the readers.


Background
--------

It is easy to understand *attention* as a feature selection problem.
Not all input features seen by a model are equally important for the task at hand. Prudence dictates we make the model pay more attention to important features than considering all the features presented to it.

1. For each feature, compute importance score. The higher the score, the more critical the feature.
2. Use these scores in downstream tasks to guide the model.

Some example concepts implementing the above steps:

- Lasso regression, l1 regularization ensures that some of the model coefficients are pushed towards zero.

- Maxpooling layers in CNN selects' the cell with maximum values and ignore the rest of cell values.

In each of the previously mentioned instances, a feature was either retained or discarded according to its importance score. Nevertheless, it's worth noting that these scores have the potential to enhance the information conveyed by the feature.

#### Context awareness

"Try to infer the meaning of the word based on its neighbors" - Many amongst us have heard this during our elementary reading classes.

In sequential features, in addition to individual elements, the relationship
among the elements carries a lot of information. This relationship is
called the context.

- In the sentence "premature optimization is the root cause
of all evil," the famous Donald Knuth quote, the word evil is better understood
from the context of "premature optimization."

- In a time series sales data, the sales of this quarter is related to the previous quarter,
the last year same quarter and similar other time based relationships.

- In image data, the neighboring pixels are related.

- In video data, pixel from the current frames can be enriched with the information from previous frames.

Self Attention is a way of enriching the features, making the features *context-aware*.


Self Attention bare bones
-----------------------

We will focus on the scaled self-attention proposed in the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf).


Its easy to comprehend (for me!) the *context-aware* phenomenon when the inputs are a list of words; basic input units to any natural language processing task.

In the sentence "John plays ukulele occasionally", we expect that for the token "John", the most related word token should be "plays". We expect to enrich the feature representation for token "John" with its relation to the token "plays".

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
a representation of words in a vector space. They map the language into a structured geometric space. Geometric relation between two word vectors will hence reflect the semantic relationship between them. We can leverage linear
algebra techniques to find similarity between words.

I personally like the 2013 google paper [Efficient estimation of word representations in vector space](https://arxiv.org/pdf/1301.3781.pdf). This paper introduces CBOW and Skip-ngram architecture
to learn word representations from large corpus. An example from this paper,
arithmetic operations performed on word vectors,


### Paris - France + Italy = Rome.

**Subracting the vector for France from Paris and adding the vector for Italy, gives a vector which is very close to Rome.**



### Pair-wise scoring

Here we find the similarity between words using dot product.

**For the initiated, to keep things simple, I have ignored scaling and
softmax normalization. Will take it up in the following sections.**



{% highlight python %}
pairwise_scoring = np.matmul(x,x.T)
{% endhighlight %}

Each word is a vector.We use the dot product similarity to find the
pair-wise scoring between the words. An illustration of pair wise calculation[^1].

![Pairwise Scoring](/assets/scoring.png)

These pairwise scores are called *attention scores*. Word close to each another
in the geometric space, will have a higher attention score.

The resultant pairwise_scoring is a 5 x 5 matrix which encodes the attention scores between these words.



### Enrich the input with attention score

In the above diagram, the row gives attention scores for word *eat* with the other words in the input. Enriching is performing of a weighted sum of the individual word embedding vectors with the attention scores.

The attention score for the word 'eat' and 'eat' is multiplied with 'eat' vector. The score for word 'the' and 'eat' is multiplied with 'the' vector and son on. Finally we add all these vectors to get the final enriched vector for the word 'eat'. Words which are semantically close to "eat" will contribute more and others will contribute less.

{% highlight python %}
enriched_x_ = np.zeros(shape=x.shape)
for i, scores in enumerate(pairwise_scoring):
    weighted_sum = np.zeros(x[0,:].shape)
    for j,score in enumerate(scores):
        weighted_sum+= x[j,:] * score
    enriched_x[i] = weighted_sum
{% endhighlight %}

We can vectorize the above operation.

{% highlight python %}
enriched_x = np.matmul(pairwise_scoring,x)
{% endhighlight %}

![Enrich input](/assets/enrich.png)


### Voila

This is the underlying idea behind self-attention.
Since the words are represented in the vector space, dot product of the
vectors provides the similarity scores between words. Semantically similar words will have high attention scores and finally will contribute more to word in question through weighted sum.


Scaled Self attention
------------
<table>
  <tr>
<td> <img src="/assets/attention.png"  alt="1" width = 150 height = 150 ></td>

<td><img src="/assets/barebone.jpg" alt="2" width = 150 height = 150>
    </td>
   </tr>
</table>

In the table above, the first column has  the picture from Attention is all you need paper. The second column depicts our bare bone attention explained
in the previous section.

1. Bare bone had a single input X, our word embeddings matrix. The scaled dot attention has three inputs, Q,K,V.

2. What are the additional boxes, scale, mask and softmax in scaled dot-product attention.

#### Query, Key, Value Model

Let us address the first question. All blogs and papers refer to input as Q,K and V. This comes from the search engine / recommendation terminology. Going back in history to early days of search engine, inverted indices was the data structure which powered searches on large databases.

Documents were indexed based on words. An entry in an inverted index can be imagined as Key,Value pair as shown,

{% highlight python %}
{keyword1,keyword21,keyword45}:{Document1,Document100,Document121}
{% endhighlight%}}

This kind of indexing accelerates keyword based searches.

Let us say the whole vocabulary is a set, V = {keyword1,....keywordm}; m keywords; and the document corpus is D ={Document1,...Documentn} ; n documents

Given a query, Q = {keyword1,keyword20,keyword21}. We can retrieve all the all documents containing the key word query as follows

results = sum( intersection(Q, K) * V) for all n Documents


{% highlight python %}
vocabulary = ['a','b','c','d']
documents  = ['d1','d2', 'd3','d4','d5']

inverted_index = {('a','b'): ['d1','d2']
                 ,('a','c'): ['d1','d2','d3']
                 ,('b'): ['d1','d2']
                 ,('c','d'): ['d4','d5']
                 }

query = ('a','c')


results = set()
for key,value in inverted_index.items():
    match = set(query).intersection(set(key))
    if  len(match) >= 1:
        for document in value:
            results.add(document)
{% endhighlight %}


The function intersection(Q, K) can be replaced by a scoring function instead of the discrete intersection. Imaging our *pairwise_scoring* replacing this function.

While explaining the bare bones attention, our input played the role of Query, Key and Value. There is no rule that Query, Key and Value should be three different inputs. Since this way of representation helps us describe multiple problems, like search, sequence to sequence learning, language models and others Query, Key, Value representation is used while describing attention.


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


Three weight matrices, wq,wk and wv are created. These matrices are used to project input matrix x to create three new matrices, wqp, wkp and wvp. The input shapes
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
- While doing the pair-wise scoring on the input matrix, you notice that the diagonal values of the resultant matrix is all 1. This would mean that we are instructing the word to attend more to itself. After projection this may not be the case. Without projection, the weights of other words are supressed.

- The matrices wq,wk and wv serve as a parameter to the final neural network and will be adjusted based on gradients during back probagation. This will allow the network to learn the new feature space. I always had this question about the importance of word embeddings for downstream
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

More importantly  we need to keep the magnitude of the dot product controlled because of the subsequent application of softmax.

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




[^1]: The resultant scoring matrix was filled with random numbers. Apologies to those meticulous readers who actually multiplied these matrices.
