---
layout: post
title: How to automatically create Base Line Estimators using scikit learn.
comments: true
---

For any machine learning problem, say a classifier in this case, it's always handy to create quickly a base line classifier against which we can compare our new models. You don't want to spend a lot of time creating these base line classifiers; you would rather spend that time in building and validating new features for your final model. In this post we will see how we can rapidly create base line classifier using scikit learn package for any dataset.

[code@github](https://github.com/subramgo/MachineLearningMisc/blob/master/BaseEstimator.py)


<!--break-->

## Input data set

Let us use **iris** dataset for demonstration purpose.

```
# Load Libraries
import numpy as np 
from sklearn import datasets


# Let us use Iris dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target

```

## DummyClassifier

Scikit provides the class *DummyClassifier* to help us create our base line model rapidly.
Module **sklearn.dummy** has the [DummyClassifier][2] class. Its api interfaces are very similar to any other model in scikit learn, use the *fit* function to build the model and *predict* function to perform classification.

```
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='stratified', random_state = 100, constant = None)
dummy.fit(x, y)


```
### Strategy - stratified

Let us look at the parameters while initializing *DummyClassifier*. The first parameter *strategy* is used to define the modus operandi of our Dummy Classifier. In the example above we have selected *stratified* as the strategy. According to this strategy, the classifier looks at the class distribution in our target variable to make its predictions.

In Iris data set case, there are 150 total records and three classes. The classes are uniformly distributed, there are 50 records per each class. 33.33% is the class distribution. 

```
## Number of classes
print "Number of classes :{}".format(dummy.n_classes_)
## Number of classes assigned to each tuple
print "Number of classes assigned to each tuple :{}".format(dummy.n_outputs_)
### Prior distribution of the classes.
print "Prior distribution of the classes {}".format(dummy.class_prior_)


Number of classes :3
Number of classes assigned to each tuple :1
Prior distribution of the classes [ 0.33333333  0.33333333  0.33333333]

```

Under the hood scikit learn uses *multinomial* distribution leveraging the class distribution values for performing the predictions. 

[The multinomial distribution is a multivariate generalisation of the binomial distribution. Take an experiment with one of p possible outcomes][1], in our case 3 possible outcomes.

Let us use *multinomial* distribution from numpy to show how scikit learn does it internally.

```
output = np.random.multinomial(1, [.33,.33,.33], size = x.shape[0])
predictions = output.argmax(axis=1)

print output[0]
print predictions[1]


```

Oh la.. our predictions are ready. Our *output* variable is a matrix of size (150,3), one dimension for each class. In the next line we use the argmax function to get the index which is set to 1. This is our predicted class. Now that we know what is happening under the hood, let us call the *predict* function and print some accuracy metrics for our dummy classifier.

```
y_predicted = dummy.predict(x)

print y_predicted
# Find model accuracy
print "Model accuracy = %0.2f"%(accuracy_score(y,y_predicted) * 100) + "%\n"

# Confusion matrix
print "Confusion Matrix"
print confusion_matrix(y, y_predicted, labels=list(set(y)))


```

### Strategy - most_frequent

In the case of *most_frequent*, the dummy classifier will always predict the label which occurs most frequently in the training set. 

### Strategy - constant

Will always predict the label provided by the user. While initializing the *DummyClassifier* class, we need to provide this label as a parameter, parameter name : constant.



### Imbalanced data set

Let us look a the models generated when our dataset is imbalanced. The most_frequent strategy we discussed will return a biased classifier, as they will tend to pick up the majority class. The accuracy score in this case will be proportional to the majority class ratio. Let us simulate an imbalanced dataset and create our dummy classifiers with different strategies.

```
from sklearn.datasets import make_classification

x, y = make_classification(n_samples = 100, n_features = 10, weights = [0.3, 0.7])
# Print the class distribution
print np.bincount(y)

```

As you can see the final print output is **[31 69]**, we have an imbalanced dataset. Let us proceed to build our dummy models with *most_frequent* and *stratified* strategies.

```

dummy1 = DummyClassifier(strategy='stratified', random_state = 100, constant = None)
dummy1.fit(x, y)
y_p1 = dummy1.predict(x)

dummy2 = DummyClassifier(strategy='most_frequent', random_state = 100, constant = None)
dummy2.fit(x, y)
y_p2 = dummy2.predict(x)

```

And finally some metrics for our models.

```
from sklearn.metrics import precision_score, recall_score, f1_score
print
print "########## Metrics #################"
print "     Dummy Model 1, strategy: stratified,    accuracy {0:.2f}, precision {0:.2f}, recall {0:.2f}, f1-score {0:.2f}"\
						.format(accuracy_score(y, y_p1), precision_score(y, y_p1), recall_score(y, y_p1), f1_score(y, y_p1))
print "     Dummy Model 2, strategy: most_frequent, accuracy {0:.2f}, precision {0:.2f}, recall {0:.2f}, f1-score {0:.2f}"\
						.format(accuracy_score(y, y_p2), precision_score(y, y_p2), recall_score(y, y_p2), f1_score(y, y_p2))

print


########## Metrics #################
     Dummy Model 1, strategy: stratified,    accuracy 0.58, precision 0.58, recall 0.58, f1-score 0.58
     Dummy Model 2, strategy: most_frequent, accuracy 0.69, precision 0.69, recall 0.69, f1-score 0.69




```

Dummy Model 2, as expected is biased towards the majority class. Dummy model 1 may serve as a realistic baseline to begin with.

### Strategies: uniform and prior

Strategy *prior* is very similar to stratified as it uses the prior distribution of classes.

```
dummy3 = DummyClassifier(strategy='prior', random_state = 100, constant = None)
dummy3.fit(x, y)
y_p3 = dummy3.predict(x)


print "     Dummy Model 3, strategy: prior, accuracy {0:.2f}, precision {0:.2f}, recall {0:.2f}, f1-score {0:.2f}"\
						.format(accuracy_score(y, y_p3), precision_score(y, y_p3), recall_score(y, y_p3), f1_score(y, y_p3))



```

Strategy *uniform* generates  predictions uniformly at random.


```
dummy4 = DummyClassifier(strategy='uniform', random_state = 100, constant = None)
dummy4.fit(x, y)
y_p4 = dummy4.predict(x)


print "     Dummy Model 4, strategy: uniform, accuracy {0:.2f}, precision {0:.2f}, recall {0:.2f}, f1-score {0:.2f}"\
						.format(accuracy_score(y, y_p4), precision_score(y, y_p4), recall_score(y, y_p4), f1_score(y, y_p4))


```

That is all for this post. Will continue with another post explaining *DummyRegressor*.


## Links

[1] https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html
[2] http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier


[1]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html
[2]: http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier



{% include comments.html %}






