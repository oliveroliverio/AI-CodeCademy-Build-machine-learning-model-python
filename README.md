* * *

# 1. Introduction to Machine Learning

## Investigate some common applications of machine learning. Then learn the difference between supervised and unsupervised learning.

**Lesson: Why Use Machine Learning?**
*Supervised learning*

Data is labelled and computer predicts output from input data.  For example, credit card fraud detection.

*Regression*

A subset of supervised learning.  Trying to predict a  continuous-valued output.  i.e., housing prices, value of cryptocurrencies.

*Classification*

A subset of supervised learning.  Trying to predict a discrete number of values. i.e.,  is the picture of a human or robot?  Is this email spam or not?

*Instructions*/*Example*

NYBD bot department with a **Naive Bayes** classifier that determines if text is "positive" or "negative."  i.e., "This hot dog sucks."

```
from texts import text_counter, text_training
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

intercepted_text = "This hot dog was awful!"

text_counts = text_counter.transform([intercepted_text])

text_classifier = MultinomialNB()

text_labels = [0] * 1000 + [1] * 1000

text_classifier.fit(text_training, text_labels)

final_pos = text_classifier.predict_proba(text_counts)[0][1]

final_neg = text_classifier.predict_proba(text_counts)[0][0]

if final_pos > final_neg:
  print("The text is positive.")
else:
  print("The text is negative.")
```

Try a new text, "I love my government."
Try your own text, "blah blah..." negative, "this is great" positive.

*Unsupervised learning*
Computer learns the inherent structure of data based on unlabelled examples

*Clustering*

A subset of unsupervised learning.  Examples: social network clustering of newsfeeds, consumer sites clustering users for recommendations, search engines that group similar objects in one cluster.

*Instructions/example*

[left off here](https://www.codecademy.com/paths/machine-learning/tracks/introduction-to-machine-learning-skill-path/modules/introduction-to-machine-learning-skill-path/lessons/why-machine-learning/exercises/unsupervised-learning)

**Article: The Machine Learning Process**
-
**Article: Supervised vs. Unsupervised**
**Article: What is Scikit-Learn?**
**Article: Scikit-Learn Cheatsheet**

* * *

# 2. Supervised Learning: Regression

## Learn the fundamentals of linear regression and implement them using scikit-learn!

**Lesson: Distance Formula**
**Article: Regression vs. Classification**
**Lesson: Linear Regression**
**Quiz: Linear Regression**
**Project: Honey Production**

## Multiple Linear Regression uses two or more independent variables to predict the value of the dependent variable.

**Article: Training Set vs Validation Set vs Test Set**
**Lesson: Multiple Linear Regression**
**Quiz: Multiple Linear Regression**

* * *

# 3. Regression Cumulative Project

## Use what you learned about regression in this off platform project!

**Info: Predict a Yelp Rating**

* * *

# 4. Supervised Learning: Introduction to Classification

## K-Nearest Neighbors is one of the most common supervised machine learning algorithms for classification.

**Lesson: K-Nearest Neighbors**
**Lesson: K-Nearest Neighbor Regressor**
**Quiz: K-Nearest Neighbors**
**Project: Breast Cancer Classifier**
**Article: Normalization**

## Learn how to evaluate the effectiveness of your machine learning model.

**Lesson: Accuracy, Recall, Precision, and F1 Score**
**Quiz: Accuracy, Recall, Precision, and F1 Score**
**Article: The Dangers of Overfitting**

## Find the probability of data samples belonging to a specific class with one of the most popular classification algorithms.

**Lesson: Logistic Regression**
**Quiz: Logistic Regression**
**Project: Predict Titanic Survival**

* * *

# 5. Supervised Learning: Advanced Classification

## Learn how to create complex decision boundaries used for classification by creating Support Vector Machines.

**Lesson: Support Vector Machines**
**Quiz: Support Vector Machines**
**Project: Sports Vector Machine**

## Learn how to build and use decision trees and random forests - two powerful supervised machine learning models.

**Lesson: Decision Trees**
**Quiz: Decision Trees**
**Project: Find the Flag**
**Lesson: Random Forests**
**Quiz: Random Forests**
**Project: Predicting Income with Random Forests**

## Bayes’ Theorem allows us to incorporate prior knowledge of conditions related to the event into probability calculations.

**Lesson: Bayes' Theorem**
**Lesson: Naive Bayes Classifier**
**Quiz: Naive Bayes Classifier**
**Project: Email Similarity**

* * *

# 6. Supervised Machine Learning Cumulative Project

## Use your knowledge of supervised machine learning models to find patterns in social media data.

**Info: Twitter Classification Cumulative Project**

* * *

# 7. Unsupervised Learning

## Learn K-Means, one of the most popular unsupervised machine learning models.

**Lesson: K-Means Clustering**
**Quiz: K-Means Clustering**
**Lesson: K-Means++ Clustering**
**Project: Handwriting Recognition using K-Means**

* * *

# 8. Unsupervised Machine Learning Cumulative Project

## Use your understanding of unsupervised learning and clustering to find patterns in a survey conducted about masculinity.

**Info: Masculinity Survey**

* * *

# 9. Perceptrons and Neural Nets

## Learn about the most basic type of neural net, the single neuron perceptron! You will use it to divide linearly-separable data.

**Article: What are Neural Networks?**
**Lesson: Perceptron**
**Quiz: Perceptron Quiz**
**Project: Perceptron Logic Gates**

* * *

# 10. Machine Learning Capstone Project

## Try to find true love using machine learning in our Date-A-Scientist final project.