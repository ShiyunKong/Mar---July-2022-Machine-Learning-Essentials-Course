from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

data = fetch_20newsgroups()
print("\n\n======\n\ntarget names=", data.target_names)

# for simplicity we take ojnly a few categories
categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space',
'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# here is an example data
print("\n\n\n=====================================\n\n\n")
print(train.data[7])

##
##
##In order to use this data for machine learning, we need to be able to
##convert the content of each string into a vector of numbers.
##For this we will use the TF–IDF vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

#model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model = make_pipeline(CountVectorizer(), MultinomialNB())


model.fit(train.data, train.target)
labels = model.predict(test.data)

print("accuracy = ", model.score(test.data, test.target))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
print(mat)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

