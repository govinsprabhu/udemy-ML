# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter ='\t', quoting = 3)

# Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Create the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)    
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values    


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# SVM
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
# Naive Bayes
"""from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)"""
    

# Fitting Decision Tree Classification to the Training set
'''from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)'''


# Fitting Random forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 20, criterion = 'entropy', random_state=0)
classifier.fit(X_train, y_train)

# Logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
TN = cm[0, 0]
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]

accuracy = (TN + TP ) / (TP + TN + FP + FN)

precision = TP/ (TP + FP)
recall = TP / (TP + FN)

f1_score = (2 * precision * recall) / (precision + recall)
# Naive bayes f1 = 0.7467
# SVM = 0.735
# Decision tree = 0.70
# Random forest = 0.72
# Logistic regression = 0.75
# KNN = 0.611
