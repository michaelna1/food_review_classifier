import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import nltk
import seaborn as sns
import pickle
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score


nltk.download('stopwords')
nltk.download('wordnet')

#read review data into a Pandas data frame
food_data = pd.read_csv('Reviews.csv')

# remove punctuation from text data using regex expression [^\w\s], which means everything but words and spaces will be removed.
food_data['Text'] = food_data['Text'].str.replace(r'[^\w\s]', "", regex = True)

# convert text to lower case by iterating over each word split from each text review string and concatenating back together.
food_data['Text'] = food_data['Text'].apply(lambda review: " ".join(word.lower() for word in review.split()))

#remove stop words using dictionary of English stop words from NLTK corpus
stop_words = stopwords.words('english')
food_data['Text'] = food_data['Text'].apply(lambda review: " ".join(word for word in review.split() if word not in stop_words))

# Lemmatization-- converted to base words
food_data['Text'] = food_data['Text'].apply(lambda review: " ".join(Word(word).lemmatize() for word in review.split()))


# 75% of the data will be used to train, while the rest will provide testing data.

#remove null values from the data
food_data.dropna(inplace=True)

#Remove neutral reviews from the training data
training_data = food_data[food_data['Score'] != 3].copy()

#Create a sentiment column for training purposes. Scores of 4-5 will be positive, while 1-2 will be negative.
def convert_to_sentiment(score):
    if score >= 4:
        return "Positive"
    else:
        return "Negative"

training_data['Sentiment'] = training_data['Score'].apply(convert_to_sentiment)

#Make a "document" of review text that can be fed into the vectorizer.
corpus = training_data['Text'].tolist()

#Vectorizer determines the number of unique tokens (words) and creates an array of vectors, in which each vector encodes the frequency of
#these words in a review. This is a bag of words model.
cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).toarray()

y = training_data['Sentiment'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#The target column, sentiment, to be used in training the machine learning model.
y = training_data['Sentiment'].tolist()

#Split the data into a training set and testing set. 75% will be used for training.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Train the multinomial naive bayes model. Alpha provides Laplace smoothing, fit_prior determines the prior probabilitities of classes.
classifier = MultinomialNB(alpha=0.8, fit_prior = True, force_alpha = True)
classifier.fit(X_train, y_train)

#Test the trained model.
y_pred = classifier.predict(X_test)

#Calculate the accuracy of the model using the test data.
accuracy = accuracy_score(y_test, y_pred)*100

#pickle the model for persistence, so that it can be used elsewhere
pickle.dump(classifier, open('classifier.pkl', 'wb'))

#Display the performance of the model on the test data as a confusion matrix.
cm = confusion_matrix(y_test, y_pred)

labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']

#Place labels into array to match confusion matrix formatting.
labels = np.asarray(labels).reshape(2,2)

#Display confusion matrix as percentages of the total number of test reviews.
sns.heatmap(cm/np.sum(cm)*100, annot=labels, fmt = "", cmap='Reds', cbar_kws={'label': 'Percentage of Test Reviews'})
plt.title('Confusion Matrix for Food Product Review Testing Data', loc='center')
plt.show()
print("")
print(f"Accuracy score for testing data is{accuracy: 0.1f}%")


