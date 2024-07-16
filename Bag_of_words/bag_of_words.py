# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

if __name__ == "__main__":
    
    # Read the IMDB Dataset 
    # https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download

    dataframe = pd.read_csv("NLP_Tutorials\IMDB Dataset.csv")

    print("Shape of the data: ", dataframe.shape)

    print("Head of the data: ", dataframe.head())

    # Create a new column "Category" which represent 1 if the sentiment is positive or 0 if it is negative
    dataframe['Category'] = dataframe['sentiment'].apply(lambda x: 1 if x =='positive' else 0)

    # Check the distribution
    print(dataframe["Category"].value_counts())

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(dataframe["review"], dataframe["Category"], test_size=0.2)

    # Create pipeline
    clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('randomforest', RandomForestClassifier())
    ])

    # Start the training
    clf.fit(X_train, y_train)

    # Test the classifier
    y_pred = clf.predict(X_test)

    # Print the result
    print(classification_report(y_test, y_pred))
