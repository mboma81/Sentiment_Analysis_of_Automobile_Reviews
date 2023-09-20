import pandas as pd
import string
import joblib

# Load the new review dataset
new_reviews = pd.read_csv("C:/Users/nkemj/Documents/DISSERTATION/NEW DATA/new_reviews.csv")

# Preprocess the new review dataset
preprocessed_reviews = new_reviews['reviews'].str.lower().str.replace('[{}]'.format(string.punctuation), '')

# Load the trained models
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
svm_model = joblib.load('svm_model.pkl')

# Load the CountVectorizers
naive_bayes_vectorizer = joblib.load('count_vectorizer.pkl')
logistic_regression_vectorizer = joblib.load('count2_vectorizer.pkl')
svm_vectorizer = joblib.load('countsv_vectorizer.pkl')

# Convert the new reviews into numerical features using the CountVectorizers
X_new_naive_bayes = naive_bayes_vectorizer.transform(preprocessed_reviews)
X_new_logistic_regression = logistic_regression_vectorizer.transform(preprocessed_reviews)
X_new_svm = svm_vectorizer.transform(preprocessed_reviews)

# Predict the sentiment using the trained models
naive_bayes_predictions = naive_bayes_model.predict(X_new_naive_bayes)
logistic_regression_predictions = logistic_regression_model.predict(X_new_logistic_regression)
svm_predictions = svm_model.predict(X_new_svm)

# Map the numerical labels to sentiment categories
sentiment_categories = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
# Create a new DataFrame to store the predictions
predictions_df = pd.DataFrame({
    'Review': new_reviews['reviews'],
    'Naive Bayes Label': naive_bayes_predictions,
    'Naive Bayes Sentiment': [sentiment_categories[prediction] for prediction in naive_bayes_predictions],
    'Logistic Regression Label': logistic_regression_predictions,
    'Logistic Regression Sentiment': [sentiment_categories[prediction] for prediction in logistic_regression_predictions],
    'SVM Label': svm_predictions,
    'SVM Sentiment': [sentiment_categories[prediction] for prediction in svm_predictions]
})

# Save the predictions to a CSV file
predictions_df.to_csv('review_predictions6.csv', index=False)

####################################

