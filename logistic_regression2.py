# Import relevant libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import string


# Setting the working directory
dataset_path = "C:/Users/nkemj/Documents/DISSERTATION/NEW DATA/review_with_labels2.csv"

# Load the dataset
data = pd.read_csv(dataset_path)
reviews = data["reviews"]
labels = data["labels"]

# Preprocess the data
# Remove punctuation from the reviews
reviews = reviews.str.replace('[{}]'.format(string.punctuation), '')

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Create a CountVectorizer for text representation
vectorizer = CountVectorizer()

# Convert text reviews into numerical features
X = vectorizer.fit_transform(reviews)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the Logistic Regression model

from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train_scaled, y_train)

# Make predictions on the test data
predictions = logistic_regression.predict(X_test_scaled)

# Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
confusion_mat = confusion_matrix(y_test, predictions)

# Print the performance metrics
print("Logistic Regression Performance Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_mat)

#save the trained model
#first, let us import joblib
import joblib

# Save the trained model
joblib.dump(logistic_regression, 'logistic_regression_model2.pkl')

# Convert y_test and predictions to Pandas Series
y_test_series = pd.Series(y_test, name='labels')
predictions_series = pd.Series(label_encoder.inverse_transform(predictions), name='predictions')

# Create the result_data DataFrame by concatenating the Pandas Series
result_data = pd.concat([data.loc[y_test_series.index, 'reviews'], y_test_series, predictions_series], axis=1)

# Save the result_data DataFrame with predictions
result_data.to_csv('logistic_regression_predictions2.csv', index=False)

# Save the sentiment analysis results to a CSV file
result_data.to_csv('sentiment_analysis_results_logistic2.csv', index=False)

# Save the vectorizer
joblib.dump(vectorizer, 'count_vectorizer2.pkl')
