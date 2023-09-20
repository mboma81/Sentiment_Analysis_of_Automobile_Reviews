# Import relevant libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import string


# Specify the full path to the dataset file
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

# Train the SVM model
svm = SVC()
svm.fit(X_train, y_train)

# Make predictions on the test data
predictions = svm.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
confusion_mat = confusion_matrix(y_test, predictions)

# Print the performance metrics
print("SVM Performance Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_mat)

# Save the predictions
# First we convert y_test and predictions to Pandas Series
y_test_series = pd.Series(y_test, name='labels')
predictions_series = pd.Series(label_encoder.inverse_transform(predictions), name='predictions')

# Create the test_data DataFrame by concatenating the Pandas Series
test_data = pd.concat([data.loc[y_test_series.index, 'reviews'], y_test_series, predictions_series], axis=1)

# Save the test_data DataFrame with predictions
test_data.to_csv('svm_predictions2.csv', index=False)


# Save the trained model
import joblib
joblib.dump(svm, 'svm_model2.pkl')

# Save the vectorizer
joblib.dump(vectorizer, 'countsv_vectorizer2.pkl')

# Save the SVM sentiment results
result_data = pd.DataFrame({'reviews': data.loc[y_test_series.index, 'reviews'], 'labels': y_test_series, 'predictions': predictions_series})

result_data.to_csv('svm2_sentiment_results2.csv', index=False)
