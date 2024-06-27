import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset of emails and labels (0 for not spam, 1 for spam)
emails = ["Buy now, special offer!!!", "Hello, can we schedule a meeting?", "Get rich quick!", "URGENT: Important message", "Free gift inside!"]
labels = [1, 0, 1, 1, 1]

# Preprocess the text data using a Count Vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)

# Example to predict a new email
new_email = ["Congratulations, you've won a prize!"]
new_email_vectorized = vectorizer.transform(new_email)
prediction = clf.predict(new_email_vectorized)

if prediction[0] == 0:
    print("Not spam")
else:
    print("Spam")
