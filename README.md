Creating a sophisticated AI-powered spam classifier involves various components and can be a complex project. Below is a simplified Python source code example using the scikit-learn library for building a basic spam classifier using a machine learning algorithm called Naive Bayes. Please note that a real-world implementation would require a much more extensive dataset and more advanced techniques.

```python
# Import necessary libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample spam and ham (non-spam) messages
messages = [
    ("Congratulations, you've won a $1000 gift card!", 1),
    ("Please find the attached report for review.", 0),
    # Add more examples here
]

# Split the messages into text and labels
text, labels = zip(*messages)

# Create a CountVectorizer to convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)

# You should now have a basic spam classifier, but in practice, more data and feature engineering are needed for better results.
```

To build a smarter and more robust spam classifier, consider using more advanced machine learning techniques (e.g., deep learning) and larger datasets. Additionally, you may want to explore natural language processing (NLP) libraries like spaCy or NLTK to enhance text preprocessing and feature extraction.
