# cancer_classifier.py

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Load and Preprocess Data
# Load dataset
dataset = pd.read_csv("cancer.csv")

# Define features (X) and label (y)
X = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define a sequential neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation="sigmoid", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Output layer for binary classification
])

# Compile the model with appropriate loss, optimizer, and evaluation metric
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# train the model
history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    verbose=1  # Set to 0 to silence output, or 2 for minimal output
)

# ----------------------------
# 4. Evaluate the Model
# ----------------------------

# Evaluate the model on training data (you can also evaluate on test data)
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
print(f"Training Accuracy: {train_accuracy:.4f}")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

