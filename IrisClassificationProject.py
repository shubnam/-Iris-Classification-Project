import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)

# Preprocess the data
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
encoder = LabelEncoder()
y = encoder.fit_transform(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = keras.utils.to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = keras.Sequential()
model.add(keras.layers.Dense(units=10, activation='relu', input_shape=(4,)))
model.add(keras.layers.Dense(units=3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions on the test set
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Print the classification report and confusion matrix
print(classification_report(np.argmax(y_test, axis=-1), y_pred))
print(confusion_matrix(np.argmax(y_test, axis=-1), y_pred))

# Plot the training history
#plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label='val_accuracy')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
#plt.legend(loc='lower right')
#plt.show()
# Plot training accuracy
plt.plot(history.history['accuracy'], label='accuracy')

# Check if validation accuracy is available
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
elif 'val_acc' in history.history:
    plt.plot(history.history['val_acc'], label='val_accuracy')
elif 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='val_accuracy')

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
