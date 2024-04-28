from keras.datasets import imdb
from keras import models, layers
import numpy as np

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Vectorize sequences
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.array(train_labels).astype('float32')
y_test = np.array(test_labels).astype('float32')


# Define the model
model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10000,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
#model.sequential we create a linear stack of layers
#layers - access various types of layers

# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#rmsprop - adjust learning rate dynamically

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])



# Train the model
history = model.fit(partial_x_train , partial_y_train , epochs=20 , batch_size=512 , validation_data=(x_val,y_val))


model.predict(x_test)
# values are close to 0 or 1.
# This indicates that the model is quite confident in its predictions. In a binary classification problem, a prediction
# value close to 0 suggests that the model predicts the class as 0, and a value
# close to 1 suggests a prediction of class 1.

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:',test_acc)
