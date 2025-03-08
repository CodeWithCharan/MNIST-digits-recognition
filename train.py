from tensorflow import keras
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# load train and test data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# rescale (0 - 255) to (0 to 1)
X_train = X_train / 255
X_test = X_test / 255

# initialize the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# compile optimizer, loss function and metrics
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# save the model
model.save("mnist_model.keras")
print("Model training complete! Saved as mnist_model.keras")