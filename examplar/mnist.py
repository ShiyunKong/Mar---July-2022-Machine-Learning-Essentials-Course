import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

mnist= tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train , x_test = x_train / 255.0, x_test / 255.0

digit = Sequential([Flatten(input_shape=(28,28)), Dense(32, activation='sigmoid'), Dense(10, activation=tf.nn.softmax)])
digit.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
digit.fit(x_train, y_train, epochs=7)


digit.evaluate(x_test, y_test)
