#keras1.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# define first architecture of the model
#
model = Sequential()    # creates an empty sequential model 
# layer 1
layer_1 = Dense(units=2, activation='sigmoid', input_dim = 3)  ## change it to 3, 4, 5, 6, .. to see results
model.add(layer_1)
# layer 2
layer_2 = Dense(units=1, activation='sigmoid')
model.add(layer_2)


print(model.summary())  # to verify model structure

print("\n\n layer_1 : ", layer_1.input_shape, layer_1.output_shape)
print("\n\n layer_2 : ", layer_2.input_shape, layer_2.output_shape)

# tell Keras what loss,optimizer
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer =opt)
# at this stage, the model is not done yet.

# now for training data
np.random.seed(9)
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[1],[0]])

# training
model.fit(X,y, epochs=7, verbose=0)



# now the model is reasy. you can use it for prediction

# we check it on training data
print(model.predict(X))

# good to test on new data, test data.

#%%
