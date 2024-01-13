import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
 
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
    
np.random.seed(10)
(x_train_image, y_train_label),(x_test_image,y_test_label) = mnist.load_data()
 
x_train = x_train_image.reshape(60000,28,28,1).astype('float32')
x_test  = x_test_image.reshape(10000,28,28,1).astype('float32')
#normalization
x_train_n = x_train/255
x_test_n = x_test/255
#one_hot code
y_train_onehot = to_categorical(y_train_label)
y_test_onehot = to_categorical(y_test_label)
 
# build  a model 
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(3,3),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Flatten())
model.add(Dense(units = 32,kernel_initializer = 'normal',activation = 'relu'))
model.add(Dense(units = 10,kernel_initializer = 'normal',activation = 'softmax'))
 
print(model.summary())

input(">>")
# train
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x=x_train_n,y = y_train_onehot ,validation_split = 0.2, epochs = 10,batch_size = 256,verbose=2)
show_train_history(train_history,'accuracy','val_accuracy')
#test
scores = model.evaluate(x_test_n,y_test_onehot)
#print accuracy of test data
print(scores[1])
# analysize 
prediction = np.argmax(model.predict(x_test_n), axis=-1) # model.predict_classes(x_test_n)
#print a confusion matrix
pd.crosstab(y_test_label,prediction,rownames=['label'],colnames=['predicti'])
#df = pd.DataFrame({'label':Y_test_label,'predict':prediction})
#df[(df.label==5)&(df.predict==3)]
