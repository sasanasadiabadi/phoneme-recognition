import numpy as np
import pandas as pd
import h5py
import os

from keras.models import Sequential
from keras.layers import Dense,Conv2D, MaxPool2D, Flatten, Dropout
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras  import optimizers

e = 0
num_class = 39
height, width, depth = 40, 1, 45
INPUT_SHAPE = (height,width,depth)

def get_model_cnn():
    model = Sequential()
    model.add(Conv2D(150,kernel_size=(7,1),padding='same',activation='sigmoid',input_shape=INPUT_SHAPE))
    model.add(MaxPool2D(pool_size=(4,1),strides=(2,1)))

    model.add(Conv2D(300,kernel_size=(5,1),padding='same',activation='sigmoid'))
    model.add(MaxPool2D(pool_size=(2,1),strides=(2,1)))

    model.add(Flatten())
    model.add(Dense(1024,activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(1024,activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(num_class,activation='softmax'))
    opt = optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# load test data
hf = h5py.File('test_data.h5', 'r')
X_test = hf['test_data'][:]
hf.close()

hf = h5py.File('test_label.h5', 'r')
Y_test = hf['test_label'][:]
hf.close()

print(np.shape(X_test))
print(np.shape(Y_test))

print('test data loaded from disk')

# load train data

hf = h5py.File('train_data.h5', 'r')
X_train = hf['train_data'][:]
hf.close()

hf = h5py.File('train_label.h5', 'r')
Y_train = hf['train_label'][:]
hf.close()

print(np.shape(X_train))
print(np.shape(Y_train))

print('train data loaded from desk...')

# initialize model or load from disk
if e==0 
    model = get_model_cnn()
    print("model initialized...")
else:
    model = model_from_json(open('model.json', 'r').read())
    model.load_weights("model_weights.h5")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Loaded model from disk...")

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=2, mode='auto')

model.fit(X_train,Y_train,validation_set=(X_test,Y_test),batch_size=256, epochs=20,
                , callbacks = [early_stop], shuffle=True ,verbose=2)

score = model.evaluate(X_test,Y_test)
print("test score", score[0])
print("test accuracy:", score[1])

# serialize model to JSON
# remove files if already exist
file1 = os.path.expanduser('~') + 'home/sasan/model.jason'
try:
    os.remove(file1)
except OSError:
    pass

file2 = os.path.expanduser('~') + 'home/sasan/model_weights.h5'
try:
    os.remove(file2)
except OSError:
    pass

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights.h5")
print("Model saved to disk")



