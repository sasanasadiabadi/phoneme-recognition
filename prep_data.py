import numpy as np
import numpy.matlib
import pandas as pd
import pickle
import os
import h5py

from keras.utils import to_categorical
from sklearn import preprocessing

num_class = 11
height, width, depth = 40, 15, 3

h = height
ws = 15 # window size
w = int((ws-1)/2) # mid window

method = 1 # zero for 2D conv / one for 1D conv

#path_train = 'C:/Users/sabadi15/Desktop/ELEC/Project/cnn_train'
#path_test = 'C:/Users/sabadi15/Desktop/ELEC/Project/cnn_test'
path_train = '/home/sasan/my_files/cnn_train'
path_test = '/home/sasan/my_files/cnn_test'

def create_data(cnn_data,path):
    X_train = []
    Y_train = []
    for csvfile in cnn_data:
        data = pd.read_csv(os.path.join(path, csvfile),header=None)
        X_train.extend(data.iloc[:, 0:-1].values)
        Y_train.extend(data.iloc[:, -1].values)
    return np.array(X_train), np.array(Y_train)


train_data = os.listdir(path_train)

X_train, Y_train = create_data(train_data,path_train)

# padding data
X_train = np.vstack((np.matlib.repmat(X_train[0,:],w,1),X_train,np.matlib.repmat(X_train[-1,:],w,1)))

Y_train = np.hstack((np.matlib.repmat(Y_train[0],1,w),Y_train[None,:],np.matlib.repmat(Y_train[-1],1,w)))

# standardizing data --- zero mean/ unit variance
mean = np.mean(X_train,axis=0)
std = np.std(X_train,axis=0)

X_train = (X_train - mean) / std
#X_train = preprocessing.scale(X_train)

X_train = np.transpose(X_train)

Xtrn = []
Ytrn = []

if method==0:
    for i in range(w,np.shape(X_train)[1]-w):
        tmp = np.zeros((h, ws, 3))
        tmp[:,:,0] = X_train[0:h,i-w:i+w+1].copy()
        tmp[:,:,1] = X_train[h:2*h,i-w:i+w+1].copy()
        tmp[:,:,2] = X_train[2*h:3*h,i-w:i+w+1].copy()

        if Y_train[0,i] != 11:
            Xtrn.append(tmp)
            Ytrn.append(Y_train[0, i])

elif method==1:
    for i in range(w, np.shape(X_train)[1] - w):
        tmp = np.zeros((h,1,3*ws))
        c = 0
        for j in range(-w,w+1):
            tmp[:,:,c]   = X_train[0:h,i+j:i+j+1].copy()
            tmp[:,:,c+1] = X_train[h:2*h,i+j:i+j+1].copy()
            tmp[:,:,c+2] = X_train[2*h:3*h,i+j:i+j+1].copy()
            c +=3
            #d = np.dstack((tmp[:,:,0],tmp[:,:,1],tmp[:,:,2]))
        if Y_train[0,i] != 11:
            Xtrn.append(tmp)
            Ytrn.append(Y_train[0,i])

Xtrn = np.array(Xtrn)
Ytrn = np.array(Ytrn)

# one-hot-encoding
Ytrn = to_categorical(Ytrn,num_classes=num_class)

print(np.shape(Xtrn))
print(np.shape(Ytrn))

hf = h5py.File('train_data.h5', 'w')
hf.create_dataset('train_data', data=Xtrn)
hf.close()

hf = h5py.File('train_label.h5', 'w')
hf.create_dataset('train_label', data=Ytrn)
hf.close()
