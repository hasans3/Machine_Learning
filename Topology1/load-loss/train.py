#train.py to train the neural network model
#Model predicts the load loss values

import csv
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json, load_model
from sklearn.model_selection import train_test_split
from pathlib import Path
from keras.callbacks import CSVLogger
history = CSVLogger('loss.csv', append=True, separator=';')

def readData():
    X = []
    Y = []

    with open('train-(n-6).csv') as File:
        reader=csv.reader(File)
        for row in reader:
            rowdata = []

            #Appending inputs to the numpy array
            rowdata.append(float(row[1]))
            rowdata.append(float(row[2]))
            rowdata.append(float(row[3]))
            rowdata.append(float(row[4]))
            rowdata.append(float(row[5]))
            rowdata.append(float(row[6]))
            rowdata.append(float(row[7]))
            rowdata.append(float(row[8]))
            rowdata.append(float(row[9]))
            rowdata.append(float(row[10]))
            rowdata.append(float(row[11]))
            rowdata.append(float(row[12]))
            rowdata.append(float(row[13]))
            rowdata.append(float(row[14]))
            rowdata.append(float(row[15]))
            rowdata.append(float(row[16]))
            rowdata.append(float(row[17]))
            rowdata.append(float(row[18]))
            rowdata.append(float(row[19]))
            rowdata.append(float(row[20]))
            rowdata.append(float(row[21]))

            #Appending inputs
            X.append(rowdata)

            #Appending outputs
            Y.append(float(row[22]))

        #Converting to numpy array
        X = np.array(X)
        Y = np.array(Y)
        index = np.arange(len(X))
        index = np.random.permutation(index)#Shuffle the array

        #selecting training dataset
        X = X[index[:31118]]
        Y = Y[index[:31118]]
        #print(X.shape)
        return X,Y

#Normalizing data
def normalizeData(X, Y):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean)/X_std
    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)
    Y = (Y - Y_mean)/Y_std
    np.savez('./normalizeParameters', X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)
    return X, Y

#NN model
def createModel():
    inp=Input(shape=(21,))
    layer1=Dense(80,activation='relu', kernel_initializer='glorot_normal')(inp)
    layer2=Dense(100,activation='relu', kernel_initializer='glorot_normal')(layer1)
    layer3=Dense(100,activation='relu', kernel_initializer='glorot_normal')(layer2)
    layer4=Dense(80,activation='relu', kernel_initializer='glorot_normal')(layer3)
    layer4a=Dense(1,activation='tanh', kernel_initializer='glorot_normal')(layer4)
    out=layer4a
    model=Model(inputs=inp, outputs=out)
    return model

#training the model
def trainModel(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse', optimizer=adam)

    # checkpoint
    filePath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filePath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, history]
    model.fit(X, Y,epochs=350,batch_size=128,callbacks=callbacks_list,validation_data=(X_test, Y_test),verbose=2)

def saveModel(model):
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights("model.h5")
	print("Saved model to disk")


if __name__ == "__main__":
    X, Y = readData()
    X, Y = normalizeData(X, Y)
    if (Path("model.json").is_file() and Path("weights.best.hdf5").is_file()):
        with open('model.json', 'r') as jfile:
            model = model_from_json(jfile.read())
        model.load_weights("weights.best.hdf5")
        print("load from the existing model...")
    else:
        model = createModel()
        print(model.summary())
        print("create a new model")

    trainModel(model, X, Y)
    saveModel(model)
