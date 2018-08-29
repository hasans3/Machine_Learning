import csv
import numpy as np
from keras.models import model_from_json
import time
csvdata = open("prediction-results.csv", "w")

#Neural network prediction
def nnmodel(Xnew, model):
    inputs = np.array(Xnew)[np.newaxis]
    outputs = model.predict(inputs, batch_size=1)
    return [float(outputs[0][0])]


def main():
    time1= time.time()
    with open('val_(n-7).csv') as File:
        reader=csv.reader(File)
        for row in reader:
            X = []
            rowdata = []
            x=0

            #Appending inputs to the numpy array
            X = [float(row[1]),float(row[2]),float(row[3]),float(row[4]),
float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),
float(row[10]),float(row[11]),float(row[12]),float(row[13]),float(row[14]),
float(row[15]),float(row[16]),float(row[17]),float(row[18]),float(row[19]),
float(row[20])]

            #normalizing the inputs
            Xnew = (X - dataNormalization['X_mean'])/dataNormalization['X_std']

            #predicting output using the model
            y = nnmodel(Xnew, model)
            y = (y*dataNormalization['Y_std'])+dataNormalization['Y_mean']
            print(y)
            '''
            #set
            if((y[0])>0.5):
                x=1
            else:
                x=0
            '''
            writer = csv.writer(csvdata)
            writer.writerow((y))
    #time2=time.time()
    #t_time = time2 -time1
    #print ('Total execution time in seconds: %s'  %t_time)
if __name__ == "__main__":
    #Loading the nn model
    with open('model.json', 'r') as jfile:
        model = model_from_json(jfile.read())
    model.load_weights('weights.best.hdf5')

    #Loading the normalized parameters
    dataNormalization = np.load('normalizeParameters.npz')
    main()
