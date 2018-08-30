#error.py to get the mean square error between the model predicted value and the actual value

import matplotlib.pyplot as plt
import csv
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

def main():
    data1=[]
    data2=[]

    with open('result_(n-9).csv','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            predicted = float(row[0])
            actual = float(row[1])
            data1.append(predicted)#appending predicted values
            data2.append(actual)#appending actual values
    mse = sqrt(mean_squared_error(data2, data1))#mean square error between the predicted & actual
    print(mse)

if __name__ == "__main__":
    main()
