import matplotlib.pyplot as plt
import csv
import math
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

x = []
x1 = []
x2 = []
i = 0
'''
with open('prediction-results.csv','r') as csvfile:
    plots = csv.reader(csvfile)
    for row in plots:
        error=abs(float(row[0])- float(row[1]))
        i += 1
        x.append((i))
        x2.append((error))
        x1.append(float(row[0]))#predicted
        x2.append(float(row[1]))#actual

#plt.plot(x, x1) # plot train loss
plt.plot(x, x2) # plot validation loss
plt.xlabel('number of test cases')
plt.ylabel('load loss [%]')
plt.title('predicted vs actual load loss')
plt.legend(['predicted ','actual'], loc='upper right')
plt.show()
'''

#csvdata = open("distance.csv", "w")
data1=[]
data2=[]

with open('prediction-results.csv','r') as csvfile:
    plots = csv.reader(csvfile)
    for row in plots:
        predicted = float(row[0])
        actual = float(row[1])
        data1.append(predicted)
        data2.append(actual)


#mse=np.sqrt(((np.array(data1) - np.array(data2)) ** 2).mean())
mse = sqrt(mean_squared_error(data2, data1))
print(mse)
