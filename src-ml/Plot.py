import matplotlib.pyplot as plt
import csv
x = []
x1 = []
x2 = []
i = 0

with open('prediction-results.csv','r') as csvfile:
    plots = csv.reader(csvfile)
    for row in plots:
        i += 1
        x.append((i))
        x1.append(float(row[0]))#predicted
        x2.append(float(row[1]))#actual

#plt.plot(x, x1) # plot train loss
plt.plot(x, x2) # plot validation loss
plt.xlabel('number of test cases')
plt.ylabel('load loss [%]')
plt.title('predicted vs actual load loss')
plt.legend(['predicted ','actual'], loc='upper right')
plt.show()
