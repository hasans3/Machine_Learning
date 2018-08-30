#Plot.py to plot the training & validation loss for different epochs
#Graphical representation of the training & validation loss


import matplotlib.pyplot as plt
import csv
x = []
x1 = []
x2 = []
i = 0

with open('loss.csv','r') as csvfile:
    plots = csv.reader(csvfile)
    for row in plots:
        i += 1
        x.append(float(row[0]))
        x1.append(float(row[1]))#predicted
        x2.append(float(row[2]))#actual

plt.plot(x, x1) # plot train loss
plt.plot(x, x2) # plot validation loss
plt.xlabel('epochs')
plt.ylabel('training loss')
plt.title('training vs validation loss')
plt.legend(['training','validation'], loc='upper right')
plt.show()
