# Electrical-ML

Predicting the load loss and blackout of a 20 component electrical system using neuraql network.\
The prediction has two parts:\
1)Blackout prediction: Predict a binary output, 1 representing complete system blackout and 0, representing no blackout.\
2)Load loss estimation: Estimating the load loss of the system.


***Results:(load loss: 07/19/2018)***

Model trained for (n-5) and predicting (n-6) load loss: MSE = 3.46.\
Model trained for (n-5) and predicting (n-7) load loss: MSE = 3.17.\
Model trained for (n-6) and predicting (n-7) load loss: MSE = 3.19.\

***Results:(Blackout prediction: 07/19/2018)***

1) model-(n-6) and predicting (n-7).\
samples 19448\
False Negatives: 102 (0.5%)\
False Positives: 46  (0.2%)

2) good-model-(n-6).\
samples 19448\
False Negatives: 72 (0.37%)\
False Positives: 4  (0.02%)



