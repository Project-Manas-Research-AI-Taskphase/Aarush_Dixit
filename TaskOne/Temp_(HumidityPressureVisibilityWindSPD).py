#GONNA MAKE ANOTHER BETTER VERSION (I SHOULD BE DOING CLASSIFICATIIOF FAHHHHHHHHHHHh)

#import libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#set random seed
np.random.seed(1610)

#reading and taking the dataset
dataset = pd.read_csv('TaskOne/Datasets/weatherHistory BIG.csv')
# Columns - Temp, App Temp, Humidity, Wind SPD, Visibility, Pressure
#(95165, 6) 
# Index(['Temperature (C)', 'Apparent Temperature (C)', 'Humidity','Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)'], dtype='object')


#taking target as temp and th rest as featues except app temp in pandas
target = dataset['Temperature (C)']
features = dataset[['Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']]
#converting these to my numpy array :)
Y = target.to_numpy()
X = features.to_numpy()


#number of numbers + splitoint at 80:20
n = len(Y)
splitpoint = math.floor(80/100 * n) 

#shuffling id then shuddling the arrays and storing them
shuffleID = np.random.permutation(n)
Y_shuffled = Y[shuffleID]
#also adding the one collumn
X_shuffled = X[shuffleID]

#setting train and test matrices
Y_train = Y_shuffled[:splitpoint]
Y_test = Y_shuffled[splitpoint:]
X_train = X_shuffled[:splitpoint]
X_test = X_shuffled[splitpoint:]

#standardising and scalling
X_train_mean = X_train.mean(axis = 0)
X_train_std = X_train.std(axis = 0)
#Also added the one collumns here to minimise lines
X_train_scaled = np.hstack([((X_train - X_train_mean) / X_train_std), np.ones([len(X_train),1])])
X_test_scaled = np.hstack([((X_test - X_train_mean) / X_train_std),np.ones([len(X_test),1])])

#making weights mattrix
weights = np.zeros(X_train_scaled.shape[1])

#Loss Histroy array
loss_history = []


### MAKING FUNCTIONS

#prediction function
def predict(x,w):
    #mulitply matrices
    return x @ w

#mean square error
def mse_calculate(y_actual, y_prediction):
    #calculate error and square
    errors = y_actual - y_prediction
    squared_errors = errors ** 2
    #return the mean
    return np.mean(squared_errors)

#mean absolute error
def mae_calculate(y_actual, y_prediction):
    #take error and mod
    errors = y_actual - y_prediction
    absolute_errors = np.abs(errors)
    #return the mean
    return np.mean(absolute_errors)
    
#R Squared
def R_squared(y_actual, y_prediction):
    residual_fit_square = np.sum((y_actual - y_prediction) ** 2)
    residual_mean_square = np.sum((y_actual - y_actual.mean()) ** 2)
    return 1 - (residual_fit_square/residual_mean_square)

#MSE Gradient
def mse_gradient(X, y_actual, y_predicted):
    return (-2/len(X)) * X.T @ (y_actual - y_predicted)

#MAE Gradient - wtf is this bruh
def mae_gradient(X, y_actual, y_predicted):
    errors = y_actual - y_predicted
    sign_of_errors = np.sign(errors)
    return (-1/len(X)) * X.T @ sign_of_errors


#START OF EXPERIENCE

start = input("Which model to train and plot? (MSE/MAE): ")
n_iterations = int(input("How many iterations? (Ex, 5000): "))
learning_rate = float(input("Learning Rate? (Ex, 0.01): "))

#setting loss functions and gradient functions
if start.lower() == "mse":
    loss_func = mse_calculate
    gradient_func  = mse_gradient

elif start.lower() == "mae":
    loss_func = mae_calculate
    gradient_func = mae_gradient

else:
    print("I don't have time for this.") 
    exit()


#training the loop
for i in range(n_iterations):
    #setting the prediction array and making the first prediction
    Y_predicted = predict(X_train_scaled, weights)
    #recording loss
    loss_history.append(loss_func(Y_train,Y_predicted))
    #taking gradient
    gradient = gradient_func(X_train_scaled, Y_train, Y_predicted)
    #updating weights
    weights = weights - (learning_rate * gradient)


#getting the final Y predicted and adding the latest loss then storing the last loss separately.
Y_train_predicted = predict(X_train_scaled, weights)
loss_history.append(loss_func(Y_train,Y_train_predicted))
final_train_loss = loss_history[-1]


#predicting the test data now
Y_test_predicted = predict(X_test_scaled,weights)


#metrics
print()
print(f"--- Results ({start.upper()}-trained model) ---")
print(f"final train loss ({start.upper()}): {final_train_loss}")
print()
print("train MeanSquareError:", mse_calculate(Y_train, Y_train_predicted))
print("test  MeanSquareError:", mse_calculate(Y_test, Y_test_predicted))
print("train MeanAbsoluteError:", mae_calculate(Y_train, Y_train_predicted))
print("test  MeanAbsoluteError:", mae_calculate(Y_test, Y_test_predicted))
print("train R**2:", R_squared(Y_train, Y_train_predicted))
print("test  R**2:", R_squared(Y_test, Y_test_predicted))

#---- weights in scaled space ----
feature_names = ['Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']
print()
print()
print("--- Weights (standardized - magnitudes comparable) ---")
for name, w in zip(feature_names, weights):
    print(name, ":", round(w, 4))
print("bias :", round(weights[-1], 4))

#---- weights converted back to raw units ----
raw_weights = weights[:-1] / X_train_std
raw_bias = weights[-1] - np.sum((weights[:-1] * X_train_mean) / X_train_std)
print()
print("--- Equation in raw units ---")
equation = f"Temp = {raw_bias:.4f}"
for name, w in zip(feature_names, raw_weights):
    equation += f" + ({w:.4f} * {name})"
print(equation)



## Drawing curvs
#---- loss curve ----
plt.plot(loss_history)
plt.xlabel("iteration")
plt.ylabel(start.upper())
plt.yscale("log")
plt.title(f"Training loss over time ({start.upper()})")
plt.show()

#---- predicted vs actual ----
plt.scatter(Y_test, Y_test_predicted, alpha=0.05, s=5)
lims = [Y_test.min(), Y_test.max()]
plt.plot(lims, lims, color="red", label="perfect prediction")
plt.xlabel("actual temperature (C)")
plt.ylabel("predicted temperature (C)")
plt.title(f"Predicted vs Actual ({start.upper()}, test set)")
plt.legend()
plt.show()

#---- residual plot ----
residuals = Y_test - Y_test_predicted
plt.scatter(Y_test_predicted, residuals, alpha=0.05, s=5)
plt.axhline(0, color="red")
plt.xlabel("predicted temperature (C)")
plt.ylabel("residual (actual - predicted)")
plt.title(f"Residuals ({start.upper()}, test set)")
plt.show()


