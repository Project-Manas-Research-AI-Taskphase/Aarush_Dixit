import numpy as np
import matplotlib.pyplot as plt

#setting random seed
np.random.seed(2007)

#taking the dataset as an array
dataset = np.genfromtxt('TaskOne/Datasets/weatherHistory.csv', delimiter=',', dtype=float, autostrip=True, skip_header=1, usecols=(2,3,4,5,6,7,8))
# Collumns (0 is summary), (1 is prec type), (2 is temp in C), (3 is app temp in C), (4 is humidity), (5 is windspeed), (6 is wind bearing in degrees), (7 is visiblity in km), (8 is preassure in millibars)
# collumns after exclusion (0 is temp in C), (1 is app temp in C), (2 is humidity), (3 is windspeed), (4 is wind bearing in degrees), (5 is visiblity in km), (6 is preassure in millibars)
# 1024 rows of data and one heading row-> [0]

#first compute regression w/ one feature temp to predict humidity.

#taking all the data into different arrays
temperature = dataset[:,0]
apparent_temperature = dataset[:,1]
humidity = dataset[:,2]
wind_spd_kmph = dataset[:,3]
wind_bearings_degrees = dataset[:,4]
visibility_km = dataset[:,5]
pressure_millibars = dataset[:,6]

#suffle
n = len(dataset)
shuffled_idx = np.random.permutation(n)

#apply shuffle to both arrays
humidity_shuffled = humidity[shuffled_idx]
temperature_shuffled = temperature[shuffled_idx]

#setting splitpoint
splitpoint = 800

#take 800 rows for training and the rest (801-1024) for testing
x_train = temperature_shuffled[:splitpoint]
x_test = temperature_shuffled[splitpoint:]
y_train = humidity_shuffled[:splitpoint]
y_test = humidity_shuffled[splitpoint:]


# TO SOLVE: Y (HUMIDITY) = B + W*X(TEMP)

#makeing the matrices to solve - 
#first make a 1D array full of ones to make the second collumn
ones_collumn = np.ones((len(x_train),1))
#gotta reshape x array so we can give it 2D and then stack it
x_train_collumn = x_train.reshape(-1,1)
#final matrices filled with feature train values
X_train = np.hstack([x_train_collumn,ones_collumn])
#initialising weights as zeroes
weights = np.zeros(2)

#making test matrices
ones_collumn_2 = np.ones((len(x_test),1))
x_test_collumn = x_test.reshape(-1,1)
X_test = np.hstack([x_test_collumn,ones_collumn_2])


#making the predictions now - 
#standalone predict
def prediction(x,w,b):
    return x*w + b

#matrix prediction
def predict(X,weights):
    #multiplying the matrices to get prediction
    return X @ weights


#Mean Square Error (MSE) Function
def mean_square_error(y_actual, y_predicted):
    #subtract
    errors = y_actual - y_predicted
    #square
    squared_errors = errors ** 2
    #mean
    MSE = np.mean(squared_errors)
    return MSE


##gradient descent - converging on weights

#learning rate and iteration
learning_rate = 0.001
n_iterations = 2**15


#tracking progreesss
loss_history = []

#trainging loop
for i in range(n_iterations):

    #predict the y values ising current weights
    Y_predicted = predict(X_train,weights)
    #record the loss
    loss = mean_square_error(y_train, Y_predicted)
    #add loss to list
    loss_history.append(loss)

    #taking gradient(derivative) 
    gradient = (-2/len(X_train)) * X_train.T @ (y_train - Y_predicted)
    #updating weight to converge on the perfect weights
    weights = weights - (learning_rate * gradient)


#plot iteration vs error
plt.plot(loss_history)
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.title("Training loss over time")
plt.show()

#draw the scatter with the new ling taking new weights and bias
plt.scatter(x_train, y_train, alpha=0.3, label="train data")
x_range = np.linspace(x_train.min(), x_train.max(), 100)
y_line = weights[0] * x_range + weights[1]
plt.plot(x_range, y_line, color="red", label="fitted line")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.legend()
plt.show()


#compare test data with the new line
plt.scatter(x_test, y_test, alpha=0.3, label="test data")
x_range = np.linspace(x_test.min(), x_test.max(), 100)
y_line = weights[0] * x_range + weights[1]
plt.plot(x_range, y_line, color="red", label="fitted line")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.legend()
plt.show()

#final compare

#final weights after converging
print()
print("final weights [w, b]:", weights)
print(f"Final Equastion is: Humidity(y) = {weights[1]} + {weights[0]}*Temp(x)")
#final MSE
print()
print("final training loss:", loss_history[-1])

#compare with test data
print()
Y_test_predicted = predict(X_test, weights)
test_mse = mean_square_error(y_test, Y_test_predicted)
print("test MSE:", test_mse)
print(f"differnce in errors is {abs(test_mse-loss_history[-1])}")




#plot Humidity vs Temp
#plt.scatter(x_train, y_train, alpha=0.3, s=10, color = "green")
#plt.xlabel("Temperature (C)")
#plt.ylabel("Humidity")
#plt.title("Temperature vs Humidity")
#plt.show()
