#LOGOSTIC REGRESSSION finding out if the masses like a certain candy
#BLAH'Logistic regression to classify candies as above- or below-median popularity based on their characteristics.'BLAH

#imprt daa lib
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#set random seed again
np.random.seed(1)

#reading the entire dataset from CSV
dataset = pd.read_csv('TaskOne/Datasets/candy-data.csv')
#Columns (['competitorname', 'chocolate', 'fruity', 'caramel', 'peanutyalmondy','nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus', 'sugarpercent','pricepercent', 'winpercent'])
#choosing which collumns I want
columns = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy','nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus', 'sugarpercent','pricepercent', 'winpercent']
dataset = dataset[columns]

#Target is yes/no based on winpercent
#features are loaded
features = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy','nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus', 'sugarpercent','pricepercent']
#why should I take median over 50?
#Y = (dataset["winpercent"] > 50).astype(int).to_numpy()
Y = (dataset["winpercent"] > dataset["winpercent"].median()).astype(int).to_numpy()
X = dataset[features].to_numpy()

#number of points and splitpoint 80:20
n = len(Y)
splitpoint = math.floor(0.8 * n)

#shuffle and apply to array
idX = np.random.permutation(n)
Y_shuffled = Y[idX]
X_shuffled = X[idX]

#setting train and test matrices
Y_train = Y_shuffled[:splitpoint]
Y_test = Y_shuffled[splitpoint:]
X_train = np.hstack([X_shuffled[:splitpoint],np.ones([splitpoint,1])])
X_test = np.hstack([X_shuffled[splitpoint:],np.ones([n-splitpoint,1])])

#making weights mattrix
weights = np.zeros(X_train.shape[1])
#bias is weights[-1] rest are weights

#Loss Histroy array
loss_history = []
'''
                            Don't need this since data set is already normalised
#standardising and scalling
X_train_mean = X_train.mean(axis = 0)
X_train_std = X_train.std(axis = 0)
#Also added the one collumns here to minimise lines
X_train_scaled = np.hstack([((X_train - X_train_mean) / X_train_std), np.ones([len(X_train),1])])
X_test_scaled = np.hstack([((X_test - X_train_mean) / X_train_std),np.ones([len(X_test),1])])
'''

### MAKING DA FUNCS

#sigmoid func
def sigmoid_func(z):
    #garbage array
    out = np.empty_like(z, dtype=float)
    #set T/F for +vs and -ve parts
    pos, neg = z >= 0, z < 0
    #compute +ve and negative parts together and sepearee
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    out[neg] = np.exp(z[neg]) / (1 + np.exp(z[neg]))
    return out

#cross entropy loss func :()
def binary_cross_entropy_loss_func(y, p, eps = 1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

#gradient of BCE (wth)
def gradient(x, y, w):
    p = sigmoid_func(x @ w)
    return x.T @ (p - y) / len(y)

#prediction function
#raw number
def predict_probability(x, w):
    return sigmoid_func(x @ w)
#into da yes/no
def predict(x, w, threshold=0.5):
    return (predict_probability(x, w) >= threshold).astype(int)


###TRAINING 
learning_rate = float(input("What learing rate do u want? (ex. 0.5): "))
n_iterations = int(input("How many epochs(iterations)? (ex. 5000): "))

#epoch is funny to say
for epoch in range(n_iterations):
    z = X_train @ weights
    p = sigmoid_func(z)
    loss_history.append(binary_cross_entropy_loss_func(Y_train, p))
    #weights -= learning_rate * (X_train.T @ (p - Y_train) / len(Y_train))
    weights -= learning_rate * gradient(X_train,Y_train,weights)

#updating the losshistory again
p_final = sigmoid_func(X_train @ weights)
loss_history.append(binary_cross_entropy_loss_func(Y_train, p_final))


##eVALUATE
def count_outcomes(y_true, y_pred):
    #which i guessed correctly as true
    correct_hits = np.sum((y_true == 1) & (y_pred == 1))
    #what i guessd correctsly as 0
    correct_passes = np.sum((y_true == 0) & (y_pred == 0))
    #which ones I guess true but were false
    false_alarms = np.sum((y_true == 0) & (y_pred == 1))
    #which ones I guessed false but were true
    missed_ones = np.sum((y_true == 1) & (y_pred == 0))
    return correct_hits, correct_passes, false_alarms, missed_ones

def score_model(y_true, y_pred):
    #getting the correct_hits, correct_passes, false_alarms, missed_ones as ch, cp, fa, mo
    ch, cp, fa, mo = count_outcomes(y_true, y_pred)
    #what fraction did I get right
    accuracy  = (ch + cp) / len(y_true)
    #of what i called good how many actuallt were
    precision = ch / (ch + fa) if (ch + fa) else 0.0
    #out of the correct ones how many i catched
    rec  = ch / (ch + mo) if (ch + mo) else 0.0
    #harmonic mean ig
    f1   = 2 * precision * rec / (precision + rec) if (precision + rec) else 0.0
    return accuracy, precision, rec, f1

majority = 1 if Y_train.mean() > 0.5 else 0
baseline_acc = np.mean(Y_test == majority)


#showinf resultsss
Y_train_pred = predict(X_train, weights)
Y_test_pred  = predict(X_test, weights)

train_metrics = score_model(Y_train, Y_train_pred)
test_metrics  = score_model(Y_test, Y_test_pred)

print(f"{'':10} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print(f"{'Train':10} " + " ".join(f"{v:6.3f}" for v in train_metrics))
print(f"{'Test':10} " + " ".join(f"{v:6.3f}" for v in test_metrics))
print(f"\nBaseline (majority class) test acc: {baseline_acc:.3f}")
print(f"Train n = {len(Y_train)}, Test n = {len(Y_test)}")
print(f"Initial loss: {loss_history[0]:.4f}  Final loss: {loss_history[-1]:.4f}")

####ploting curves
#loss curve
plt.plot(loss_history)
plt.title("BCE loss over training")   # first plot
plt.xlabel("Epoch"); plt.ylabel("BCE loss")
plt.show()

#weight compared to each other using bar
plt.barh(features, weights[:-1])
plt.axvline(0, color='k', lw=0.8); plt.xlabel("Weight"); plt.title("Learned feature weights")  # second
plt.show()


print(np.abs(weights).max())



