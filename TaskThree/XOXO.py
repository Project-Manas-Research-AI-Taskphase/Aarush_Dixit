#GONNE make a 2-4-1 MLP. Uh yeah
#immma use some naming conventions
# w^l_jk is wl_j_k altho we wont need indvidiual elemnts in actual training (where j is target and k is incoming)
# Wl and bl for the matricies

# We need da libraries
import numpy as np
from matplotlib import pyplot as plt


###########################This is to turn on/off the checkgrads functions thats primarly used to see if the back prop is correct Turning it off saves computing power when you know you did it right
RUN_GRAD_CHECK = True


# now wee neeed the data but its only 4 freaking things
# each collumn is one examples top one goes to top neouron and bottom one goes to bottom input neuron
X = np.array([[0, 0, 1, 1],
             [0, 1, 0, 1]], 
             dtype=np.float64)

# Output
y = np.array([[0, 1, 1, 0]], dtype=np.float64)

#activation function
def sigmoid(z):
    out = 1.0 / (1.0 + np.exp(-z))
    return out

def sigmoid_derivative(a):
    #sigmid' is just sigmoid(1-sigmoid)
    #we will already have a from sigmoid
    out = a * (1 - a)
    return out

#initalising paramters
def init_params(seed=16):
    # Seed gives same resul every use
    rng = np.random.default_rng(seed)

    #dividing by root2 so random numbers dont saturate sigmoid
    W1 = rng.standard_normal((4, 2)) / np.sqrt(2)
    b1 = np.zeros((4, 1))

    W2 = rng.standard_normal((1, 4)) / np.sqrt(4)
    b2 = np.zeros((1, 1))

    #weights need to be random to stop all the nerons from getting the same gradient forever. and bias can be 0 since weights are already diff.
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

#DA FORWAD PASSS
def forward(params, A0):
    #layer 1
    Z1 = params["W1"] @ A0 + params["b1"] # 4x2 X 2xN + 4x1 -> 4xN
    A1 = sigmoid(Z1) # 4xN

    #layer2
    Z2 = params["W2"] @ A1 + params["b2"] # 1x4 X 4xN + 1x1 -> 1xN
    A2 = sigmoid(Z2) # 1xN output

    # Savign the activations to compute gradients.
    activations_cache = {"A0": A0, "A1": A1, "A2": A2}
    return A2, activations_cache

#LOSS FUNCTIONNNNNNNNNNNNNNNNNNNN
def cross_entropy_loss( A2, Y):
    #   L = -(1/N) * sum[ y*log(a) + (1-y)*log(1-a) ]
    #we need the eps and clip cos log0 = -infiity and that can break the model. A confidently right or wrong pred could break the model
    eps = 1e-12
    A2 = np.clip(A2,eps, 1-eps)
    out = -np.mean( Y * np.log(A2) + (1-Y) * np.log(1-A2) )
    return out

#DA BACKWORD PASSS V IMP
def backward(params, cache, Y):
    #take the activatios out of cache for use
    A0, A1, A2 = cache["A0"], cache["A1"], cache["A2"]
    #the number of examples use for mean.
    N = Y.shape[1]

    #output error
    Delta2 = (A2 - Y) / N

    dW2 = Delta2 @ A1.T # 1xN X Nx4 -> 1x4
    db2 = np.sum(Delta2, axis=1, keepdims=True) # 1x1

    #hidden layer error
    #uses the BP2 from neilson's book
    Delta1 = (params["W2"].T @ Delta2) * sigmoid_derivative(A1) # 4xN

    dW1 = Delta1 @ A0.T # 4xN X Nx2 -> 4x2
    db1 = np.sum(Delta1, axis=1, keepdims=True) #4x1

    # every gradient has same shape as its parameter.
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

#this puts my naming schme in use entirely not needed.
def param_name(key, idx):
    layer_no = key[1:]
    #cos weights and bases named differently.
    if key[0] == "W":
        j, k = idx
        return f"w{layer_no}_{j}_{k}"
    return f"b{layer_no}_{idx[0]}"

#gradient CHECK: to check if the backprop is getting the right answers for derivatobes
def gradients_check(params, X, Y, eps = 1e-5):
   
    _, cache = forward(params, X)
    analytic_grads = backward(params, cache, Y)

    print(f"{'param':<10}{'analytic':>15}{'numeric':>15}{'rel. error':>14}")

    worst = 0.0     
    for key in ["W1", "b1", "W2", "b2"]:
        P = params[key]                     
        for idx in np.ndindex(P.shape):    
            original = P[idx]

            P[idx] = original + eps
            loss_plus = cross_entropy_loss(forward(params, X)[0], Y)

            P[idx] = original - eps
            loss_minus = cross_entropy_loss(forward(params, X)[0], Y)

            P[idx] = original               

            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = analytic_grads["d" + key][idx]

            denom = max(abs(numeric), abs(analytic), 1e-12)
            rel_err = abs(analytic - numeric) / denom
            worst = max(worst, rel_err)

            print(f"{param_name(key, idx):<10}{analytic:>15.8f}{numeric:>15.8f}{rel_err:>14.2e}")

    verdict = "PASSED" if worst < 1e-7 else "FAILED"
    print(f"\nWorst relative error: {worst:.2e}  ->  gradient check {verdict}\n")
    return worst

######################THE TRAINING DUN DUN DUNNNNNNNNNN###################3
def train(params, X, Y, lr = 1, epochs = 10000):
    for epoch in range(epochs + 1):
        A2, cache = forward(params, X)  # Predict
        loss = cross_entropy_loss(A2, Y)    # Measure lossss    
        grads = backward(params, cache, Y)  # measure grad for everything

        for key in params:
            params[key] -= lr * grads["d" + key]

        if epoch % 1000 == 0:
            print(f"epoch {epoch:>5}  loss {loss:.6f}")
        
    return params

##############PLOT##########
def plot_decision_boundary(params, X, Y, filename="xor_decision_boundary.png"):

    #make a grid of (x0,x1)
    xs = np.linspace(-0.5, 1.5, 300)
    xx, yy = np.meshgrid(xs, xs)

    # Stack the grid into columns so it is the same as X
    grid = np.vstack([xx.ravel(), yy.ravel()])
    probs, _ = forward(params, grid)
    probs = probs.reshape(xx.shape)   # back to a 300 x 300 image

    plt.figure(figsize=(6, 5))
    # Colour = predicted probability that XOR = 1 at that point.
    filled = plt.contourf(xx, yy, probs, levels=75, cmap="RdBu_r", alpha=0.8)
    plt.colorbar(filled, label="Predicted P(XOR = 1)")
    # The decision boundary is exactly where the network is 50/50.
    plt.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)
    # The 4 training points on top, coloured by their true label.
    plt.scatter(X[0], X[1], c=Y[0], cmap="RdBu_r", edgecolors="black", s=200, zorder=3)

    plt.title("Learned decision boundary of the 2-4-1 MLP on XOR")
    plt.xlabel("input neuron 0")
    plt.ylabel("input neuron 1")
    plt.tight_layout()
    #plt.savefig(filename, dpi=150)   
    plt.show()  

#RUNNING THE ACTUAL CODEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
if __name__ == "__main__":
    s = input("Enter random seed (Entering diff ones show different cool descision boundaries, ALso only put from R+)(blank = 16, the one in the pic I put16): ").strip()
    seed = int(s) if s else 16
    params = init_params(seed=seed)

    if RUN_GRAD_CHECK:
        print("=== Gradient check (before training) ===")
        gradients_check(params, X, y)

    print("=== Training ===")
    params = train(params, X, y, lr=1.0, epochs=10000)

    print("\n=== Predictions ===")
    A2, _ = forward(params, X)
    for i in range(X.shape[1]):
        print(f"input ({int(X[0, i])}, {int(X[1, i])})  target {int(y[0, i])}  "
              f"output {A2[0, i]:.4f}  predicted {int(A2[0, i] > 0.5)}")

    print("\n=== Learned parameters ===")
    for key in ["W1", "b1", "W2", "b2"]:
        for idx in np.ndindex(params[key].shape):
            print(f"{param_name(key, idx):<8} = {params[key][idx]: .4f}")

    plot_decision_boundary(params, X, y)
