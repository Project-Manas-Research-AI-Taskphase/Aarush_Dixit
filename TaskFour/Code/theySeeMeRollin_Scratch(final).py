# IMPOrt LIB
import numpy as np
import matplotlib.pyplot as plt
import time
import json, os, math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


##################################### SETTINGS #####################################
#keep one line uncommented in each thing

#OPTIMIZER = "SGD"
#OPTIMIZER = "MOM"
OPTIMIZER = "ADAM"
#OPTIMIZER = "ADAMW"

#LEarning rate (Try 0.01 for Momentum, 0.001 for adam/adamW)
LR = {"SGD": 0.01, "MOM": 0.1, "ADAM": 0.001, "ADAMW": 0.001}[OPTIMIZER]

#SCHEDULER = "COSINE"
# SCHEDULER = "STEP"
SCHEDULER = "NONE"

MOMENTUM   = 0.9       #used by MOM
STEP_EVERY = 5         #STEPLR - drop the lr every 5 epochs
GAMMA      = 0.1       #by this factor

WD_ON = False
WD = {"SGD": 5e-4, "MOM": 5e-4, "ADAM": 5e-4, "ADAMW": 0.01}[OPTIMIZER]
if not WD_ON:
    WD = 0.0

P  = 0               #dropout: chance of dropping a neuron, 0 = off
BATCHNORM = False       #False = off

EPOCHS = 19
SEEDS  = [0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#turn on to compare the test set
EVAL_TEST = False
TRAIN_EVAL_SIZE = 5000

################################loading data sets and setting up######################################

# Classes: airplane=0, automobile=1, bird=2, cat=3, deer=4, dog=5, frog=6, horse=7, ship=8, truck=9
# Raw data: (50000, 32, 32, 3) uint8 0–255. After ToTensor each image is (3, 32, 32) float 0–1.

data_path = "TaskFour/Datasets"
transform = transforms.ToTensor()

#LOAD DATASET
cifar10_train_full = datasets.CIFAR10(data_path, train=True,  download=True, transform=transform)
cifar10_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
classes = cifar10_train_full.classes

#set generator seed
g = torch.Generator().manual_seed(16)

#getting training and val set
cifar10_train, cifar10_val = random_split(cifar10_train_full, [45000, 5000], generator=g)

#normalise each colour channel to mean 0, std 1 using ONLY the 45k training images
#using val/test pixels here would leak info about them
idx = np.array(cifar10_train.indices)
mean = [float((cifar10_train_full.data[idx, :, :, c] / 255.0).mean()) for c in range(3)]
std  = [float((cifar10_train_full.data[idx, :, :, c] / 255.0).std())  for c in range(3)]

#normalizeing func
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

cifar10_train_full.transform = transform
cifar10_test.transform = transform

g_te = torch.Generator().manual_seed(16)
train_eval_idx = torch.randperm(len(cifar10_train), generator=g_te)[:TRAIN_EVAL_SIZE].tolist()
cifar10_train_eval = Subset(cifar10_train, train_eval_idx)

#set global seed
torch.manual_seed(19)

#loading batches for each thng
train_loader      = DataLoader(cifar10_train,      batch_size=128, shuffle=True)
train_eval_loader = DataLoader(cifar10_train_eval, batch_size=500, shuffle=False)
val_loader        = DataLoader(cifar10_val,        batch_size=500, shuffle=False)
test_loader       = DataLoader(cifar10_test,       batch_size=500, shuffle=False)

################################## SCHEDULER ##################################

def get_lr(epoch):
    if SCHEDULER == "STEP":
        return LR * GAMMA ** (epoch // STEP_EVERY)
    if SCHEDULER == "COSINE":
        return 0.5 * LR * (1 + math.cos(math.pi * epoch / EPOCHS))
    return LR

##################################DROPOUT + BATCHNORM######################

def dropout(x, p, training):
    if not training or p == 0:
        return x
    keep = 1 - p
    mask = (torch.rand_like(x) < keep).float()     # 1 = keep, 0 = drop
    return x * mask / keep


class BatchNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n))
        self.beta  = nn.Parameter(torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var",  torch.ones(n))

    def forward(self, x):
        if self.training:
            mean = x.mean(0)
            var  = ((x - mean) ** 2).mean(0) #biased var: still used to normalise THIS batch
            n = x.size(0)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean.detach()
            self.running_var  = 0.9 * self.running_var  + 0.1 * var.detach() * n / (n - 1) #unbiased for test time
        else:
            mean, var = self.running_mean, self.running_var

        x_hat = (x - mean) / torch.sqrt(var + 1e-5)
        return self.gamma * x_hat + self.beta


#################################################3 BUILDING THE MODEL##########################
#3072-1024-512-128-10
#each hidden layer: linear -> batchnorm -> relu -> dropout

class multiLayerPerceptron(nn.Module):

    def __init__(self, input_size = 32*32*3, output_size = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, output_size)

        if BATCHNORM:
            self.bn1 = BatchNorm(1024)
            self.bn2 = BatchNorm(512)
            self.bn3 = BatchNorm(128)

    def hidden(self, X, fc, bn):
        X = fc(X)
        if BATCHNORM:
            X = bn(X)
        X = F.relu(X)
        X = dropout(X, P, self.training)
        return X

    def forward(self, X):
        X = X.view(X.size(0), -1)                       #(B,3,32,32) -> (B,3072)
        X = self.hidden(X, self.fc1, self.bn1 if BATCHNORM else None)
        X = self.hidden(X, self.fc2, self.bn2 if BATCHNORM else None)
        X = self.hidden(X, self.fc3, self.bn3 if BATCHNORM else None)
        return self.fc4(X) # not F.log_softmax(self.fc4(X), dim=1) cos Cross-Entropy does inbuild softmax



##############################TRAINING#########################################################################################################

#RUNNING THE TRAINING
def run(epochs, seed):
    torch.manual_seed(seed)
    model = multiLayerPerceptron().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    params   = list(model.parameters())
    velocity = [torch.zeros_like(p) for p in params]   #MOM
    m        = [torch.zeros_like(p) for p in params]   #ADAM: avg gradient
    v        = [torch.zeros_like(p) for p in params]   #ADAM: avg squared gradient
    t = 0                                              #ADAM: step counter
    b1, b2, eps = 0.9, 0.999, 1e-8

    train_losses, train_acc = [], []
    train_run_losses, train_run_acc = [], []
    val_losses, val_acc = [], []
    lrs = []

    best_val_acc, best_epoch, best_state = -1.0, 0, None

    start_time = time.time()

    for i in range(epochs):
        lr = get_lr(i)
        lrs.append(lr)
        model.train()

        trn_corr, trn_loss_sum = 0, 0.0

        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            model.zero_grad()
            loss.backward()

            #optimizer steps
            t += 1
            with torch.no_grad():
                for k, p in enumerate(params):
                    if OPTIMIZER == "ADAMW":
                        g = p.grad
                        p -= lr * WD * p
                    else:
                        g = p.grad + WD * p

                    if OPTIMIZER == "SGD":
                        p -= lr * g

                    elif OPTIMIZER == "MOM":
                        velocity[k] = MOMENTUM * velocity[k] + g
                        p -= lr * velocity[k]

                    elif OPTIMIZER  in ("ADAM", "ADAMW"):
                        m[k] = b1 * m[k] + (1 - b1) * g
                        v[k] = b2 * v[k] + (1 - b2) * g * g
                        m_hat = m[k] / (1 - b1 ** t)
                        v_hat = v[k] / (1 - b2 ** t)
                        p -= lr * m_hat / (v_hat.sqrt() + eps)


            trn_corr     += (y_pred.argmax(1) == y_train).sum().item()
            trn_loss_sum += loss.item() * y_train.size(0)

        train_run_losses.append(trn_loss_sum / len(cifar10_train))
        train_run_acc.append(trn_corr * 100 / len(cifar10_train))

        tr_loss, tr_corr = evaluate(model, train_eval_loader)
        train_losses.append(tr_loss)
        train_acc.append(tr_corr * 100 / len(cifar10_train_eval))

        vl_loss, vl_corr = evaluate(model, val_loader)
        val_losses.append(vl_loss)
        val_acc.append(vl_corr * 100 / len(cifar10_val))

        if val_acc[-1] > best_val_acc:
            best_val_acc, best_epoch = val_acc[-1], i + 1 #epochs counted from 1
            best_state = {k: t_.detach().clone() for k, t_ in model.state_dict().items()}

        print(f'epoch {i+1} | lr {lr:.5f} | train loss {train_losses[-1]:.4f} (running {train_run_losses[-1]:.4f}) '
              f'| val loss {val_losses[-1]:.4f} | train acc {train_acc[-1]:.2f}% | val acc {val_acc[-1]:.2f}%')

    print(f'Duration: {(time.time() - start_time)/60:.2f} mins')

    #hand back the best-val model, not the last-epoch one,
    #so best val acc, test acc and the confusion matrix all describe the SAME weights
    model.load_state_dict(best_state)
    print(f'restored best epoch {best_epoch} (val acc {best_val_acc:.2f}%)')

    history = {
        "train_losses": train_losses,
        "train_acc": train_acc,
        "train_run_losses": train_run_losses,
        "train_run_acc": train_run_acc,
        "val_losses": val_losses,
        "val_acc": val_acc,
        "lrs": lrs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }
    return model, history

#Eval
def evaluate(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss_sum += criterion(out, y).item() * y.size(0)
            correct  += (out.argmax(1) == y).sum().item()
    return loss_sum / len(loader.dataset), correct


################################ EVALUATION ################################
def test_eval(model):
    test_loss, test_corr = evaluate(model, test_loader)
    test_acc = test_corr * 100 / len(cifar10_test)
    print(f'TEST | loss {test_loss:.4f} | accuracy {test_acc:.2f}%')
    return test_loss, test_acc

def confusion(model, loader, title="Validation confusion matrix"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            out = model(X.to(DEVICE))
            all_preds.append(out.argmax(1).cpu())
            all_labels.append(y)

    cm = confusion_matrix(torch.cat(all_labels), torch.cat(all_preds))   # row = true, col = predicted
    print(cm)

    ConfusionMatrixDisplay(cm, display_labels=classes).plot(xticks_rotation=45, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.show()

################################ PLOTTING DA CURVY################################

def plot_seeds(results, name):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.5))
    ep = range(1, EPOCHS + 1)

    for r in results:
        seed, h = r["seed"], r["history"]
        be = h["best_epoch"]

        line, = ax1.plot(ep, h["val_losses"], label=f"seed {seed}") # solid = val
        c = line.get_color() # same colour per seed
        ax1.plot(ep, h["train_losses"], "--", color=c, alpha=0.7) # dashed = train (eval mode)
        ax1.plot(ep, h["train_run_losses"], ":", color=c, alpha=0.4) # dotted = running train (train mode)
        ax1.plot(be, h["val_losses"][be - 1], "o", color=c, markersize=8, markeredgecolor="k")

        label = f"seed {seed} (best val {h['best_val_acc']:.2f}% @ ep {be})"
        ax2.plot(ep, h["val_acc"], color=c, label=label)
        ax2.plot(ep, h["train_acc"], "--", color=c, alpha=0.7)
        ax2.plot(ep, h["train_run_acc"], ":", color=c, alpha=0.4)
        ax2.plot(be, h["best_val_acc"], "o", color=c, markersize=8, markeredgecolor="k")

        # test is only drawn on the final run, at the epoch whose weights were tested
        if r["test_acc"] is not None:
            ax1.plot(be, r["test_loss"], "*", color=c, markersize=14, markeredgecolor="k")
            ax2.plot(be, r["test_acc"], "*", color=c, markersize=14, markeredgecolor="k")

    ax3.plot(ep, results[0]["history"]["lrs"], marker="o")

    key = "solid = val, dashed = train (eval mode), dotted = running train, ● = best val"
    if EVAL_TEST:
        key += ", ★ = test"
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    ax3.set(xlabel="Epoch", ylabel="Learning rate", title=f"LR schedule ({SCHEDULER})")
    ax1.legend(); ax2.legend()
    fig.suptitle(f"{name}\n{key}")
    fig.tight_layout()
    fig.savefig(f"results/{name}.png", dpi=150)
    plt.show()

################################MAIN ###############################

if __name__ == "__main__":

    print(f"{OPTIMIZER} lr {LR} | {SCHEDULER} | wd {WD} | dropout {P} | batchnorm {BATCHNORM} | {DEVICE} | test {EVAL_TEST}")
    results = []

    for SEED in SEEDS:
        print(f"\n========== seed {SEED} ==========")
        model, history = run(EPOCHS, SEED)

        test_loss, test_acc = test_eval(model) if EVAL_TEST else (None, None)
        results.append({"seed": SEED, "history": history, "test_loss": test_loss, "test_acc": test_acc})

    # summary across seeds (all numbers below come from validation unless EVAL TEST is on)
    best_val = [r["history"]["best_val_acc"] for r in results]
    best_ep  = [r["history"]["best_epoch"] for r in results]
    gap      = [r["history"]["val_losses"][-1] - r["history"]["train_losses"][-1] for r in results]
    print(f"\nbest val acc:             {np.mean(best_val):.2f} ± {np.std(best_val):.2f}%  (epochs {best_ep})")
    print(f"final val - train loss:   {np.mean(gap):.4f} ± {np.std(gap):.4f}  (both eval mode)")
    if EVAL_TEST:
        tests = [r["test_acc"] for r in results]
        print(f"test acc (best-val ckpt): {np.mean(tests):.2f} ± {np.std(tests):.2f}%")

    name = f"{OPTIMIZER}_lr{LR}_{SCHEDULER}_wd{WD}_p{P}_bn{BATCHNORM}_ep{EPOCHS}" + ("_TEST" if EVAL_TEST else "")
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_seeds(results, name)
    confusion(model, val_loader, title=f"Val confusion ({name}, seed {SEEDS[-1]}, best epoch)")
