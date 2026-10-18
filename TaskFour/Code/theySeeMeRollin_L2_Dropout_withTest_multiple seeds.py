# IMPOrt LIB
import numpy as np
import matplotlib.pyplot as plt
import time
import json, os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


################################loading data sets and setting up######################################

# Classes: airplane=0, automobile=1, bird=2, cat=3, deer=4, dog=5, frog=6, horse=7, ship=8, truck=9
# Raw data: (50000, 32, 32, 3) uint8 0–255. After ToTensor each image is (3, 32, 32) float 0–1.

data_path = "TaskFour/Datasets"
transform = transforms.ToTensor()

cifar10_train_full = datasets.CIFAR10(data_path, train=True,  download=False, transform=transform)
cifar10_test = datasets.CIFAR10(data_path, train=False, download=False, transform=transform)
classes = cifar10_train_full.classes

#set generator seed
g = torch.Generator().manual_seed(16)
#getting training and val set
cifar10_train, cifar10_val = random_split(cifar10_train_full, [45000, 5000], generator=g)

#set global seed
torch.manual_seed(19)

#loading batches for each thng
train_loader = DataLoader(cifar10_train, batch_size=128, shuffle=True)
val_loader   = DataLoader(cifar10_val,   batch_size=500, shuffle=False)
test_loader  = DataLoader(cifar10_test,  batch_size=500, shuffle=False)


'''
images type:  <class 'numpy.ndarray'>
images shape: (50000, 32, 32, 3)
images dtype: uint8
images min/max: 0 255
labels type:  <class 'list'>
labels count: 50000
first 10 labels: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3]
classes: ['airplane' = 0, 'automobile' = 1, 'bird' = 2, 'cat'= 3, 'deer' = 4, 'dog'= 5, 'frog' = 6, 'horse' = 7, 'ship' = 8, 'truck' = 9]

Training:   45000
Validation: 5000
Testing:    10000
torch.Size([128, 3, 32, 32]) torch.Size([128])
'''

#################################################3 BUILDING THE MODEL##########################

#imma build 3072-1024-512-128-10 
#parameters = 3738506
#3145728 1024 524288 512 65536 128 1280 10 3738506

class multiLayerPerceptron(nn.Module):

    def __init__(self, input_size = 32*32*3, output_size = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(p=P)

    def forward(self, X):
         
        X = X.view(X.size(0), -1)  
        X = self.dropout(F.relu(self.fc1(X)))
        X = self.dropout(F.relu(self.fc2(X)))
        X = self.dropout(F.relu(self.fc3(X)))
        X = self.fc4(X)

        return F.log_softmax(X, dim=1)


##############################TRAINING#######################################


def run(epochs, lr, optim, momen, wd, seed):
    torch.manual_seed(seed)     # same starting weights + shuffle order every run
    model = multiLayerPerceptron()

    criterion = nn.CrossEntropyLoss()
    if(optim == "SGD" or optim == "MOM"):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momen, weight_decay=wd)
    elif(optim == "ADAM"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)



    train_losses, val_losses, test_losses = [], [], []
    train_correct, val_correct, test_correct = [], [], []

    start_time = time.time()

    for i in range(epochs):
        model.train()

        trn_corr = 0
        trn_seen = 0 # images actually seen
        trn_loss_sum = 0.0 # running loss total for the epoch

        for b_iter, (X_train, y_train) in enumerate(train_loader, start=1):
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            predicted = torch.max(y_pred.data, 1)[1]
            trn_corr += (predicted == y_train).sum().item()
            trn_seen += y_train.size(0)
            trn_loss_sum += loss.item() * y_train.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b_iter % 100 == 0:
                accuracy = trn_corr * 100 / trn_seen
                print(f'epoch: {i} batch {b_iter} loss:{loss.item():.4f} accuracy:{accuracy:.2f}%')

        train_losses.append(trn_loss_sum / trn_seen) # whole-epoch average
        train_correct.append(trn_corr)

        val_corr = 0
        val_loss_sum = 0.0

        model.eval()
        with torch.no_grad():
            for X_val, y_val in val_loader:
                out = model(X_val)
                val_loss_sum += criterion(out, y_val).item() * y_val.size(0)
                predicted = torch.max(out.data, 1)[1]
                val_corr += (predicted == y_val).sum().item()

        val_losses.append(val_loss_sum / len(cifar10_val))
        val_correct.append(val_corr)

        print(f'epoch {i} | train loss {train_losses[-1]:.4f} | val loss {val_losses[-1]:.4f} '
              f'| train acc {trn_corr*100/trn_seen:.2f}% | val acc {val_corr*100/len(cifar10_val):.2f}%')

    total_time = time.time() - start_time
    print(f'Duration: {total_time/60:.2f} mins')

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_correct": train_correct,
        "val_correct": val_correct,
    }
    return model, history
    
################################ EVALUATION ################################

def test_eval(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_corr, test_loss_sum = 0, 0.0

    with torch.no_grad():
        for X_test, y_test in test_loader:
            out = model(X_test)
            test_loss_sum += criterion(out, y_test).item() * y_test.size(0)
            predicted = torch.max(out.data, 1)[1]
            test_corr += (predicted == y_test).sum().item()

    test_loss = test_loss_sum / len(cifar10_test)
    test_acc = test_corr * 100 / len(cifar10_test)
    print(f'TEST | loss {test_loss:.4f} | accuracy {test_acc:.2f}% ({test_corr}/{len(cifar10_test)})')
    return test_loss, test_acc

def confusion(model, loader, title="Validation confusion matrix"):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            out = model(X)
            predicted = torch.max(out, 1)[1]
            all_preds.append(predicted)
            all_labels.append(y)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds)

    np.set_printoptions(linewidth=150)
    print(np.arange(10).reshape(1, 10)) # column headers: predicted class
    print(cm)

    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.show()


################################ PLOTTING DA CURVY################################

def plot_curves(history, test_loss, test_acc):
    last = len(history["train_losses"]) - 1

    plt.plot(history["train_losses"], label="Training loss")
    plt.plot(history["val_losses"],   label="Validation loss")
    plt.scatter(last, test_loss, marker="*", s=200, color="red", zorder=3,
                label=f"Final test loss ({test_loss:.3f})")              
    plt.xlabel("Epoch"); plt.ylabel("Loss")                              
    plt.legend()
    plt.show()

    train_accuracy = [t * 100 / len(cifar10_train) for t in history["train_correct"]]   
    val_accuracy   = [t * 100 / len(cifar10_val)   for t in history["val_correct"]]    

    plt.plot(train_accuracy, label="Training accuracy")
    plt.plot(val_accuracy,   label="Validation accuracy")
    plt.scatter(last, test_acc, marker="*", s=200, color="red", zorder=3,
                label=f"Final test accuracy ({test_acc:.2f}%)")         
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")                     
    plt.legend()
    plt.show()

def plot_seeds(results, name):
    colors = plt.cm.tab10.colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for k, (seed, history, test_loss, test_acc) in enumerate(results):
        c = colors[k % 10]
        ep = range(1, len(history["train_losses"]) + 1)
        train_acc = [t * 100 / len(cifar10_train) for t in history["train_correct"]]
        val_acc   = [t * 100 / len(cifar10_val)   for t in history["val_correct"]]

        ax1.plot(ep, history["train_losses"], color=c, linestyle="--", alpha=0.6)
        ax1.plot(ep, history["val_losses"],   color=c, label=f"seed {seed}")
        ax2.plot(ep, train_acc, color=c, linestyle="--", alpha=0.6)
        ax2.plot(ep, val_acc,   color=c, label=f"seed {seed} (test {test_acc:.2f}%)")

    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss  (dashed = train, solid = val)")
    ax2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy  (dashed = train, solid = val)")
    ax1.legend(); ax2.legend()
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(f"results/{name}.png", dpi=150)
    plt.show()

################################MAIN ###############################

if __name__ == "__main__":

    OPTIMIZER = input("What optimizer do you want to use? (SGD / MOM / ADAM): ").strip().upper()
    EPOCHS = int(input("Enter number of epochs (12 etc): "))
    LR = float(input("Enter learning rate (0.01 for SGD/MOM, 0.001 for ADAM): "))
    MOMENTUM = float(input("Enter momentum (0.9 etc): ")) if OPTIMIZER == "MOM" else 0
    WD = float(input("Enter Weight Decay (0.00025 etc for SGD/MOM, 0.01 etc for ADAM): "))
    P = float(input("Enter fraction for dropout (0.5) "))
    SEEDS = [int(s) for s in input("Enter seeds separated by spaces (e.g. 0 1 2): ").split()]

    results = []

    for SEED in SEEDS:
        print(f"\n========== seed {SEED} ==========")
        model, history = run(EPOCHS, LR, OPTIMIZER, MOMENTUM, WD, SEED)
        test_loss, test_acc = test_eval(model)
        results.append((SEED, history, test_loss, test_acc))

    # summary across seeds
    final_val = [h["val_correct"][-1] * 100 / len(cifar10_val) for _, h, _, _ in results]
    tests     = [r[3] for r in results]
    print(f"\nfinal val acc: {np.mean(final_val):.2f} ± {np.std(final_val):.2f}%")
    print(f"test acc:      {np.mean(tests):.2f} ± {np.std(tests):.2f}%")


    name = f"{OPTIMIZER}_lr{LR}_wd{WD}_p{P}_ep{EPOCHS}"
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump([{"seed": s, "history": h, "test_loss": tl, "test_acc": ta}
                   for s, h, tl, ta in results], f, indent=2)

    plot_seeds(results, name)
    confusion(model, val_loader, title=f"Val confusion ({name}, seed {SEEDS[-1]})")
