#IMPORT THE LIB againnnnnnnnnn

import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split




################################the setup#################################################################################################################
SEED = 16
BATCH_SIZE = 64        # examples per gradient step for stochastic gd
LEARNING_RATE = 0.1    # step size for SGD
EPOCHS = 10            # full passes over the 55,000 training images
VAL_SIZE = 5000        # held out from the 60k training set to watch for overfitting


###########data file directry###############################################################################################################################
DATA_DIR = Path(__file__).parent.parent / "Datasets"

# MNIST's pixel mean and standard deviation (after scaling pixels to 0..1).
# These two numbers are standard and computed on the 60k training images.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


torch.manual_seed(SEED)   # same weights + same shuffling every run
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##READING FILESSSSSSSS files are idx (binary) and not png or jpeg######################################################################################
def read_idx_images(path):
    with open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        # Check the magic number, so a swapped image/label file fails loudly here
        # instead of silently training on garbage.
        assert magic == 2051, f"{path.name} is not an IDX image file (magic {magic})"
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
    return pixels.reshape(count, rows * cols) 


def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"{path.name} is not an IDX label file (magic {magic})"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels                               


###########DATA SET##############
#you ony need len and getitem dataloader does the rest. I think
class MNISTFromIDX(Dataset):
    def __init__(self, images_path, labels_path):
        images = read_idx_images(images_path)
        labels = read_idx_labels(labels_path)
        assert len(images) == len(labels), "image and label files do not match"

        x = torch.from_numpy(images.copy()).float() / 255.0
        self.images = (x - MNIST_MEAN) / MNIST_STD           

        self.labels = torch.from_numpy(labels.copy()).long() 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]   


def make_loaders():
    full_train = MNISTFromIDX(DATA_DIR / "train-images.idx3-ubyte",
                              DATA_DIR / "train-labels.idx1-ubyte")
    test_set = MNISTFromIDX(DATA_DIR / "t10k-images.idx3-ubyte",
                            DATA_DIR / "t10k-labels.idx1-ubyte")

    # 55k train and 5k validation. 
    split_gen = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full_train,
                                      [len(full_train) - VAL_SIZE, VAL_SIZE],
                                      generator=split_gen)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, val_loader, test_loader

######################3The networkkk#######################33##########################################################################3
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)   # W1 (128, 784), b1 (128,)
        self.fc2 = nn.Linear(128, 64)    # W2 (64, 128),  b2 (64,)
        self.fc3 = nn.Linear(64, 10)     # W3 (10, 64),   b3 (10,)
        self.relu = nn.ReLU()

    def forward(self, A0):                  # A0: (B, 784), B is batch size
        Z1 = self.fc1(A0)                # A0 X W1.T + b1 -> (B, 128)
        A1 = self.relu(Z1)                  # max(0, Z1)     -> (B, 128)

        Z2 = self.fc2(A1)                # A1 X W2.T + b2 -> (B, 64)
        A2 = self.relu(Z2)                  #                -> (B, 64)

        Z3 = self.fc3(A2)                # A2 X W3.T + b3 -> (B, 10)

        #no need for extra softmax since crossentropyloss applies softmax itself 
        return Z3

#####################One Epoch of Training##############################33###################################################################
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()   # training mode 
    total_loss, correct, seen = 0.0, 0, 0

    for A0, Y in loader:                       # A0: (B, 784), Y: (B,)
        A0, Y = A0.to(DEVICE), Y.to(DEVICE)

        Z3 = model(A0)                         
        loss = loss_fn(Z3, Y)                  

        optimizer.zero_grad()                  
        
        loss.backward()                        

        optimizer.step()                   

        total_loss += loss.item() * len(Y)     #
        correct += (Z3.argmax(dim=1) == Y).sum().item()   
        seen += len(Y)

    return total_loss / seen, correct / seen


##########################EVALLLLLLLLLLLluation###################################################################################################
@torch.no_grad()   

def evaluate(model, loader, loss_fn):
    model.eval()   # evaluation mode
    total_loss, correct, seen = 0.0, 0, 0
    for A0, Y in loader:
        A0, Y = A0.to(DEVICE), Y.to(DEVICE)
        Z3 = model(A0)
        total_loss += loss_fn(Z3, Y).item() * len(Y)
   
        correct += (Z3.argmax(dim=1) == Y).sum().item()
        seen += len(Y)
    return total_loss / seen, correct / seen


#######################PLOTINGGGG##########################################################################################################3
def plot_curves(history, filename="mnist_training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_loss.plot(epochs, history["train_loss"], "o-", label="train")
    ax_loss.plot(epochs, history["val_loss"], "o-", label="validation")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, [100 * a for a in history["train_acc"]], "o-", label="train")
    ax_acc.plot(epochs, [100 * a for a in history["val_acc"]], "o-", label="validation")
    ax_acc.axhline(97, color="gray", linestyle="--", linewidth=1, label="97% target")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)


    fig.suptitle("784-128-64-10 MLP on MNIST (SGD)")
    fig.tight_layout()
    #fig.savefig(filename, dpi=150)  
    plt.show()


@torch.no_grad()
def plot_mistakes(model, test_set, filename="mnist_mistakes.png", n=12):

    model.eval()
    A0 = test_set.images.to(DEVICE)
    Y = test_set.labels.to(DEVICE)
    probs = torch.softmax(model(A0), dim=1)  
    preds = probs.argmax(dim=1)
    wrong = (preds != Y).nonzero().flatten()[:n].cpu()

    fig, axes = plt.subplots(2, n // 2, figsize=(1.8 * n // 2, 4.4))
    for ax, i in zip(axes.flat, wrong):
        img = test_set.images[i] * MNIST_STD + MNIST_MEAN 
        ax.imshow(img.reshape(28, 28), cmap="gray_r")
        ax.set_title(f"true {Y[i].item()}  pred {preds[i].item()}\n"
                     f"({100 * probs[i, preds[i]].item():.0f}% sure)", fontsize=9)
        ax.axis("off")
    fig.suptitle("Test images the network got wrong")
    fig.tight_layout()
    #fig.savefig(filename, dpi=150)
    plt.show()


####MAINNNNNNNNNNNN##############################################################################################################################
if __name__ == "__main__":   
    train_loader, val_loader, test_loader = make_loaders()
    print(f"device: {DEVICE}")
    print(f"train {len(train_loader.dataset)}  val {len(val_loader.dataset)}  "
          f"test {len(test_loader.dataset)}")

    # Peek at one batch so the shapes are visible.
    A0, Y = next(iter(train_loader))
    print(f"one batch: images {tuple(A0.shape)} {A0.dtype}, labels {tuple(Y.shape)} {Y.dtype}")

    model = MLP().to(DEVICE)
    print(model)
    for name, p in model.named_parameters():
        print(f"  {name:<14} {tuple(p.shape)}")
    print(f"  total parameters: {sum(p.numel() for p in model.parameters()):,}")


    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    print("\n=== Training ===")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn)
        for key, value in zip(history, [tr_loss, tr_acc, va_loss, va_acc]):
            history[key].append(value)
        print(f"epoch {epoch:>2}  train loss {tr_loss:.4f} acc {100 * tr_acc:.2f}%   "
              f"val loss {va_loss:.4f} acc {100 * va_acc:.2f}%")

    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"\n=== TEST ACCURACY: {100 * test_acc:.2f}%  (loss {test_loss:.4f}) ===")
    print("target >= 97%:", "MET" if test_acc >= 0.97 else "NOT MET")

    plot_curves(history)
    plot_mistakes(model, test_loader.dataset)