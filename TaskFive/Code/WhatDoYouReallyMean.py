#import di lib
import os
import time
import struct 
import numpy as np 
import matplotlib.pyplot as plt


#SETTINGS
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Datasets")

EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
SEED = 0

#SEED
np.random.seed(SEED)  # all np.random calls below now repeat exactly on every run

#READING THE DATA
def read_idx_images(path):
    with open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"{path} is not an IDX image file (magic {magic})"
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
    return pixels.reshape(count, rows * cols)
 
 
def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"{path} is not an IDX label file (magic {magic})"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

#LOADING THE DATA
def load_mnist():
    x_train = read_idx_images(DATA_DIR + "/train-images.idx3-ubyte")
    y_train = read_idx_labels(DATA_DIR + "/train-labels.idx1-ubyte")
    x_test = read_idx_images(DATA_DIR + "/t10k-images.idx3-ubyte")
    y_test = read_idx_labels(DATA_DIR + "/t10k-labels.idx1-ubyte")
 
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)
 
    x_train = ((x_train / 255.0 - 0.1307) / 0.3081).astype(np.float32)
    x_test = ((x_test / 255.0 - 0.1307) / 0.3081).astype(np.float32)
 
    return x_train, y_train.astype(np.int64), x_test, y_test.astype(np.int64)



######################################LAYERS#######################3

def conv_forward(x, W, b, pad):
    N, C, H, Wd = x.shape # batch size, input channels, height, width
    F, _, k, _ = W.shape # number of filters, kernel size
    H_out = H + 2 * pad - k + 1 # output size formula (W - F + 2P)/S + 1 with S = 1
    W_out = Wd + 2 * pad - k + 1
 
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
 
    out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)  # start the sum at zero
    for i in range(k): # kernel row
        for j in range(k): # kernel column
            patch = x_pad[:, :, i:i + H_out, j:j + W_out]
 
            out += np.einsum("nchw,fc->nfhw", patch, W[:, :, i, j], optimize=True)
 
    out += b.reshape(1, F, 1, 1)  # one bias per filter
    cache = (x_pad, W, pad) # saved for the backward pass
    return out, cache
 
 
def conv_backward(dout, cache):
    x_pad, W, pad = cache
    F, C, k, _ = W.shape
    _, _, H_out, W_out = dout.shape
 
    dx_pad = np.zeros_like(x_pad) # gradient wrt the padded input, filled below
    dW = np.zeros_like(W)
 
    for i in range(k):
        for j in range(k):
            patch = x_pad[:, :, i:i + H_out, j:j + W_out]
 
            dW[:, :, i, j] = np.einsum("nfhw,nchw->fc", dout, patch, optimize=True)
 
            dx_pad[:, :, i:i + H_out, j:j + W_out] += np.einsum(
                "nfhw,fc->nchw", dout, W[:, :, i, j], optimize=True)
 
    db = dout.sum(axis=(0, 2, 3))
 
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
    dx = dx_pad[:, :, pad:H_pad - pad, pad:W_pad - pad]
    return dx, dW, db

###ReLU
def relu_forward(x):
    out = np.maximum(0, x)
    return out, x
 
 
def relu_backward(dout, x):
    return dout * (x > 0)
 
 ###Maxpool
def maxpool_forward(x):
    N, C, H, W = x.shape
 
    blocks = x.reshape(N, C, H // 2, 2, W // 2, 2)
    out = blocks.max(axis=(3, 5))
    return out, (x, out)
 
 
def maxpool_backward(dout, cache):
    x, out = cache
    N, C, H, W = x.shape
 
    out_big = out.repeat(2, axis=2).repeat(2, axis=3)
 
    mask = (x == out_big)
 
    dout_big = dout.repeat(2, axis=2).repeat(2, axis=3)
 
    return dout_big * mask
 
##DULLY CONNECTED LAYER
def fc_forward(x, W, b):
    return x @ W + b, x 
 
def fc_backward(dout, x, W):
    dx = dout @ W.T
    dW = x.T @ dout
    db = dout.sum(axis=0)
    return dx, dW, db
 
###SOFT_MAXPLUS_CROSSENTROPY
def softmax_cross_entropy(logits, y):
    N = logits.shape[0]
 
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
 
    loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
 
    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1
    dlogits /= N
    return loss, dlogits


########################################################MODELLLL#######################

def init_params():
    def he(shape, fan_in):
        return (np.random.randn(*shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)
 
    return {
        #8 filters., 1 chanel * 3 * 3 = 9
        "W1": he((8, 1, 3, 3), 1 * 3 * 3),   "b1": np.zeros(8, dtype=np.float32),
        # 16 filters, 8 cchanels* 3 * 3 = 72
        "W2": he((16, 8, 3, 3), 8 * 3 * 3),  "b2": np.zeros(16, dtype=np.float32),
        #16 channels * 7 * 7 -> 10 outputs
        "W3": he((16 * 7 * 7, 10), 16 * 7 * 7), "b3": np.zeros(10, dtype=np.float32),
    }
 

#THE FORQWARD PASS
def forward(params, x):
    c1, c1_cache = conv_forward(x, params["W1"], params["b1"], pad=1)
    r1, r1_cache = relu_forward(c1)
    p1, p1_cache = maxpool_forward(r1)
 
    c2, c2_cache = conv_forward(p1, params["W2"], params["b2"], pad=1)
    r2, r2_cache = relu_forward(c2)
    p2, p2_cache = maxpool_forward(r2)
 
    flat = p2.reshape(p2.shape[0], -1)
    logits, fc_cache = fc_forward(flat, params["W3"], params["b3"])
 
    caches = (c1_cache, r1_cache, p1_cache, c2_cache, r2_cache, p2_cache, fc_cache, p2.shape)
    return logits, caches
 
#backaward pass -> chainrule
def backward(params, dlogits, caches):
    c1_cache, r1_cache, p1_cache, c2_cache, r2_cache, p2_cache, fc_cache, p2_shape = caches
    grads = {}
 
    dflat, grads["W3"], grads["b3"] = fc_backward(dlogits, fc_cache, params["W3"])
    dp2 = dflat.reshape(p2_shape)
 
    dr2 = maxpool_backward(dp2, p2_cache)
    dc2 = relu_backward(dr2, r2_cache)
    dp1, grads["W2"], grads["b2"] = conv_backward(dc2, c2_cache)
 
    dr1 = maxpool_backward(dp1, p1_cache)
    dc1 = relu_backward(dr1, r1_cache)
    _, grads["W1"], grads["b1"] = conv_backward(dc1, c1_cache)
 
    return grads
 
 
#prediction
def predict(params, x, batch_size=500):
    preds = []
    for start in range(0, len(x), batch_size):
        logits, _ = forward(params, x[start:start + batch_size])
        preds.append(logits.argmax(axis=1))
    return np.concatenate(preds)
 
#evaliation
def evaluate(params, x, y, batch_size=500):
    total_loss, correct = 0.0, 0
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start + batch_size], y[start:start + batch_size]
        logits, _ = forward(params, xb)
        loss, _ = softmax_cross_entropy(logits, yb)
        total_loss += loss * len(xb)
        correct += (logits.argmax(axis=1) == yb).sum()
    return total_loss / len(x), correct / len(x)
 

#gradient check by numerical gradient
def gradient_check():
    print("Gradient check (float64, tiny random batch)")
    rng = np.random.RandomState(1)
    x = rng.randn(2, 1, 28, 28)
    y = np.array([3, 7])
    params = {k: v.astype(np.float64) for k, v in init_params().items()}
 
    logits, caches = forward(params, x)
    _, dlogits = softmax_cross_entropy(logits, y)
    grads = backward(params, dlogits, caches)
 
    eps = 1e-4
    for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        p = params[name]
        worst = 0.0
        for _ in range(10):
            idx = tuple(rng.randint(s) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps
            loss_plus, _ = softmax_cross_entropy(forward(params, x)[0], y)
            p[idx] = old - eps
            loss_minus, _ = softmax_cross_entropy(forward(params, x)[0], y)
            p[idx] = old
            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = grads[name][idx]
 
            rel = abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-7)
            worst = max(worst, rel)
        print(f"  {name}: worst relative error = {worst:.2e}  {'OK' if worst < 1e-6 else 'CHECK THIS'}")
 


###########################TRAININGGGGGGGGGGGGGGGGGGGGGGGGG
def train():
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Loaded {len(x_train)} training and {len(x_test)} test images")
 
    x_val, y_val = x_train[55000:], y_train[55000:]
    x_train, y_train = x_train[:55000], y_train[:55000]
 
    params = init_params()
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
 
    for epoch in range(EPOCHS):
        start_time = time.time()
        order = np.random.permutation(len(x_train))
        running_loss = 0.0
 
        for step, start in enumerate(range(0, len(x_train), BATCH_SIZE)):
            batch = order[start:start + BATCH_SIZE]
            xb, yb = x_train[batch], y_train[batch]
 
            logits, caches = forward(params, xb)
            loss, dlogits = softmax_cross_entropy(logits, yb)
            grads = backward(params, dlogits, caches)
 
            for k in params:
                velocity[k] = MOMENTUM * velocity[k] + grads[k]
                params[k] -= LEARNING_RATE * velocity[k]
 
            running_loss += loss
            if step % 100 == 0:
                print(f"  epoch {epoch + 1} step {step:4d}  loss {loss:.4f}")
 
        train_loss = running_loss / (step + 1)
        val_loss, val_acc = evaluate(params, x_val, y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val acc {val_acc * 100:.2f}% | {time.time() - start_time:.0f}s")
 
    test_loss, test_acc = evaluate(params, x_test, y_test)
    print(f"\nTEST accuracy: {test_acc * 100:.2f}%  (test loss {test_loss:.4f})")
 
    np.savez("cnn_weights.npz", **params)
    make_plots(params, history, x_test)
 



##############MAKING THE PLOTSSSSSSSSSSSSSSS
def make_plots(params, history, x_test):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(5, 4))
    plt.plot(epochs, history["train_loss"], "o-", label="train")
    plt.plot(epochs, history["val_loss"], "o-", label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    #plt.savefig("plot_loss.png")
 
    W1 = params["W1"]
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    for f in range(8):
        axes[f].imshow(W1[f, 0], cmap="gray")
        axes[f].set_title(f"filter {f}")
        axes[f].axis("off")
    plt.suptitle("conv1 filters (3x3)")
    plt.tight_layout()
    #plt.savefig("plot_filters.png")
 
    img = x_test[:1]
    c1, _ = conv_forward(img, params["W1"], params["b1"], pad=1)
    r1 = np.maximum(0, c1)
    p1, _ = maxpool_forward(r1)
    c2, _ = conv_forward(p1, params["W2"], params["b2"], pad=1)
    r2 = np.maximum(0, c2)
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    axes[0, 0].imshow(img[0, 0], cmap="gray")
    axes[0, 0].set_title("input")
    for f in range(8):
        axes[1, f].imshow(r1[0, f], cmap="viridis")
        axes[1, f].set_title(f"conv1 #{f}", fontsize=8)
        axes[2, f].imshow(r2[0, f], cmap="viridis")
        axes[2, f].set_title(f"conv2 #{f}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    plt.suptitle("Feature maps (after ReLU)")
    plt.tight_layout() 
    #plt.savefig("plot_feature_maps.png")
 
    x = np.random.randn(1, 1, 28, 28).astype(np.float32)
    c1, cc1 = conv_forward(x, params["W1"], params["b1"], pad=1)
    r1, rc1 = relu_forward(c1)
    p1, pc1 = maxpool_forward(r1)
    c2, cc2 = conv_forward(p1, params["W2"], params["b2"], pad=1)
    d_c2 = np.zeros_like(c2)
    d_c2[0, 0, 7, 7] = 1.0
    d_p1, _, _ = conv_backward(d_c2, cc2)
    d_x, _, _ = conv_backward(relu_backward(maxpool_backward(d_p1, pc1), rc1), cc1)
    rf = np.abs(d_x[0, 0])
    rows, cols = np.nonzero(rf)
    size = (rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
    print(f"Receptive field of a conv2 unit, measured: {size[0]}x{size[1]} (hand calculation: 8x8)")
    plt.figure(figsize=(4, 4))
    plt.imshow(rf, cmap="hot")
    plt.title(f"conv2 unit receptive field: {size[0]}x{size[1]}")
    plt.tight_layout()
    #plt.savefig("plot_receptive_field.png")
 
    print("Saved plot_loss.png, plot_filters.png, plot_feature_maps.png, plot_receptive_field.png")
    plt.show()
 

#main/start
if __name__ == "__main__":
    gradient_check()
    print()
    train()
 
