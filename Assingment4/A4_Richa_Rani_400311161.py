import numpy as np
import torch
from torchvision import datasets
import matplotlib.pyplot as plt

#hyperparameters
seed_global= 1131
lr= 0.01 #learning rate
layers = [784, 156, 156, 10] # 784 pixels, 2 hidden layers, 10 classes
batch_size = 256
lam = 0.0018738 # L2 weight decay
early_stopping= 10 # patience
max_epochs= 30
np.random.seed(seed_global)
torch.manual_seed(seed_global)

#load data (as seen in the assignment doc)
num_classes = 10

train_dataset = datasets.FashionMNIST(root="./data", train=True,  download=True)
test_dataset  = datasets.FashionMNIST(root="./data", train=False, download=True)

X_train_full = train_dataset.data.numpy().reshape(-1, 28*28).astype("float32") / 255.0
y_train_full = train_dataset.targets.numpy()

X_test  = test_dataset.data.numpy().reshape(-1, 28*28).astype("float32") / 255.0
y_test  = test_dataset.targets.numpy()

print("Full train shape:", X_train_full.shape, "test shape:", X_test.shape)

# ---- train / validation split (80 / 20) ----
N_full = X_train_full.shape[0]
validation_size = int(0.2 * N_full)

X_val   = X_train_full[:validation_size]
y_val   = y_train_full[:validation_size]
X_train = X_train_full[validation_size:]
y_train = y_train_full[validation_size:]

print("Train:", X_train.shape, "Val:", X_val.shape)

# ---- one-hot encoding ----
def one_hot_encode(labels, num_classes):
    N = labels.shape[0]
    Y = np.zeros((N, num_classes), dtype=np.float32)
    Y[np.arange(N), labels] = 1.0
    return Y

Y_train = one_hot_encode(y_train, num_classes)
Y_val   = one_hot_encode(y_val,   num_classes)
Y_test  = one_hot_encode(y_test,  num_classes)

# ---- standardize using TRAIN mean/std only ----
train_mean = X_train.mean(axis=0)
train_std  = X_train.std(axis=0)
train_std[train_std == 0] = 1.0  # avoid division by zero

X_train = (X_train - train_mean) / train_std
X_val   = (X_val   - train_mean) / train_std
X_test  = (X_test  - train_mean) / train_std

#initialization 
def init_uniform_1_over_sqrt_m(rng, m, n, dtype=np.float32):
    # W_ij ~ Unif(-1/sqrt(m), 1/sqrt(m)), m = fan-in
    bound = 1.0 / np.sqrt(m)
    W = rng.uniform(-bound, bound, size=(m, n)).astype(dtype) #bound 1/sqrt(m) and uniform dist between 
    b = np.zeros((n,), dtype=dtype)
    return W, b

def init_uniform_sqrt6_over_m_plus_n(rng, m, n, dtype=np.float32):
    # W_ij ~ Unif(-sqrt(6/(m+n)), sqrt(6/(m+n)))
    bound = np.sqrt(6.0 / (m + n))
    W = rng.uniform(-bound, bound, size=(m, n)).astype(dtype)
    b = np.zeros((n,), dtype=dtype)
    return W, b

def build_params(layers, init_name, seed_init):
    #builds param as per the init strat chosen
    rng = np.random.default_rng(seed_init)
    W_list, b_list = [], []
    for i in range(len(layers) - 1):
        #fan in fan out methods for model B (just fan in for A)
        m = layers[i]
        n = layers[i+1]
        #based on init name itll do loop
        if init_name == "uniform_1_sqrt_m":
            W, b = init_uniform_1_over_sqrt_m(rng, m, n)
        elif init_name == "uniform_sqrt6_m_plus_n":
            W, b = init_uniform_sqrt6_over_m_plus_n(rng, m, n)
        #add W b to list and return them 
        W_list.append(W)
        b_list.append(b)

    return W_list, b_list


# Activations
def relu(z):
    return np.maximum(z, 0.0)

def relu_grad(z): #gradient reLu the piecewise func
    return (z > 0).astype(float)

def softmax(z): #output activation 
    # keep dims to fix error
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True) #e^z/sum(e^z)

#loss functions 
def cross_entropy_onehot(p, t):
    # p,t : (N,K)
    return -np.mean(np.sum(t * np.log(p + 1e-12), axis=1))

def l2_penalty(W_list):
    #sum over all layers l of ||W^(l)||_F^2
    total = 0.0
    for W in W_list:
        total += np.sum(W * W)
    return total


#  Forward & Backward passes
def forward_pass(X, W_list, b_list):
    #Hidden layers: ReLU
    #Output layer: softmax
    A = X # current activations
    caches = [] # store (A_prev, Z) for each layer

    # hidden layers: 0 to L-2. zip will iterate through both till -1 or the last elemet
    for W, b in zip(W_list[:-1], b_list[:-1]):
        Z = np.dot(A, W) + b 
        caches.append((A, Z))# save A^{(l-1)} and Z^{(l)}
        A = relu(Z) # activation for next layer

    # output layer: softmax
    ZL = np.dot(A , W_list[-1]) + b_list[-1] # (N, K)
    P= softmax(ZL) # (N, K) activiation 
    caches.append((A, ZL)) # last hidden A, output Z

    return P, caches

def backward_pass(P, T, W_list, b_list, caches, lam=0.0):
    #P predicted probs
    #T one-hot targets

    #Returns
    #loss CE + L2
    #grads_W: list dJ/dW_l same shapes as W_list
    #grads_b: list dJ/db_l same shapes as b_list
    L = len(W_list)#number of weight layers
    N = T.shape[0] #batch size

    # loss = CE + lambda * sum ||W||^2 
    ce = cross_entropy_onehot(P, T)
    l2 = l2_penalty(W_list)
    loss = ce + lam * l2
    #empty gradient containers
    grads_W = [None] * L #list of len L
    grads_b = [None] * L

    # output layer 
    # derivative dJ/dZ^L = (P - T)/N  (softmax + CE)
    dZ = (P - T) / N # (N,K)  dJ/dz^(L)
    A_prev, ZL = caches[-1] # last hidden A, output Z
    grads_W[-1] = np.dot(A_prev.T , dZ) + 2 * lam * W_list[-1] # (n_{L-1}, K)
    grads_b[-1] = dZ.sum(axis=0) # (K,)
    dA = np.dot(dZ, W_list[-1].T) # backprop to A^{L-1}

    # hidden layers: L-1 to 1
    for i in range(L-2, -1, -1):
        A_prev, Z = caches[i] # A^{(l-1)}, Z^{(l)}
        dZ = dA * relu_grad(Z) 
        #grads
        grads_W[i] = np.dot(A_prev.T, dZ) + 2 * lam * W_list[i]
        grads_b[i] = dZ.sum(axis=0)
        dA = np.dot(dZ, W_list[i].T)# backprop to previous layer

    return loss, grads_W, grads_b


# Loss evaluation & prediction
def evaluate_loss(X, T, W_list, b_list, lam, eval_batch_size=2048):
    #Compute CE + L2 on a dataset using mini-batches
    N = X.shape[0] #total ce loss * batch sizes 
    total_ce = 0.0
    count = 0 #number of samples processed so far 

    for start in range(0, N, eval_batch_size):
        xb = X[start:start + eval_batch_size]
        tb = T[start:start + eval_batch_size]
        P, _ = forward_pass(xb, W_list, b_list)
        ce = cross_entropy_onehot(P, tb)
        total_ce += ce * xb.shape[0]
        count += xb.shape[0]

    ce_mean = total_ce / count #avg over samples 
    l2 = l2_penalty(W_list) #sum of squares of all weights in all layers 
    return ce_mean + lam * l2

def predict_classes(X, W_list, b_list, eval_batch_size=2048):
    #Return predicted class indices on X.
    N = X.shape[0]
    preds = [] #store predictions for each mini batch in this list 

    for start in range(0, N, eval_batch_size): #loop over X in mini batches 
        xb = X[start:start + eval_batch_size]
        P, _ = forward_pass(xb, W_list, b_list) #frwrd pass to get prob P 
        batch_preds = np.argmax(P, axis=1) #each row is sample, shape is batch size. returns index of largest P 
        preds.append(batch_preds)
    #concatenate all of the batch pred into one big array of (N,1)
    return np.concatenate(preds, axis=0)

# Training one model
def train_one_model(X_train, Y_train, X_val, Y_val, init_name, seed_init):
    #Train ONE MLP with mini-batch GD + L2 + early stopping
    rng = np.random.default_rng(seed_init)
    N = X_train.shape[0]

    # init parameters
    W_list, b_list = build_params(layers, init_name, seed_init)

    best_val = np.inf
    best_W, best_b = None, None
    wait = 0

    train_losses = []
    val_losses   = []

    for epoch in range(1, max_epochs + 1):
        # shuffle indices. medium article in src 
        idx = rng.permutation(N)

        epoch_loss_sum = 0.0
        epoch_count = 0

        #minibatch
        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = X_train[batch_idx]
            tb = Y_train[batch_idx]

            P, caches = forward_pass(xb, W_list, b_list)
            loss_batch, grads_W, grads_b = backward_pass(P, tb, W_list, b_list, caches, lam=lam)
            #batch size
            bs = xb.shape[0]
            epoch_loss_sum += loss_batch * bs
            epoch_count += bs

            # parameter update
            for i in range(len(W_list)):
                W_list[i] -= lr * grads_W[i]
                b_list[i] -= lr * grads_b[i]

        # avg training loss
        train_loss = epoch_loss_sum / epoch_count
        train_losses.append(train_loss)

        #validaion loss
        val_loss = evaluate_loss(X_val, Y_val, W_list, b_list, lam)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        # early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_W = [W.copy() for W in W_list]
            best_b = [b.copy() for b in b_list]
            wait = 0
        else:
            wait += 1
            if wait >= early_stopping:
                print("Early stop")
                W_list = best_W
                b_list = best_b
                break

    return W_list, b_list, train_losses, val_losses


# Main: train Model A & B
def main():
    results = {}

    # Model A: Unif(-1/sqrt(m), 1/sqrt(m)) 
    print("\nTraining Model A (Unif(-1/sqrt(m), 1/sqrt(m)))...")
    init_name_A = "uniform_1_sqrt_m"
    W_A, b_A, train_A, val_A = train_one_model(X_train, Y_train, X_val, Y_val, init_name_A, seed_global)
    #predict the classes
    y_train_pred_A = predict_classes(X_train, W_A, b_A)
    y_test_pred_A= predict_classes(X_test,  W_A, b_A)
    #train error
    #if correct. y_train_pred == y_train. (1-correct)*100 gives percent error
    train_err_A= 100* (1.0- np.mean(y_train_pred_A == y_train))
    test_err_A= 100* (1.0- np.mean(y_test_pred_A  == y_test))

    print("\nresults[\"A\"]:")
    print(f"W shape : {[W.shape for W in W_A]}")
    print(f"b shape: {[b.shape for b in b_A]}")
    print(f"#train losses: {len(train_A)}")
    print(f"#val losses: {len(val_A)}")
    print(f"train_err(A): {train_err_A:.2f}%")
    print(f"test_err(B): {test_err_A:.2f}%")

    #Model B: Unif(-sqrt(6/(m+n)), sqrt(6/(m+n))) 
    print("\nTraining Model B (Unif(-sqrt(6/(m+n)), sqrt(6/(m+n))))...")
    init_name_B = "uniform_sqrt6_m_plus_n"
    W_B, b_B, train_B,val_B = train_one_model(X_train, Y_train, X_val, Y_val,init_name_B, seed_global)
    #predict classes
    y_train_pred_B= predict_classes(X_train, W_B, b_B)
    y_test_pred_B= predict_classes(X_test,W_B, b_B)
    
    #train error
    #if correct. y_train_pred == y_train. (1-correct)*100 gives percent error
    train_err_B= 100* (1.0 - np.mean(y_train_pred_B == y_train))
    test_err_B= 100 *(1.0 - np.mean(y_test_pred_B  == y_test))

    print("\nresults[\"A\"]:")
    print(f"W shape : {[W.shape for W in W_B]}")
    print(f"b shape: {[b.shape for b in b_B]}")
    print(f"#train losses: {len(train_B)}")
    print(f"#val losses: {len(val_B)}")
    print(f"train_err(B): {train_err_B:.2f}%")
    print(f"test_err(B): {test_err_B:.2f}%")
    
    # Print misclassification errors 
    print("\nMisclassification errors:")
    print(f"Model A - train: {train_err_A:6.2f}% | test: {test_err_A:6.2f}%")
    print(f"Model B - train: {train_err_B:6.2f}% | test: {test_err_B:6.2f}%")

    # plot learning curves
    epochs_A= np.arange(1,len(train_A) + 1) #=1 for index
    epochs_B= np.arange(1,len(train_B) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_A, train_A, label="Train loss (Model A)")
    plt.plot(epochs_A, val_A,   label="Val loss (Model A)", linestyle="--")
    plt.plot(epochs_B, train_B, label="Train loss (Model B)")
    plt.plot(epochs_B, val_B,   label="Val loss (Model B)", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / validation loss for Model A & B")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



main()
