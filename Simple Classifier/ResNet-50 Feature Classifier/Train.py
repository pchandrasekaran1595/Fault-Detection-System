import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from time import time
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader as DL

from DatasetTemplates import DS
import utils as u

# ******************************************************************************************************************** #

def fit_(model=None, optimizer=None, scheduler=None, epochs=None, early_stopping_patience=None,
         trainloader=None, validloader=None, criterion=None, device=None,
         save_to_file=False, path=None, verbose=False):

    def getAccuracy(y_pred=None, y_true=None):
        y_pred, y_true = torch.sigmoid(y_pred).detach(), y_true.detach()

        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        return torch.count_nonzero(y_pred == y_true).item() / len(y_true)
    
    u.breaker()
    u.myprint("Training ...", "cyan")
    u.breaker()

    model.to(device)

    DLS = {"train": trainloader, "valid": validloader}
    bestLoss = {"train": np.inf, "valid": np.inf}
    bestAccs = {"train": 0.0, "valid": 0.0}

    Losses = []
    Accuracies = []

    if save_to_file:
        file = open(os.path.join(path, "Metrics.txt"), "w+")

    start_time = time()
    for e in range(epochs):
        e_st = time()

        epochLoss = {"train": 0.0, "valid": 0.0}
        epochAccs = {"train": 0.0, "valid": 0.0}

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            lossPerPass = []
            accsPerPass = []

            for X, y in DLS[phase]:
                X = X.to(device)
                if y.dtype == torch.int64:
                    y = y.to(device).view(-1)
                else:
                    y = y.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output = model(X)
                    loss = criterion(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                lossPerPass.append(loss.item())
                accsPerPass.append(getAccuracy(output, y))
            epochLoss[phase] = np.mean(np.array(lossPerPass))
            epochAccs[phase] = np.mean(np.array(accsPerPass))
        Losses.append(epochLoss)
        Accuracies.append(epochAccs)

        if early_stopping_patience:
            if epochLoss["valid"] < bestLoss["valid"]:
                bestLoss = epochLoss
                bestLossEpoch = e + 1
                torch.save({"model_state_dict": model.state_dict(),
                            "optim_state_dict": optimizer.state_dict()},
                           os.path.join(path, "State.pt"))
                early_stopping_step = 0
            else:
                early_stopping_step += 1
                if early_stopping_step > early_stopping_patience:
                    print("\nEarly Stopping at Epoch {}".format(e + 1))
                    break

        if epochLoss["valid"] < bestLoss["valid"]:
            bestLoss = epochLoss
            bestLossEpoch = e + 1
            torch.save({"model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict()},
                        os.path.join(path, "State.pt"))

        if epochAccs["valid"] > bestAccs["valid"]:
            bestAccs = epochAccs
            bestAccsEpoch = e + 1

        if verbose:
            u.myprint("Epoch: {} | Train Loss: {:.5f} | Valid Loss: {:.5f} | Train Accs : {:.5f} | \
Valid Accs : {:.5f} | Time: {:.2f} seconds".format(e + 1,
                                                   epochLoss["train"], epochLoss["valid"],
                                                   epochAccs["train"], epochAccs["valid"],
                                                   time() - e_st), "cyan")

        if save_to_file:
            text = "Epoch: {} | Train Loss: {:.5f} | Valid Loss: {:.5f} | Train Accs : {:.5f} | \
Valid Accs : {:.5f} | Time: {:.2f} seconds\n".format(e + 1,
                                                     epochLoss["train"], epochLoss["valid"],
                                                     epochAccs["train"], epochAccs["valid"],
                                                     time() - e_st)
            file.write(text)

        if scheduler:
            scheduler.step(epochLoss["valid"])

    u.breaker()
    u.myprint("-----> Best Validation Loss at Epoch {}".format(bestLossEpoch), "cyan")
    u.breaker()
    u.myprint("-----> Best Validation Accs at Epoch {}".format(bestAccsEpoch), "cyan")
    u.breaker()
    u.myprint("Time Taken [{} Epochs] : {:.2f} minutes".format(len(Losses), (time() - start_time) / 60), "cyan")
    u.breaker()
    u.myprint("Training Complete", "cyan")

    if save_to_file:
        text_1 = "\n-----> Best Validation Loss at Epoch {}\n".format(bestLossEpoch)
        text_2 = "-----> Best Validation Accs at Epoch {}\n".format(bestAccsEpoch)
        text_3 = "Time Taken [{} Epochs] : {:.2f} minutes\n".format(len(Losses), (time() - start_time) / 60)

        file.write(text_1)
        file.write(text_2)
        file.write(text_3)
        file.close()

    return Losses, Accuracies, bestLossEpoch, bestAccsEpoch

# ******************************************************************************************************************** #

def trainer(part_name=None, model=None, epochs=None, lr=None, wd=None, batch_size=None, early_stopping=None):
    base_path = os.path.join(u.DATASET_PATH, part_name)
    
    p_features, n_features = np.load(os.path.join(base_path, "Positive_Features.npy")), np.load(os.path.join(base_path, "Negative_Features.npy"))
    kf = KFold(n_splits=5, shuffle=True, random_state=u.SEED).split(p_features)
    for tr_idx, va_idx in kf:
        train_indices, valid_indices = tr_idx, va_idx
        break

    p_train, p_valid = p_features[train_indices], p_features[valid_indices]
    n_train, n_valid = n_features[train_indices], n_features[valid_indices]

    tr_data_setup = DS(p_vector=p_train, n_vector=n_train)
    va_data_setup = DS(p_vector=p_valid, n_vector=n_valid)
    tr_data = DL(tr_data_setup, batch_size=batch_size, shuffle=True, pin_memory=True, generator=torch.manual_seed(u.SEED), )
    va_data = DL(va_data_setup, batch_size=batch_size, shuffle=False, pin_memory=True)
    optimizer = model.getOptimizer(lr=lr, wd=wd)

    checkpoint_path = os.path.join(os.path.join(u.DATASET_PATH, part_name), "Checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, A, _, _ = fit_(model=model, optimizer=optimizer, scheduler=None, epochs=epochs,
                      early_stopping_patience=None, trainloader=tr_data, validloader=va_data, device=u.DEVICE,
                      criterion=torch.nn.BCEWithLogitsLoss(), save_to_file=True, path=checkpoint_path, verbose=True)

    TL, VL, TA, VA = [], [], [], []

    for i in range(len(L)):
        TL.append(L[i]["train"])
        VL.append(L[i]["valid"])
        TA.append(A[i]["train"])
        VA.append(A[i]["valid"])

    # Plots
    x_Axis = np.arange(1, len(L)+1)
    plt.figure("Plots", figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_Axis, TL, "r", label="Training Loss")
    plt.plot(x_Axis, VL, "b", label="validation Loss")
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(x_Axis, TA, "r", label="Training Accuracy")
    plt.plot(x_Axis, VA, "b", label="validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(os.path.join(u.DATASET_PATH, part_name), "Graphs.jpg"))
    plt.close(fig="Plots")

# ******************************************************************************************************************** #
