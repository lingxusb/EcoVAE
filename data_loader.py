import torch
import numpy as np

def load_data():
    # For example, you might also load data from a CSV file or a database.

    x_train = torch.Tensor(np.loadtxt("./data/training/train.txt"))  # 1000 samples, 100 features
    x_eval = torch.Tensor(np.loadtxt("./data/training/eval.txt"))    # 200 samples, 100 features
    
    return x_train, x_eval