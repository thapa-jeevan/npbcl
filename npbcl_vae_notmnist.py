import os

try:
    os.mkdir('./Gens')
    os.mkdir('./saves')
except:
    pass
import numpy as np
import matplotlib

matplotlib.use('Agg')
from data_generators import OneNotMnistGenerator
from ibpbcl_vae import IBP_BCL

hidden_size = [500, 500, 100]
# alpha = [80.0, 80.0, 20.0, 80.0, 80.0]
alpha = [40.0, 40.0, 20.0, 40.0, 40.0]
no_epochs = 250  #
no_tasks = 10
coreset_size = 0  # 50
coreset_method = "rand"
single_head = True
batch_size = 64

data_gen = OneNotMnistGenerator()
model = IBP_BCL(hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size, single_head)

liks, _ = model.batch_train(batch_size)
np.save('./saves/notmnist_likelihoods.npy', liks)
