import os

import matplotlib
import numpy as np
import torch

from data_generators import PermutedMnistGenerator
from ibpbcl import IBP_BCL

matplotlib.use('Agg')

torch.manual_seed(8)
np.random.seed(10)

hidden_size = [200]
alpha = [30]
no_epochs = 5
no_tasks = 5
coreset_size = 0  # 200
coreset_method = "kcen"
single_head = False
batch_size = 256

os.makedirs('./saves', exist_ok=True)

data_gen = PermutedMnistGenerator(no_tasks)
# data_gen = SplitMnistGenerator()
# data_gen = NotMnistGenerator()
# data_gen = FashionMnistGenerator()
model = IBP_BCL(hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size, single_head, grow=False)

accs, _ = model.batch_train(batch_size)
np.save('./saves/permutedmnist_accuracies.npy', accs)
