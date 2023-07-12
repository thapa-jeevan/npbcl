import argparse
import os
import torch

import matplotlib
import numpy as np

from data_generators import PermutedMnistGenerator
from ibpbcl import IBP_BCL

parser = argparse.ArgumentParser(description='Train', add_help=True)
parser.add_argument('--n_hidden_layers', type=int, default=1, help='Number of tasks')
parser.add_argument('--n_epochs', type=int, default=5, help='Number of tasks')
parser.add_argument('--n_tasks', type=int, default=5, help='Number of tasks')
parser.add_argument('--single_head', action="store_true")
args = parser.parse_args()

matplotlib.use('Agg')

torch.manual_seed(8)
np.random.seed(10)


hidden_size = [200] * args.n_hidden_layers
alpha = [30] * args.n_hidden_layers
no_epochs = args.n_epochs
no_tasks = args.n_tasks
coreset_size = 0  # 200
coreset_method = "kcen"
single_head = args.single_head
batch_size = 256

os.makedirs('./saves', exist_ok=True)

data_gen = PermutedMnistGenerator(no_tasks)
# data_gen = SplitMnistGenerator()
# data_gen = NotMnistGenerator()
# data_gen = FashionMnistGenerator()
model = IBP_BCL(hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size, single_head, grow=False, dataset="perm_mnist")

accs, _ = model.batch_train(batch_size)
np.save('./saves/permutedmnist_accuracies.npy', accs)
