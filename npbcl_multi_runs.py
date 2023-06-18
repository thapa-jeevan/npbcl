import argparse
import os

import matplotlib
import numpy as np

from data_generators import FashionMnistGenerator, NotMnistGenerator, PermutedMnistGenerator, SplitMnistGenerator
from ibpbcl import IBP_BCL

parser = argparse.ArgumentParser(description='Train', add_help=True)
parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of tasks')
parser.add_argument('--n_epochs', type=int, default=5, help='Number of tasks')
parser.add_argument('--n_tasks', type=int, default=5, help='Number of tasks')
parser.add_argument('--single_head', action="store_true")
parser.add_argument('--dataset', type=str, help='Dataset')
args = parser.parse_args()

matplotlib.use('Agg')


# torch.manual_seed(8)
# np.random.seed(10)

def get_dataset(dataset, no_tasks=None):
    return {"perm_mnist": PermutedMnistGenerator(no_tasks),
            "split_mnist": SplitMnistGenerator(),
            "not_mnist": NotMnistGenerator,
            "fashion_mnist": FashionMnistGenerator,
            }[dataset]


for i in range(3):
    hidden_size = [200] * args.n_hidden_layers
    alpha = [30] * args.n_hidden_layers
    no_epochs = args.n_epochs
    no_tasks = args.n_tasks
    coreset_size = 0  # 200
    coreset_method = "kcen"
    single_head = args.single_head
    batch_size = 256

    os.makedirs('./saves', exist_ok=True)

    data_gen = get_dataset(args.dataset, no_tasks)

    model = IBP_BCL(hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size, single_head, grow=False)

    accs, _ = model.batch_train(batch_size)
    file_path = f'./reports/hls{args.n_hidden_layers}{args.dataset}_accuracies{i}.npy'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, accs)
