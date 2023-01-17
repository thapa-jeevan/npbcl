from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from cl_utils import get_scores, k_center, plot_masks, rand_from_batch, concatenate_results
from ibpbnn import IBP_BNN

# matplotlib.use('Agg')
torch.manual_seed(7)
np.random.seed(10)


class IBP_BCL:
    def __init__(self, hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size=0, single_head=True,
                 grow=False):
        """
        hidden_size : list of network hidden layer sizes
        alpha : IBP prior concentration parameters
        data_gen : Data Generator
        coreset_size : Size of coreset to be used (0 represents no coreset)
        single_head : To given single head output for all task or multihead output for each task seperately.
        """
        # Intializing Hyperparameters for the model.
        self.hidden_size = hidden_size  # [200]
        self.alpha = alpha  # 30
        self.beta = [1.0 for i in range(len(hidden_size))]
        self.no_epochs = no_epochs  # 5

        self.coreset_method = k_center if coreset_method != "kcen" else rand_from_batch
        self.coreset_size = coreset_size  # 0
        self.single_head = single_head  # False
        self.grow = grow  # False
        self.cuda = torch.cuda.is_available()

        self.data_gen = data_gen
        self.max_tasks = self.data_gen.max_iter

    def batch_train(self, batch_size):
        # Intializing coresets and dimensions.
        in_dim, out_dim = self.data_gen.get_dims()
        x_coresets, y_coresets = [], []
        x_testsets, y_testsets = [], []

        all_acc = np.array([])

        fig1, ax1 = plt.subplots(1, self.max_tasks, figsize=[10, 5])

        # Training the model sequentially.
        for task_id in range(self.max_tasks):
            # Loading training and test data for current task
            x_train, y_train, x_test, y_test = self.data_gen.next_task()
            x_testsets.append(x_test)
            y_testsets.append(y_test)

            # If this is the first task we need to initialize few variables.
            if task_id == 0:
                prev_masks = None
                prev_pber = None
                kl_mask = None
                mf_weights = None
                mf_variances = None

            # Select coreset if coreset size is non-zero
            if self.coreset_size > 0:
                x_coresets, y_coresets, x_train, y_train = self.coreset_method(
                    x_coresets, y_coresets, x_train, y_train, self.coreset_size)

            # Training the network
            mf_model = IBP_BNN(in_dim, self.hidden_size, out_dim, x_train.shape[0], self.max_tasks,
                               prev_means=mf_weights, prev_log_variances=mf_variances,
                               prev_masks=prev_masks,
                               alpha=self.alpha, beta=self.beta, prev_pber=prev_pber,
                               kl_mask=kl_mask, single_head=self.single_head)

            # TODO: merge in init params
            mf_model.grow_net = self.grow

            if self.cuda:
                mf_model = mf_model.cuda()

            mf_model.batch_train(x_train, y_train, task_id, self.no_epochs, batch_size,
                                 display_epoch=max(self.no_epochs // 5, 1))

            # TODO: single params obtain
            mf_weights, mf_variances = mf_model.get_weights()
            prev_masks, self.alpha, self.beta = mf_model.get_IBP()

            self.hidden_size = deepcopy(mf_model.size[1:-1])

            plot_masks(prev_masks, task_id)
            # Calculating Union of all task masks and also for visualizing the layer wise network sparsity
            sparsity = []
            kl_mask = []
            M = len(mf_variances[0])
            for j in range(M):
                c_layer_masks = prev_masks[j]
                canvas = np.zeros_like(c_layer_masks[task_id])
                for t_id in range(task_id + 1):
                    din, dout = c_layer_masks[t_id].shape
                    canvas[:din, :dout] = canvas[:din, :dout] + c_layer_masks[t_id]
                # Plotting union mask
                mask = (canvas > 0.01) * 1.0
                # Calculating network sparsity
                kl_mask.append(mask)
                filled = np.mean(mask)
                sparsity.append(filled)

            ax1[task_id].imshow(mask, vmin=0, vmax=1)
            fig1.savefig("union_mask.png")
            print("Network sparsity : ", sparsity)

            # TODO: Network growth stopped irrespective of input
            mf_model.grow_net = False

            acc = get_scores(mf_model, self.max_tasks, x_testsets, y_testsets, x_coresets, y_coresets,
                             self.hidden_size, self.single_head, task_id, batch_size, kl_mask)

            torch.save(mf_model.state_dict(), "./saves/model_last_" + str(task_id))
            del mf_model
            torch.cuda.empty_cache()
            all_acc = concatenate_results(acc, all_acc)
            print([f"task_{i}" for i in range(task_id + 1)])
            print(all_acc)
            print('*****')

        return [all_acc, prev_masks]
