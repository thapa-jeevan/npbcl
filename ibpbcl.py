import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# from ibpbnn_wo_mask import IBP_BNN
from ibpbnn import IBP_BNN

# matplotlib.use('Agg')
torch.manual_seed(7)
np.random.seed(10)


class IBP_BCL:
    def __init__(self, hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size=0, single_head=True,
                 grow=False, dataset=None):
        """
        hidden_size : list of network hidden layer sizes
        alpha : IBP prior concentration parameters
        data_gen : Data Generator
        coreset_size : Size of coreset to be used (0 represents no coreset)
        single_head : To given single head output for all task or multihead output for each task seperately.
        """
        # Intializing Hyperparameters for the model.
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = [1.0 for i in range(len(hidden_size))]
        self.no_epochs = no_epochs
        self.data_gen = data_gen
        if (coreset_method != "kcen"):
            self.coreset_method = self.rand_from_batch
        else:
            self.coreset_method = self.k_center
        self.coreset_size = coreset_size
        self.single_head = single_head
        self.grow = grow
        self.cuda = torch.cuda.is_available()
        if dataset:
            self.save_path = f"results/{dataset}"
            self.ckpt_save_path = f"checkpoints/{dataset}"
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(self.ckpt_save_path, exist_ok=True)

    def rand_from_batch(self, x_coreset, y_coreset, x_train, y_train, coreset_size):
        """ Random coreset selection """
        # Randomly select from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
        idx = np.random.choice(x_train.shape[0], coreset_size, False)
        x_coreset.append(x_train[idx, :])
        y_coreset.append(y_train[idx, :])
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)
        return x_coreset, y_coreset, x_train, y_train

    def k_center(self, x_coreset, y_coreset, x_train, y_train, coreset_size):
        """ K-center coreset selection """
        # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
        dists = np.full(x_train.shape[0], np.inf)
        current_id = 0
        dists = self.update_distance(dists, x_train, current_id)
        idx = [current_id]
        for i in range(1, coreset_size):
            current_id = np.argmax(dists)
            dists = self.update_distance(dists, x_train, current_id)
            idx.append(current_id)
        x_coreset.append(x_train[idx, :])
        y_coreset.append(y_train[idx, :])
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)
        return x_coreset, y_coreset, x_train, y_train

    def update_distance(self, dists, x_train, current_id):
        for i in range(x_train.shape[0]):
            current_dist = np.linalg.norm(x_train[i, :] - x_train[current_id, :])
            dists[i] = np.minimum(current_dist, dists[i])
        return dists

    def merge_coresets(self, x_coresets, y_coresets):
        # Merges the current task coreset to rest of the coresets
        merged_x, merged_y = x_coresets[0], y_coresets[0]
        for i in range(1, len(x_coresets)):
            merged_x = np.vstack((merged_x, x_coresets[i]))
            merged_y = np.vstack((merged_y, y_coresets[i]))
        return merged_x, merged_y

    def logit(self, x):
        eps = 10e-8
        return (np.log(x + eps) - np.log(1 - x + eps))

    def get_soft_logit(self, masks, task_id):
        var = []
        for i in range(len(masks)):
            var.append(self.logit(masks[i][task_id] * 0.98 + 0.01))

        return var

    def get_scores(self, model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size,
                   no_epochs, single_head, batch_size=None, kl_mask=None):
        # Retrieving the current model parameters
        mf_model = model
        mf_weights, mf_variances = model.get_weights()
        prev_masks, alpha, beta = mf_model.get_IBP()
        acc = []

        # In case the model is single head or have coresets then we need to test accodingly.
        if single_head:  # If model is single headed.
            if len(x_coresets) > 0:  # Model has non zero coreset size
                del mf_model
                torch.cuda.empty_cache()
                x_train, y_train = self.merge_coresets(x_coresets, y_coresets)
                prev_pber = self.get_soft_logit(prev_masks, i)
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                final_model = IBP_BNN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], self.max_tasks,
                                      prev_means=mf_weights, prev_log_variances=mf_variances,
                                      prev_masks=prev_masks, alpha=alpha, beta=beta, prev_pber=prev_pber,
                                      kl_mask=kl_mask, single_head=single_head)
                final_model.ukm = 1
                final_model.batch_train(x_train, y_train, 0, 1, bsize, 1)
            else:  # Model does not have coreset
                final_model = model

        # Testing for all previously learned tasks
        for i in range(len(x_testsets)):
            if not single_head:  # If model is multi headed.
                if len(x_coresets) > 0:
                    try:
                        del mf_model
                    except:
                        pass
                    torch.cuda.empty_cache()
                    x_train, y_train = x_coresets[i], y_coresets[i]  # coresets per task
                    prev_pber = self.get_soft_logit(prev_masks, i)
                    bsize = x_train.shape[0] if (batch_size is None) else batch_size
                    final_model = IBP_BNN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0],
                                          self.max_tasks,
                                          prev_means=mf_weights, prev_log_variances=mf_variances,
                                          prev_masks=prev_masks, alpha=alpha, beta=beta, prev_pber=prev_pber,
                                          kl_mask=kl_mask, learning_rate=0.0001, single_head=single_head)
                    final_model.ukm = 1
                    final_model.batch_train(x_train, y_train, i, 1, bsize, 1, init_temp=final_model.min_temp)
                else:
                    final_model = model

            x_test, y_test = x_testsets[i], y_testsets[i]

            pred = final_model.prediction_prob(x_test, i)
            pred_mean = np.mean(pred, axis=1)
            pred_y = np.argmax(pred_mean, axis=1)
            y = np.argmax(y_test, axis=1)
            # Calculating the accuracy of the model on given test/validaiton data.
            cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
            acc.append(cur_acc)

        return acc

    def concatenate_results(self, score, all_score):
        # Concats the current accuracies on all task to previous result in form of matrix
        if all_score.size == 0:
            all_score = np.reshape(score, (1, -1))
        else:
            new_arr = np.empty((all_score.shape[0], all_score.shape[1] + 1))
            new_arr[:] = np.nan  # Puts nan in place of empty values (tasks that previous model was not trained on)
            new_arr[:, :-1] = all_score
            all_score = np.vstack((new_arr, score))
        return all_score

    def batch_train(self, batch_size=None):
        # Intializing coresets and dimensions.
        in_dim, out_dim = self.data_gen.get_dims()
        x_coresets, y_coresets = [], []
        x_testsets, y_testsets = [], []
        x_trainset, y_trainset = [], []
        all_acc = np.array([])
        self.max_tasks = self.data_gen.max_iter
        fig1, ax1 = plt.subplots(1, self.max_tasks, figsize=[self.max_tasks * 2, 5])

        fig_sparsity_acc, ax_sparsity_acc = plt.subplots(1, 2, figsize=[self.max_tasks * 2, 5])
        sparsity_acc_tasks = []
        # Training the model sequentially.
        for task_id in range(self.max_tasks):
            # Loading training and test data for current task
            x_train, y_train, x_test, y_test = self.data_gen.next_task()
            x_testsets.append(x_test)
            y_testsets.append(y_test)
            # Initializing the batch size for training 
            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            # If this is the first task we need to initialize few variables.
            if task_id == 0:
                prev_masks = None
                prev_pber = None
                kl_mask = None
                mf_weights = None
                mf_variances = None
            # Select coreset if coreset size is non zero
            if self.coreset_size > 0:
                x_coresets, y_coresets, x_train, y_train = self.coreset_method(x_coresets, y_coresets, x_train, y_train,
                                                                               self.coreset_size)

            # Training the network
            mf_model = IBP_BNN(in_dim, self.hidden_size, out_dim, x_train.shape[0], self.max_tasks,
                               prev_means=mf_weights, prev_log_variances=mf_variances,
                               prev_masks=prev_masks, alpha=self.alpha, beta=self.beta, prev_pber=prev_pber,
                               kl_mask=kl_mask, single_head=self.single_head)

            mf_model.grow_net = self.grow
            if (self.cuda):
                mf_model = mf_model.cuda()
                if torch.cuda.device_count() > 1:
                    mf_model = nn.DataParallel(mf_model)  # enabling data parallelism
            mf_model.batch_train(x_train, y_train, task_id, self.no_epochs, bsize, max(self.no_epochs // 5, 1))
            mf_weights, mf_variances = mf_model.get_weights()
            prev_masks, self.alpha, self.beta = mf_model.get_IBP()
            self.hidden_size = deepcopy(mf_model.size[1:-1])

            # Figure of masks that has been learned for all seen tasks.
            fig, ax = plt.subplots(1, task_id + 1, figsize=[self.max_tasks * 2, 5])
            for i, m in enumerate(prev_masks[0][:task_id + 1]):
                if (task_id == 0):
                    ax.imshow(m, vmin=0, vmax=1)
                else:
                    ax[i].imshow(m, vmin=0, vmax=1)
            fig.savefig(os.path.join(self.save_path, "all_masks.png"))
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
            fig1.savefig(os.path.join(self.save_path, "union_mask.png"))

            sparsity_acc_tasks.append(sparsity[0])
            ax_sparsity_acc[0].plot(sparsity_acc_tasks, color="blue")
            ax_sparsity_acc[0].scatter(np.arange(len(sparsity_acc_tasks)), sparsity_acc_tasks, color="blue")

            print("Network sparsity : ", sparsity)

            mf_model.grow_net = False

            acc = self.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets,
                                  self.hidden_size, self.no_epochs, self.single_head, batch_size, kl_mask)

            ckpt_path = os.path.join(self.ckpt_save_path, "model_last_" + str(task_id))
            torch.save(mf_model.state_dict(), ckpt_path)
            del mf_model
            torch.cuda.empty_cache()
            all_acc = self.concatenate_results(acc, all_acc)
            ax_sparsity_acc[1].plot(np.nanmean(all_acc, axis=1), color="red")
            ax_sparsity_acc[1].scatter(np.arange(len(sparsity_acc_tasks)), np.nanmean(all_acc, axis=1), color="red")

            fig_sparsity_acc.savefig(os.path.join(self.save_path, "sparsity_acc_tasks.png"))

            print(all_acc);
            print('*****')

        return [all_acc, prev_masks]
