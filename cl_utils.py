import matplotlib.pyplot as plt
import numpy as np
import torch

from ibpbnn import IBP_BNN


def rand_from_batch(x_coreset, y_coreset, x_train, y_train, coreset_size):
    """ Random coreset selection """
    # Randomly select from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
    idx = np.random.choice(x_train.shape[0], coreset_size, replace=False)
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
    dists = update_distance(dists, x_train, current_id)
    idx = [current_id]
    for i in range(1, coreset_size):
        current_id = np.argmax(dists)
        dists = update_distance(dists, x_train, current_id)
        idx.append(current_id)
    x_coreset.append(x_train[idx, :])
    y_coreset.append(y_train[idx, :])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)
    return x_coreset, y_coreset, x_train, y_train


def update_distance(dists, x_train, current_id):
    for i in range(x_train.shape[0]):
        current_dist = np.linalg.norm(x_train[i, :] - x_train[current_id, :])
        dists[i] = np.minimum(current_dist, dists[i])
    return dists


def merge_coresets(x_coresets, y_coresets):
    return np.vstack(x_coresets), np.vstack(y_coresets)


def logit(x, eps=10e-8):
    return np.log(x + eps) - np.log(1 - x + eps)


def get_soft_logit(masks, task_id):
    return [logit(masks[i][task_id] * 0.98 + 0.01) for i in range(len(masks))]


def concatenate_results(score, all_score):
    # Concats the current accuracies on all task to previous result in form of matrix
    if all_score.size == 0:
        all_score = np.reshape(score, (1, -1))
    else:
        new_arr = np.empty((all_score.shape[0], all_score.shape[1] + 1))
        new_arr[:] = np.nan  # Puts nan in place of empty values (tasks that previous model was not trained on)
        new_arr[:, :-1] = all_score
        all_score = np.vstack((new_arr, score))
    return all_score


def plot_masks(prev_masks, task_id):
    # Figure of masks that has been learned for all seen tasks.
    fig, ax = plt.subplots(1, task_id + 1, figsize=[10, 5])
    for i, m in enumerate(prev_masks[0][:task_id + 1]):
        if task_id == 0:
            ax.imshow(m, vmin=0, vmax=1)
        else:
            ax[i].imshow(m, vmin=0, vmax=1)
    fig.savefig("all_masks.png")


def get_scores(model, max_tasks, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size,
               single_head, task_id, batch_size=None, kl_mask=None):
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
            x_train, y_train = merge_coresets(x_coresets, y_coresets)
            prev_pber = get_soft_logit(prev_masks, task_id)
            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            final_model = IBP_BNN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], max_tasks,
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
                prev_pber = get_soft_logit(prev_masks, i)
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                final_model = IBP_BNN(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0],
                                      max_tasks,
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
