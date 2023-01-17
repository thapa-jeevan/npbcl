import numpy as np
import torch
import torch.nn.functional as F

from utils import parameterized_truncated_normal


def reparam_bernoulli(logp, K, device, temp=0.1, eps=1e-8):
    din, dout = logp.shape[1], logp.shape[2]
    # Sampling from the gumbel distribution and Reparameterizing
    U = torch.tensor(np.reshape(np.random.uniform(size=K * din * dout), [K, din, dout])).float().to(device)
    L = ((U + eps).log() - (1 - U + eps).log())
    B = torch.sigmoid((L + logp) / temp.unsqueeze(0))
    return B


def truncated_normal(shape, stddev=0.01):
    ''' Initialization : Function to return an truncated_normal initialized parameter'''
    uniform = torch.from_numpy(np.random.uniform(0, 1, shape)).float()
    return parameterized_truncated_normal(uniform, mu=0.0, sigma=stddev, a=-2 * stddev, b=2 * stddev)


# Done
def constant_init(init, shape=None):
    ''' Initialization : Function to return an constant initialized parameter'''
    if shape is None:
        return torch.tensor(init).float()
    return torch.ones(shape).float() * init


def softplus(x, beta=1.0, threshold=20.0):
    return F.softplus(x, beta=beta, threshold=threshold)


# Done
def softplus_inverse(x, beta=1.0, threshold=20.0):
    eps = 10e-8
    mask = (x <= threshold).float().detach()
    xd1 = x * mask
    xd2 = xd1.mul(beta).exp().sub(1.0 - eps).log().div(beta)
    xd3 = xd2 * mask + x * (1 - mask)
    return xd3


def extend_tensor(device):
    def _extend_tensor(tensor, dims=None, extend_with=0.0):
        if dims is None:
            return tensor
        else:
            if len(dims) == 1:
                temp = tensor.cpu().detach().numpy()
                D = temp.shape[0]
                new_array = np.zeros(dims[0] + D) + extend_with
                new_array[:D] = temp
            elif len(dims) == 2:
                temp = tensor.cpu().detach().numpy()
                D1, D2 = temp.shape
                new_array = np.zeros((D1 + dims[0], D2 + dims[1])) + extend_with
                new_array[:D1, :D2] = temp

            return torch.tensor(new_array).float().to(device)

    return _extend_tensor


def log_gumb(temp, log_alpha, log_sample):
    # Returns log probability of gumbel distribution
    eps = 10e-8
    exp_term = log_alpha + log_sample * (-temp.unsqueeze(0))
    log_prob = exp_term + (temp + eps).log().unsqueeze(0) - 2 * softplus(exp_term)
    return log_prob


def sample_gauss(mean, logvar, sample_size, device):
    N, M = mean.shape
    # samples xN x M
    return torch.randn(sample_size, N, M).to(device) * ((0.5 * logvar).exp().unsqueeze(0)) + mean.unsqueeze(0)


def logit(x, eps=1e-8):
    return (x + eps).log() - (1 - x + eps).log()


def accuracy(val_step, x_test, y_test, task_id, batch_size=1000):
    '''Prints the accuracy of the model for a given input output pairs'''
    N = x_test.shape[0]
    if batch_size > N:
        batch_size = N

    costs = []
    cur_x_test = x_test
    cur_y_test = y_test
    total_batch = int(np.ceil(N * 1.0 / batch_size))

    avg_acc = 0.
    for i in range(total_batch):
        start_ind = i * batch_size
        end_ind = np.min([(i + 1) * batch_size, N])
        batch_x = cur_x_test[start_ind:end_ind, :]
        batch_y = cur_y_test[start_ind:end_ind, :]
        acc = val_step(batch_x, batch_y, task_id, temp=0.1, fix=True)
        avg_acc += acc / total_batch
    print(avg_acc)
