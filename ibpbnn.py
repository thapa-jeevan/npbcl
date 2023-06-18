from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
from scipy.special import beta as BETA

from adam import Adam
from utils import *

torch.manual_seed(7)
np.random.seed(10)


class IBP_BNN(nn.Module):
    # Done
    def __init__(self, input_size, hidden_size, output_size, training_size, max_tasks,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,  learning_rate=0.01,
                 prior_mean=0.0, prior_var=0.1, alpha=None, beta=None, prev_pber=None, re_mode='gumbsoft',
                 single_head=False, acts=None):
        super(IBP_BNN, self).__init__()
        """
        input_size : Input Layer Dimension.
        hidden_size : List Representing the hidden layer sizes.
        output_size : Output Layer Dimenison.
        training_size : Number of training data points (for defining global multiplier for KL divergences).
        no_train_samples : Number of posterior samples to be taken while training for calculating gradients.
        no_test_sample : Number of posterior samples to be taken while testing for calculating gradients.
        prev_means : parameter means learned by training on previously seen tasks.
        prev_log_variances : parameter log variances learned by training on previously seen tasks. 
        prev_masks : IBP based masks learned for all the tasks previously seen.
        kl_mask : Union of all prev_masks (Used for prior masking).
        learning_rate : The learning rate used for the weight update.
        prior_mean : Initial prior mean.
        prior_variances : Initial prior variance.
        alpha : IBP concentration parameter.
        beta : IBP rate parameter.
        prev_pber : (Not required) Used as a initialization for bernoulli probabilty for current task mask.
        re_mode : Reparameterization (default is gumbel softmax)
        single_head : Weather to use task based seperate heads or single head for all the task.
        """

        # Input and Output placeholders
        '''
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.y = tf.placeholder(tf.float32, [None, output_size])
        self.temp = tf.placeholder(tf.float32, [1])
        self.task_idx = tf.placeholder(tf.int32)
        '''

        # Hyperparameters
        self.temp_prior = 0.05  # prior over the gumbel dist. (Temperature Prior)
        self.min_temp = 0.3  # minimum limit on temperature.
        self.eps = 10e-8  # Small value to avoid errors (e.g Div 0).
        self.curr_masks = []

        self.no_layers = len(hidden_size) + 1  # number of non-output layers.
        self.single_head = single_head  # Task id based multihead or single head output structure.
        self.no_train_samples = no_train_samples  # Training posterior sample size for approximation of Expectation.
        self.no_pred_samples = no_pred_samples  # Testing posterior sample size for approximation of Expectation.

        self.training_size = training_size  # Number of training data points (for normlaizing gradient values).

        self.init_var = -6.0  # Prior initializaion log variance.
        self.acts = acts  # Per layer Activations if are explicitly mentioned

        self.device = 'cuda'
        self.relu = F.relu
        self.learning_rate = learning_rate
        self.init_temp = 10.0  # Initial temperature for concrete distribution

        self.prior_mu = torch.tensor(prior_mean).float()  # Saving prior parameters for reusing while growing
        self.prior_var = torch.tensor(prior_var).float()

        # Parameter Initiliatlizations
        self.intialize(alpha, beta, input_size, hidden_size, output_size, prev_means,
                       prev_log_variances, prior_mean, prior_var, prev_pber, re_mode)

        # Initializing the session and optimizer for current model.
        self.assign_optimizer(learning_rate)

    # Done
    def intialize(self, alpha, beta, input_size, hidden_size, output_size, prev_means,
                  prev_log_variances, prior_mean, prior_var, prev_pber, re_mode):
        # Default values for IBP prior parameters per hidden layer.
        if (alpha is None):
            alpha = [4.0 for i in range(len(hidden_size))]
        if (beta is None):
            beta = [1.0 for i in range(len(hidden_size))]
        # Creating priors and current set of weights that have been learned.
        self.def_parameters(input_size, hidden_size, output_size, prev_means, prev_log_variances, prior_mean, prior_var)
        # Initilizing the model IBP stick breaking parameters.
        self.init_ibp_params(alpha, beta, re_mode, prev_pber)

    # Done
    def truncated_normal(self, shape, stddev=0.01):
        ''' Initialization : Function to return an truncated_normal initialized parameter'''
        uniform = torch.from_numpy(np.random.uniform(0, 1, shape)).float()
        return parameterized_truncated_normal(uniform, mu=0.0, sigma=stddev, a=-2 * stddev, b=2 * stddev)

    # Done    
    def constant(self, init, shape=None):
        ''' Initialization : Function to return an constant initialized parameter'''
        if (shape is None):
            return torch.tensor(init).float()
        return torch.ones(shape).float() * init

    # Done    
    def def_parameters(self, in_dim, hidden_size, out_dim, init_means, init_variances, prior_mean, prior_var):
        # A single list containing all layer sizes
        layer_sizes = deepcopy(hidden_size)
        layer_sizes.append(out_dim)
        layer_sizes.insert(0, in_dim)

        lvar_init = self.init_var  # initialization for log variances if not given.
        # Defining means and logvariances for weights and biases for model weights and priors. 
        # Variational Posterior parameters
        self.W_m = nn.ParameterList([])  # weight means
        self.b_m = nn.ParameterList([])  # bias means
        self.W_v = nn.ParameterList([])  # weight variances
        self.b_v = nn.ParameterList([])  # bias variances
        self.W_last_m = nn.ParameterList([])  # last layers weight mean
        self.b_last_m = nn.ParameterList([])  # last layers bias mean
        self.W_last_v = nn.ParameterList([])  # last layer weight var
        self.b_last_v = nn.ParameterList([])  # last layer bias var
        # Prior Parameters
        self.prior_W_m = []
        self.prior_b_m = []
        self.prior_W_v = []
        self.prior_b_v = []
        self.prior_W_last_m = []
        self.prior_b_last_m = []
        self.prior_W_last_v = []
        self.prior_b_last_v = []

        # Initialization for non-last layer parameters.
        for i in range(len(hidden_size)):

            din = layer_sizes[i]
            dout = layer_sizes[i + 1]

            Wi_m_val = self.truncated_normal([din, dout], stddev=0.01)
            bi_m_val = self.truncated_normal([dout], stddev=0.01)
            Wi_v_val = self.constant(lvar_init, shape=[din, dout])
            bi_v_val = self.constant(lvar_init, shape=[dout])
            Wi_m_prior = torch.zeros(din, dout) + torch.tensor(prior_mean).view(1, 1)
            bi_m_prior = torch.zeros(dout) + torch.tensor(prior_mean).view(-1)
            Wi_v_prior = torch.zeros(din, dout) + torch.tensor(prior_var).view(1, 1)
            bi_v_prior = torch.zeros(dout) + torch.tensor(prior_var).view(-1)

            if init_means is None or len(init_means[0]) == 0:  # If intial means were not present or given.
                pass
            else:  # Intial Means are present
                Wi_m_val = init_means[0][i]
                bi_m_val = init_means[1][i]
                Wi_m_prior = init_means[0][i]
                bi_m_prior = init_means[1][i]
                if init_variances is None or len(init_variances[0]) == 0:  # Means are given but variances are not known
                    pass
                else:  # Both means and variances were given/known.
                    Wi_v_val = init_variances[0][i]
                    bi_v_val = init_variances[1][i]
                    Wi_v_prior = init_variances[0][i].exp()
                    bi_v_prior = init_variances[1][i].exp()

            Wi_m = nn.Parameter(Wi_m_val)
            bi_m = nn.Parameter(bi_m_val)
            Wi_v = nn.Parameter(Wi_v_val)
            bi_v = nn.Parameter(bi_v_val)

            # Append Variational parameters
            self.W_m.append(Wi_m)
            self.b_m.append(bi_m)
            self.W_v.append(Wi_v)
            self.b_v.append(bi_v)
            # Append Prior parameters
            self.prior_W_m.append(Wi_m_prior)
            self.prior_b_m.append(bi_m_prior)
            self.prior_W_v.append(Wi_v_prior)
            self.prior_b_v.append(bi_v_prior)

        # Copying the previously trained last layer weights in case of multi head output
        if init_means is not None and init_variances is not None:
            init_Wlast_m = init_means[2]
            init_blast_m = init_means[3]
            init_Wlast_v = init_variances[2]
            init_blast_v = init_variances[3]
            no_tasks = len(init_Wlast_m)
            for i in range(no_tasks):  # Iterating over previous tasks to copy last layer
                W_i_m = init_Wlast_m[i]
                b_i_m = init_blast_m[i]
                W_i_v = init_Wlast_v[i]
                b_i_v = init_blast_v[i]
                Wi_m_prior = init_Wlast_m[i]
                bi_m_prior = init_blast_m[i]
                Wi_v_prior = init_Wlast_v[i].exp()
                bi_v_prior = init_blast_v[i].exp()

                Wi_m = nn.Parameter(W_i_m)
                bi_m = nn.Parameter(b_i_m)
                Wi_v = nn.Parameter(W_i_v)
                bi_v = nn.Parameter(b_i_v)
                # Copying last layer variational parameters for previous tasks
                self.W_last_m.append(Wi_m)
                self.b_last_m.append(bi_m)
                self.W_last_v.append(Wi_v)
                self.b_last_v.append(bi_v)
                # Copying last layer prior parameters for previous tasks
                self.prior_W_last_m.append(Wi_m_prior)
                self.prior_b_last_m.append(bi_m_prior)
                self.prior_W_last_v.append(Wi_v_prior)
                self.prior_b_last_v.append(bi_v_prior)

        # Adding the last layer weights for current task.
        if (not self.single_head or len(self.W_last_m) == 0):
            din = layer_sizes[-2]
            dout = layer_sizes[-1]
            if init_means is not None and init_variances is None:
                Wi_m_val = init_means[2][0]
                bi_m_val = init_means[3][0]
            else:
                Wi_m_val = self.truncated_normal([din, dout], stddev=0.01)
                bi_m_val = self.truncated_normal([dout], stddev=0.01)
            Wi_v_val = self.constant(lvar_init, shape=[din, dout])
            bi_v_val = self.constant(lvar_init, shape=[dout])

            Wi_m = nn.Parameter(Wi_m_val)
            bi_m = nn.Parameter(bi_m_val)
            Wi_v = nn.Parameter(Wi_v_val)
            bi_v = nn.Parameter(bi_v_val)

            Wi_m_prior = torch.zeros(din, dout) + torch.tensor(prior_mean).view(1, 1)
            bi_m_prior = torch.zeros(dout) + torch.tensor(prior_mean).view(-1)
            Wi_v_prior = torch.zeros(din, dout) + torch.tensor(prior_var).view(1, 1)
            bi_v_prior = torch.zeros(dout) + torch.tensor(prior_var).view(-1)

            # Variatonal Parameters for current task
            self.W_last_m.append(Wi_m)
            self.b_last_m.append(bi_m)
            self.W_last_v.append(Wi_v)
            self.b_last_v.append(bi_v)
            # Prior parameters for current task
            self.prior_W_last_m.append(Wi_m_prior)
            self.prior_b_last_m.append(bi_m_prior)
            self.prior_W_last_v.append(Wi_v_prior)
            self.prior_b_last_v.append(bi_v_prior)

        # Zipping Everything (current posterior parameters) into single entity (self.weights) 
        means = [self.W_m, self.b_m, self.W_last_m, self.b_last_m]
        logvars = [self.W_v, self.b_v, self.W_last_v, self.b_last_v]
        self.size = layer_sizes
        self.weights = [means, logvars]

    # Done 
    def extend_tensor(self, tensor, dims=None, extend_with=0.0):
        if (dims is None):
            return tensor
        else:
            if (len(tensor.shape) != len(dims)):
                print(tensor.shape, dims)
                assert 1 == 12

            if (len(dims) == 1):
                temp = tensor.cpu().detach().numpy()
                D = temp.shape[0]
                new_array = np.zeros(dims[0] + D) + extend_with
                new_array[:D] = temp
            elif (len(dims) == 2):
                temp = tensor.cpu().detach().numpy()
                D1, D2 = temp.shape
                new_array = np.zeros((D1 + dims[0], D2 + dims[1])) + extend_with
                new_array[:D1, :D2] = temp

            return torch.tensor(new_array).float().to(self.device)

    def dynamize_Adam(self, reset=False, amsgrad=False):
        with torch.no_grad():
            if (reset or self.optimizer == None):
                self.optimizer = self.get_optimizer(self.learning_rate, fix=False)
                self.optimizer.step()
            else:
                optim = self.optimizer
                newoptim = self.get_optimizer(self.learning_rate, fix=False)

                for i in range(len(optim.param_groups)):
                    group_old = optim.param_groups[i]
                    group_new = newoptim.param_groups[i]

                    for j in range(len(group_old['params'])):
                        params_old = group_old['params'][j]
                        params_new = group_new['params'][j]

                        amsgrad = group_old['amsgrad']
                        newoptim.param_groups[i]['amsgrad'] = amsgrad

                        state_old = optim.state[params_old]
                        state_new = newoptim.state[params_new]

                        state_new['step'] = torch.zeros_like(params_new.data)

                        state_new['exp_avg'] = torch.zeros_like(params_new.data)
                        state_new['exp_avg_sq'] = torch.zeros_like(params_new.data)

                        exp_avg = state_new['exp_avg']
                        exp_avg_sq = state_new['exp_avg_sq']
                        max_exp_avg_sq = None
                        if (amsgrad):
                            state_new['max_exp_avg_sq'] = torch.zeros_like(params_new.data)
                            max_exp_avg_sq = state_new['max_exp_avg_sq']

                        if (len(state_old) == 0):
                            pass
                        else:
                            if (len(state_old['exp_avg'].shape) == 2):
                                no, do = state_old['exp_avg'].shape
                                exp_avg[:no, :do] = state_old['exp_avg']
                                exp_avg_sq[:no, :do] = state_old['exp_avg_sq']
                                if (max_exp_avg_sq is not None):
                                    max_exp_avg_sq[:no, :do] = state_old['max_exp_avg_sq']
                                state_new['step'][:no, :do] = state_old['step']

                            elif (len(state_old['exp_avg'].shape) == 1):
                                no = state_old['exp_avg'].shape[0]
                                exp_avg[:no] = state_old['exp_avg']
                                exp_avg_sq[:no] = state_old['exp_avg_sq']
                                if (max_exp_avg_sq is not None):
                                    max_exp_avg_sq[:no] = state_old['max_exp_avg_sq']
                                state_new['step'][:no] = state_old['step']

                            else:
                                assert 1 == 2, 'error in dynamic adam'

                        state_new['exp_avg'] = exp_avg
                        state_new['exp_avg_sq'] = exp_avg_sq

                        newoptim.state[params_new] = state_new

                del optim
                self.optimizer = newoptim

                # Done

    def softplus(self, x, beta=1.0, threshold=20.0):
        return F.softplus(x, beta=beta, threshold=threshold)

    # Done
    def softplus_inverse(self, x, beta=1.0, threshold=20.0):
        eps = 10e-8
        mask = (x <= threshold).float().detach()
        xd1 = x * mask
        xd2 = xd1.mul(beta).exp().sub(1.0 - eps).log().div(beta)
        xd3 = xd2 * mask + x * (1 - mask)
        return xd3

    # Done
    def init_ibp_params(self, alpha, beta, re_mode, init_pber):
        # Reparameterization Mode (incase needed in future)
        self.reparam_mode = re_mode
        # Initializing the IBP parameters
        self.alphas = []  # prior concentration
        self.betas = []  # prior rate
        self._concs1, self._concs2 = nn.ParameterList([]), nn.ParameterList([])  # Posterior parameters based on p_bers.
        self._p_bers = nn.ParameterList([])  # Variational parameters for IBP posterior.

        # Iteration over layers to inialize IBP parameters per layer.
        for l in range(self.no_layers - 1):
            din, dout = self.size[l], self.size[l + 1]  # Layer dimenisons

            self.alphas.append(self.constant(alpha[l], shape=[dout]))  # Prior
            self.betas.append(self.constant(beta[l], shape=[dout]))  # Prior
            # Modified Variatonal Parameters contrained to be positive by taking inverse softplus then softplus.
            _conc1 = nn.Parameter(self.softplus_inverse(self.constant(np.ones((dout)) * alpha[l] + 1.0)))
            _conc2 = nn.Parameter(self.softplus_inverse(self.constant(np.ones((dout)) * beta[l] + 1.0)))
            # Real variationa parameters
            self._concs1.append(_conc1)
            self._concs2.append(_conc2)
            # Initializing the bernoulli probability variational parameters.
            if self.reparam_mode is 'gumbsoft':
                if (init_pber is None):  # If initlization given
                    _p_ber_init = self.logit(torch.tensor(np.float32(np.ones((din, dout)) * (0.5))))
                else:  # Default Initializaiton
                    _p_ber_init = self.constant(np.float32(init_pber[l]))
                _p_ber = nn.Parameter(_p_ber_init)
                # Taking sigmoid to constraint to bernoulli probability to range [0,1].
                self._p_bers.append(_p_ber)  # intermediate parameter.

    # Done                
    def _prediction(self, inputs, task_idx, no_samples, const_mask=False, temp=0.1):
        return self._prediction_layer(inputs, task_idx, no_samples, const_mask, temp=temp)

    # Done
    def sample_gauss(self, mean, logvar, sample_size):
        N, M = mean.shape

        device = self.device
        return (torch.randn(sample_size, N, M).to(device) * ((0.5 * logvar).exp().unsqueeze(0)) + mean.unsqueeze(
            0))  # samples xN x M

    # Not Done : Update the shrinked size usage
    def Linear(self, input, layer, no_samples=1, const_mask=False, temp=0.1, task_id=None):
        if (layer < len(self.size) - 2):
            params = [self.W_m[layer], self.W_v[layer], self.b_m[layer], self.b_v[layer]]
        else:
            if self.single_head:
                params = [self.W_last_m[0], self.W_last_v[0], self.b_last_m[0], self.b_last_v[0]]
            else:
                params = [self.W_last_m[task_id], self.W_last_v[task_id],
                          self.b_last_m[task_id], self.b_last_v[task_id]]

        shape = input.shape
        if len(shape) == 2:
            x = input.unsqueeze(1)
        else:
            x = input

        # x is Batch x sample_size|1 x Din
        A, B, C = x.shape
        # x = x.view(A,B,C,1).permute(0,1,3,2) # Batch x sample_size|1 x 1 x Din
        x = x.permute(1, 0, 2)
        if (B == 1):
            x = x.repeat(no_samples, 1, 1)

        weight_mean, weight_logvar, bias_mean, bias_logvar = params
        weights = self.sample_gauss(weight_mean, weight_logvar, no_samples)  # sample_size x Din x Dout
        biass = self.sample_gauss(bias_mean.unsqueeze(0), bias_logvar.unsqueeze(0),
                                  no_samples)  # sample_size x 1 x Dout
        _, din, dout = weights.shape

        # Sampling mask or bernoulli random varible
        if (layer < len(self.size) - 2):
            temp = temp[layer]
            vs, bs, logit_post = self.ibp_sample(layer, no_samples, temp=temp)  # Sampling through IBP
            self.KL_B.append(self._KL_B(layer, vs, bs, logit_post, temp=temp))
            # Generating masked weights and biases for current layer
            weight = weights * bs  # weights * ibp_mask
            bias = biass * (bs.max(dim=1)[0].unsqueeze(1))  # bias * ibp_mask
        else:
            weight = weights  # weights
            bias = biass  # bias
        ret = torch.bmm(x[:, :, :din], weight) + bias
        return ret.permute(1, 0, 2)

    # Done
    def _prediction_layer(self, x, task_id=-1, no_samples=1, const_mask=False, temp=0.1):
        if (self.W_last_m[0].is_cuda):
            self.device = 'cuda'

        activations = self.acts
        self.KL_B = []  # KL Divergence terms for the bernoulli distribution
        lsizes = self.size
        iterto = len(lsizes) - 1
        for i in range(iterto):
            if (i < iterto - 1):
                x = self.Linear(x, layer=i, no_samples=no_samples, const_mask=const_mask, temp=temp, task_id=task_id)

                if (activations is not None):
                    act = activations[i]
                    if (act == 'linear'):
                        pass
                    elif (act == 'relu'):
                        x = self.relu(x)
                else:
                    x = self.relu(x)
            else:
                x = self.Linear(x, layer=i, no_samples=no_samples, const_mask=const_mask, temp=temp, task_id=task_id)
        return x

    # Done
    def v_post_distr(self, layer, shape):
        # Real variationa parameters
        _conc1, _conc2 = self._concs1[layer], self._concs2[layer]
        conc1, conc2 = 1.0 / self.softplus(_conc1), 1.0 / self.softplus(_conc2)
        # dist = tod.beta.Beta(conc1, conc2)
        # return dist.sample(shape)
        eps = 10e-8
        rand = torch.rand(shape).unsqueeze(2).to(self.device) + eps
        a = conc1.view(-1).unsqueeze(0).unsqueeze(0) + eps
        b = conc2.view(-1).unsqueeze(0).unsqueeze(0) + eps
        samples = (1.0 - rand.log().mul(b).exp() + eps).log().mul(a).exp()
        # samples = (1.0-(rand+eps).pow(b)+eps).pow(a)
        K, din = shape
        dout = conc1.view(-1).shape[0]
        # print(samples.shape, K, din, dout)
        assert samples.shape == torch.Size([K, din, dout])
        if (samples.mean() != samples.mean()):
            print(conc1, conc2, _conc1, _conc2)
            assert 1 == 2

        return samples

    # Done
    def ibp_sample(self, l, no_samples, temp=0.1):
        din = self.size[l]  # current layer input dimenisions
        # Independently sampling current layer IBP posterior : K x din x dout
        vs = self.v_post_distr(l, shape=[no_samples, din])
        pis = torch.cumprod(vs, dim=2)  # Calcuting Pi's using nu's (IBP prior log probabilities): K x din x dout

        logit_post = self._p_bers[l].unsqueeze(0) + torch.log(pis + 10e-8)

        # Reparameterized bernoulli samples: K x din x dout
        bs = self.reparam_bernoulli(logit_post, no_samples, self.reparam_mode, temp=temp)
        return vs, bs, logit_post

    # Done    
    def reparam_bernoulli(self, logp, K, mode='gumbsoft', temp=0.1):
        if (temp is 0.1):
            assert 1 == 2
        din, dout = logp.shape[1], logp.shape[2]
        eps = self.eps  # epsilon a small value to avoid division error.
        # Sampling from the gumbel distribution and Reparameterizing
        if self.reparam_mode is 'gumbsoft':  # Currently we are doing bernoulli sampling so bernoulli samples.
            U = torch.tensor(np.reshape(np.random.uniform(size=K * din * dout), [K, din, dout])).float().to(self.device)
            L = ((U + eps).log() - (1 - U + eps).log())
            B = torch.sigmoid((L + logp) / temp.unsqueeze(0))
        return B

    # Done
    def def_cost(self, x, y, task_id, temp, fix=False):
        # KL Divergence and Objective Calculation.
        self.cost1 = self._KL_term().div(self.training_size)  # Gaussian prior KL Divergence
        self.cost2 = None
        self.cost3 = None
        if (not fix):
            self.cost2, pred = self._logpred(x, y, task_id, temp=temp)  # Log Likelihood
            # IBP KL Divergences
            self.cost3 = (self._KL_v() + sum(self.KL_B)).div(self.training_size)
            self.cost = self.cost1 - self.cost2 + self.cost3
            self.acc = (y.argmax(dim=-1) == F.softmax(pred, dim=-1).mean(1).argmax(dim=-1)).float().mean()
            return self.cost, self.cost1, self.cost2, self.cost3, self.acc

    # Done
    def _KL_term(self):
        # Returns the KL divergence for gaussian prior of parameters
        kl = [torch.tensor(0).to(self.device)]
        # Calculating KL Divergence for non output layer weights
        for i in range(self.no_layers - 1):
            din = self.size[i]
            dout = self.size[i + 1]

            kl_mask = torch.tensor(1.0).to(self.device)
            kl_mask_b = torch.tensor(1.0).to(self.device)

            m, v = self.W_m[i], self.W_v[i]

            # Prior mean and variance
            m0, v0 = (self.prior_W_m[i].to(self.device) * kl_mask), (
                    self.prior_W_v[i].to(self.device) * kl_mask + (
                    1.0 * (1 - kl_mask) * self.prior_var.to(self.device)))
            # print(v,v0)
            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * torch.sum((v0.log() - v))
            mu_diff_term = 0.5 * torch.sum(((v.exp() + (m0 - m) ** 2) / v0))
            # Adding the current KL Divergence
            kl.append(const_term + log_std_diff + mu_diff_term)

            # Calculating KL Divergence for non output layer biases
            m, v = self.b_m[i], self.b_v[i]
            m0, v0 = (self.prior_b_m[i].to(self.device) * kl_mask_b), (
                    self.prior_b_v[i].to(self.device) * kl_mask_b + (1 - kl_mask_b) * self.prior_var.to(self.device))
            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum((v0).log() - v)
            mu_diff_term = 0.5 * torch.sum((v.exp() + (m0 - m) ** 2) / (v0))

            # Adding the current KL Divergence
            kl.append(const_term + log_std_diff + mu_diff_term)
        # Calculating KL Divergence for output layer weights
        no_tasks = len(self.W_last_m)
        din = self.size[-2]
        dout = self.size[-1]
        for i in range(no_tasks):
            # Last Layer weights
            m, v = self.W_last_m[i], self.W_last_v[i]
            m0, v0 = (self.prior_W_last_m[i]).to(self.device), (self.prior_W_last_v[i]).to(self.device)
            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * torch.sum(v0.log() - v)
            mu_diff_term = 0.5 * torch.sum((v.exp() + (m0 - m) ** 2) / v0)
            kl.append(const_term + log_std_diff + mu_diff_term)
            # Last layer Biases
            m, v = self.b_last_m[i], self.b_last_v[i]
            m0, v0 = (self.prior_b_last_m[i]).to(self.device), (self.prior_b_last_v[i]).to(self.device)
            const_term = -0.5 * dout
            log_std_diff = 0.5 * torch.sum(v0.log() - v)
            mu_diff_term = 0.5 * torch.sum((v.exp() + (m0 - m) ** 2) / v0)
            kl.append(const_term + log_std_diff + mu_diff_term)
        return sum(kl)

    # Done
    def log_gumb(self, temp, log_alpha, log_sample):
        # Returns log probability of gumbel distribution
        eps = 10e-8
        exp_term = log_alpha + log_sample * (-temp.unsqueeze(0))
        log_prob = exp_term + (temp + eps).log().unsqueeze(0) - 2 * self.softplus(exp_term)
        return log_prob

    # Done
    def _KL_B(self, l, vs, bs, logit_post, temp=0.1):
        if (temp is 0.1):
            assert 1 == 2
        # Calculates the KL Divergence between two Bernoulli distributions 
        din, dout = self.size[l], self.size[l + 1]
        eps = 10e-8
        if self.reparam_mode is 'gumbsoft':
            pis = torch.cumprod(vs, dim=2)  # bernoulli prior probabilities : K x din x dout
            logit_gis = logit_post  # Logit of posterior probabilities : K x din x dout
            logit_pis = torch.log(pis + eps)  # Logit of prior probabilities : K x din x dout
            # logit_pis = self.logit(pis)# Logit of prior probabilities : K x din x dout
            log_sample = (bs + eps).log() - (1 - bs + eps).log()  # Logit of samples : K x din x dout
            tau = temp  # Gumbel softmax temperature for varaitonal posterior
            tau_prior = torch.tensor(self.temp_prior).to(self.device).repeat(tau.shape)
            # Calculating sample based KL Divergence betweent the two gumbel distribution
            b_kl1 = (self.log_gumb(tau, logit_gis, log_sample))  # posterior logprob samples : K x din x dout
            b_kl2 = (self.log_gumb(tau_prior, logit_pis, log_sample))  # prior logprob samples : K x din x dout
            b_kl = (b_kl1 - b_kl2).mean(0).mean(0).sum()  # .div(b_kl1.shape[0])
            # print(temp[0], self.temp_prior, b_kl)

        return b_kl

    # Done
    def _KL_v(self):
        # Calculates the KL Divergence between two Beta distributions
        v_kl = []
        euler_const = -torch.digamma(torch.tensor(1.0))
        for l in range(self.no_layers - 1):
            alpha, beta = self.alphas[l].to(self.device), self.betas[l].to(self.device)
            conc1, conc2 = self.softplus(self._concs1[l]), self.softplus(self._concs2[l])
            eps = 10e-8
            a_numpy = alpha.cpu().detach().numpy()
            b_numpy = np.ones_like(a_numpy)
            v_kl1 = ((conc1 - alpha) / (conc1 + eps)) * (-euler_const - torch.digamma(conc2) - 1.0 / (conc2 + eps))
            v_kl2 = ((conc1 + eps).log() + (conc2 + eps).log()) + torch.log(
                eps + torch.tensor(BETA(a_numpy, b_numpy))).to(self.device)
            v_kl3 = -(conc2 - 1) / (conc2 + eps)
            v_kl4 = torch.tensor(0.0).to(self.device)

            v_kl.append(sum(v_kl1 + v_kl2 + v_kl3 + v_kl4))

        ret = torch.sum(sum(v_kl))
        return ret

    # Done
    def _logpred(self, inputs, targets, task_idx, temp=0.1):
        # Returns the log likelihood of model w.r.t the current posterior
        # Predicitons for given input and task id : N x K x O
        pred = self._prediction(inputs, task_idx, self.no_train_samples, temp=temp)
        target = targets.unsqueeze(1).repeat(1, self.no_train_samples, 1)  # Formating desired output : N x K x O
        loss = torch.sum(- target * F.log_softmax(pred, dim=-1), dim=-1)
        log_lik = - (loss).mean()  # Crossentropy Loss
        return log_lik, pred

    # Done
    def assign_optimizer(self, learning_rate=0.01):
        self.optimizer = self.get_optimizer(learning_rate, False)

    # Done
    def get_optimizer(self, learning_rate=0.01, fix=False):
        if (not fix):
            # Non different optimizers for all variables togeather
            params = list(self.parameters())

            # return Adam(params, lr=learning_rate)
            normals = []
            harders = []

            for j, p in enumerate(params):
                found = False
                list_hard = list(self._p_bers) + list(self._concs1) + list(self._concs2)
                for i in range(len(list_hard)):
                    if (p is list_hard[i]):
                        harders.append(j)
                        found = True
                if (not found):
                    normals.append(j)

            # print(normals, harders)
            normal_params = [params[p] for p in normals]
            harder_params = [params[p] for p in harders]
            # ls =[p for p in list(self.parameters()) if p not in self._p_bers]
            opt_all = Adam(normal_params, lr=learning_rate)
            opt_all.add_param_group({
                'amsgrad': False,
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'lr': learning_rate * 10,
                'params': harder_params
            })
            return opt_all
        else:
            # Optimizer for training fixed mask model.
            opt_fix = Adam(self.parameters(), lr=learning_rate * 0.1)
            return opt_fix

    # Done
    def prediction(self, x_test, task_idx, const_mask):
        # Test model
        if not const_mask:
            # Predicitons for given input and task id : N x K x O
            return self._prediction(x_test, task_idx, self.no_train_samples)

    # Done
    def accuracy(self, x_test, y_test, task_id, batch_size=1000):
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
            acc = self.val_step(batch_x, batch_y, task_id, temp=0.1, fix=True)
            avg_acc += acc / total_batch
        print(avg_acc)

    # Done
    def prediction_prob(self, x_test, task_idx, batch_size=1000):
        # Returns the output probabilities for a given input
        with torch.no_grad():
            N = x_test.shape[0]
            if batch_size > N:
                batch_size = N

            costs = []
            cur_x_test = x_test
            total_batch = int(np.ceil(N * 1.0 / batch_size))
            prob = []
            for i in range(total_batch):
                start_ind = i * batch_size
                end_ind = np.min([(i + 1) * batch_size, N])
                batch_x = torch.tensor(cur_x_test[start_ind:end_ind, :]).float().to(self.device)
                pred_const_mask = self._prediction(batch_x, task_idx, self.no_pred_samples, const_mask=True,
                                                   temp=self.min_temp)

                pred = F.softmax(pred_const_mask, dim=-1).cpu().detach().numpy()
                prob.append(pred)
            prob = np.concatenate(prob, axis=0)

            return prob

    # Done
    def get_weights(self):
        # Returns the current weights of the model.
        means = [[W.cpu().detach().data for W in self.W_m], [W.cpu().detach().data for W in self.b_m],
                 [W.cpu().detach().data for W in self.W_last_m], [W.cpu().detach().data for W in self.b_last_m]]
        logvars = [[W.cpu().detach().data for W in self.W_v], [W.cpu().detach().data for W in self.b_v],
                   [W.cpu().detach().data for W in self.W_last_v], [W.cpu().detach().data for W in self.b_last_v]]
        ret = [means, logvars]
        return ret

    # Done
    def get_IBP(self):
        # Returns the current masks and IBP params of the model.
        alphas = [self.softplus(m).cpu().detach().numpy() for m in self._concs1]
        betas = [self.softplus(m).cpu().detach().numpy() for m in self._concs2]
        for i in range(len(alphas)):
            alphas[i] = max(max(alphas[i]), max(self.alphas[i].cpu().detach().numpy()))
            betas[i] = 1.0
        print("IBP prior alpha :", alphas)
        ret = [None, alphas, betas]
        return ret

    # Done
    def logit(self, x):
        eps = self.eps
        return (x + eps).log() - (1 - x + eps).log()

    def train_step_all(self, x, y, task_id, temp, fix=False):
        self.optimizer.zero_grad()
        cost, c1, c2, c3, acc = self.def_cost(x, y, task_id=task_id, temp=temp, fix=fix)
        cost.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return cost.data, c1.data, c2.data, c3.data

        # Done

    def val_step(self, x, y, task_id, temp, fix=False):
        # get cost
        with torch.no_grad():
            cost, c1, c2, c3, acc = self.def_cost(x, y, task_id=task_id, temp=temp, fix=fix)

        return c2, acc

    # Done 
    def batch_train(self, x_train, y_train, task_idx, no_epochs=100, batch_size=100, display_epoch=10, init_temp=10.0):
        '''
        This function trains the model on a given training dataset also splits it into training and validation sets.
        x_train : Trianing input Data
        y_train : Target data
        task_idx : Task id representing the task.
        no_epochs : Numebr fo epochs to train the model for.
        batch_size : mini batch size to be used for gradient updates.
        display_epoch : Frequency of displaying runtime estimates of model for diagnostics.
        two_opt : Use two different optimizer for the probs and weight parameters.
        '''
        if (self.W_last_m[0].is_cuda):
            self.device = 'cuda'
        num_sel_epochs = no_epochs - 1
        # self.optimizer = self.get_optimizer(self.learning_rate, False)
        # Training the data with vairiable masks..
        M_total = x_train.shape[0]  # Total size of the training data.
        val_size = int(0.04 * M_total)  # Validation size to Keep
        if (val_size >= x_train.shape[0]):
            val_size = 0
        perm_inds = np.arange(x_train.shape[0])
        np.random.shuffle(perm_inds)
        x_train, y_train = x_train[perm_inds], y_train[perm_inds]
        N = x_train.shape[0] - val_size
        if batch_size > N:
            batch_size = N

        costs = []
        temp = [self.constant(init_temp, shape=p.shape).to(self.device) for p in self._p_bers]
        perm_inds = np.arange(x_train.shape[0] - val_size)
        np.random.shuffle(perm_inds)

        # Variables to Keep track of the Model costs
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        eps = 10e-12
        div_temp = [((tmp.div(self.min_temp) + eps).log().div(no_epochs * total_batch)).exp().to(self.device) for tmp in
                    temp]  # exp(log(T/0.25)/x) = dt
        display_epoch = max(display_epoch, 1)
        # Only if validation set is not empty
        val_inds = np.arange(x_train.shape[0] - val_size, x_train.shape[0], 1)
        cur_x_val = torch.tensor(x_train[val_inds]).float().to(self.device)
        cur_y_val = torch.tensor(y_train[val_inds]).float().to(self.device)

        for epoch in range(1, no_epochs + 1):
            # Batchwise Training
            perm_inds = np.arange(x_train.shape[0] - val_size)
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]
            # Reinitializing Variables to Keep track of the Model costs
            avg_cost = 0
            avg_cost1 = 0.
            avg_cost2 = 0.
            avg_cost3 = 0.

            # Loop over all batches
            for i in range(total_batch):
                start_ind = i * batch_size
                end_ind = np.min([(i + 1) * batch_size, N])
                batch_x = torch.tensor(cur_x_train[start_ind:end_ind, :]).float().to(self.device)
                batch_y = torch.tensor(cur_y_train[start_ind:end_ind, :]).float().to(self.device)
                # Run optimization op (backprop) and cost op (to get loss value)

                c, c1, c2, c3 = self.train_step_all(batch_x, batch_y, task_idx, temp)

                # Compute average loss
                avg_cost += c / total_batch
                avg_cost1 += c1 / total_batch
                avg_cost2 += c2 / total_batch
                avg_cost3 += c3 / total_batch
                # Annealing the model temperature used in gumbel softmax re-parameterization
                temp = [torch.clamp(tmp / div_temp[g], min=self.min_temp) for g, tmp in enumerate(temp)]

            vc, acc = self.val_step(cur_x_val, cur_y_val, task_idx, temp)

            if epoch % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch),
                      "total cost=", "{:.4f}".format(avg_cost),
                      "kl_weight=", "{:.4f}".format(avg_cost1),
                      "likelihood=", "{:.4f}".format(- avg_cost2),
                      "ibp kl=", "{:.4f}".format(avg_cost3),
                      "val_loss=", "{:.4f}".format(vc),
                      "val_acc=", "{:.4f}".format(acc))

                print("Temperature :", [tmp.mean() for tmp in temp])
            costs.append(avg_cost)

        print("Optimization Finished!")
        return costs
