import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    """Returns the computed log-likelihood

    Args:
        x (_type_): the sampled weights or biases
        mu (_type_): mean of gaussian distribution
        sigma (_type_): standard deviation of gaussian dist
        do_sum (bool, optional): _description_. a boolean indicating whether to sum the log-likelihoods
        over all elements or to take the mean.

    Returns:
        _type_: Gaussian Log likelihood
    """
    cte_term = -(0.5) * np.log(2 * np.pi)  # constant term
    det_sig_term = -torch.log(sigma)  # Determinant term
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner**2)

    if do_sum:
        out = (cte_term + det_sig_term + dist_term).sum()  # sum over all weights
    else:
        out = (cte_term + det_sig_term + dist_term).mean()
    return out


class isotropic_gauss_prior(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        self.cte_term = -(0.5) * np.log(2 * np.pi)
        self.det_sig_term = -np.log(self.sigma)

    def loglike(self, x, do_sum=True):

        dist_term = -(0.5) * ((x - self.mu) / self.sigma) ** 2
        if do_sum:
            return (self.cte_term + self.det_sig_term + dist_term).sum()
        else:
            return (self.cte_term + self.det_sig_term + dist_term).mean()


class isotropic_mixture_gauss_prior(object):
    def __init__(self, mu1=0, mu2=0, sigma1=0.1, sigma2=1.5, pi=0.5):
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.pi1 = pi
        self.pi2 = 1 - pi

        self.cte_term = -(0.5) * np.log(2 * np.pi)

        self.det_sig_term1 = -np.log(self.sigma1)

        self.det_sig_term2 = -np.log(self.sigma2)

    def loglike(self, x, do_sum=True):

        dist_term1 = -(0.5) * ((x - self.mu1) / self.sigma1) ** 2
        dist_term2 = -(0.5) * ((x - self.mu2) / self.sigma2) ** 2

        if do_sum:
            return (
                torch.log(
                    self.pi1
                    * torch.exp(self.cte_term + self.det_sig_term1 + dist_term1)
                    + self.pi2
                    * torch.exp(self.cte_term + self.det_sig_term2 + dist_term2)
                )
            ).sum()
        else:
            return (
                torch.log(
                    self.pi1
                    * torch.exp(self.cte_term + self.det_sig_term1 + dist_term1)
                    + self.pi2
                    * torch.exp(self.cte_term + self.det_sig_term2 + dist_term2)
                )
            ).mean()


class BayesLinearNormalDist(nn.Module):

    def __init__(self, num_inp_feats, num_out_feats, prior_class, with_bias=True):
        super(BayesLinearNormalDist, self).__init__()
        self.num_inp_feats = num_inp_feats
        self.num_out_feats = num_out_feats
        self.prior = prior_class
        self.with_bias = with_bias

        # Learnable parameters -> Initialisation is set empirically.
        self.W_mu = nn.Parameter(
            torch.Tensor(self.num_inp_feats, self.num_out_feats).uniform_(-0.1, 0.1)
        )
        self.W_p = nn.Parameter(
            torch.Tensor(self.num_inp_feats, self.num_out_feats).uniform_(-3, -2)
        )

        self.b_mu = nn.Parameter(torch.Tensor(self.num_out_feats).uniform_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.num_out_feats).uniform_(-3, -2))

    def forward(self, X, sample=0, local_rep=False, ifsample=True):

        if not ifsample:  # When training return MLE of w for quick validation
            # pdb.set_trace()
            if self.with_bias:
                output = torch.mm(X, self.W_mu) + self.b_mu.expand(
                    X.size()[0], self.num_out_feats
                )
            else:
                output = torch.mm(X, self.W_mu)
            return output, torch.Tensor([0]).cuda()

        else:
            if not local_rep:
                # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
                # the same random sample is used for every element in the minibatch
                # pdb.set_trace()
                W_mu = self.W_mu.unsqueeze(1).repeat(1, sample, 1)
                W_p = self.W_p.unsqueeze(1).repeat(1, sample, 1)

                b_mu = self.b_mu.unsqueeze(0).repeat(sample, 1)
                b_p = self.b_p.unsqueeze(0).repeat(sample, 1)

                eps_W = W_mu.data.new(W_mu.size()).normal_()
                eps_b = b_mu.data.new(b_mu.size()).normal_()

                if not ifsample:
                    eps_W = eps_W * 0
                    eps_b = eps_b * 0

                # sample parameters
                std_w = 1e-6 + f.softplus(W_p, beta=1, threshold=20)
                std_b = 1e-6 + f.softplus(b_p, beta=1, threshold=20)

                W = W_mu + 1 * std_w * eps_W
                b = b_mu + 1 * std_b * eps_b

                if self.with_bias:
                    lqw = isotropic_gauss_loglike(
                        W, W_mu, std_w
                    ) + isotropic_gauss_loglike(b, b_mu, std_b)
                    lpw = self.prior.loglike(W) + self.prior.loglike(b)
                else:
                    lqw = isotropic_gauss_loglike(W, W_mu, std_w)
                    lpw = self.prior.loglike(W)

                W = W.view(W.size()[0], -1)
                b = b.view(-1)

                if self.with_bias:
                    # wx + b
                    output = torch.mm(X, W) + b.unsqueeze(0).expand(
                        X.shape[0], -1
                    )  # (batch_size, num_out_featsput)
                else:
                    output = torch.mm(X, W)

            else:
                W_mu = self.W_mu.unsqueeze(0).repeat(X.size()[0], 1, 1)
                W_p = self.W_p.unsqueeze(0).repeat(X.size()[0], 1, 1)

                b_mu = self.b_mu.unsqueeze(0).repeat(X.size()[0], 1)
                b_p = self.b_p.unsqueeze(0).repeat(X.size()[0], 1)
                # pdb.set_trace()
                eps_W = W_mu.data.new(W_mu.size()).normal_()
                eps_b = b_mu.data.new(b_mu.size()).normal_()

                # sample parameters
                std_w = 1e-6 + f.softplus(W_p, beta=1, threshold=20)
                std_b = 1e-6 + f.softplus(b_p, beta=1, threshold=20)

                W = W_mu + 1 * std_w * eps_W
                b = b_mu + 1 * std_b * eps_b

                # W = W.view(W.size()[0], -1)
                # b = b.view(-1)
                # pdb.set_trace()

                if self.with_bias:
                    output = (
                        torch.bmm(X.view(X.size()[0], 1, X.size()[1]), W).squeeze() + b
                    )  # (batch_size, num_out_featsput)
                    lqw = isotropic_gauss_loglike(
                        W, W_mu, std_w
                    ) + isotropic_gauss_loglike(b, b_mu, std_b)
                    lpw = self.prior.loglike(W) + self.prior.loglike(b)
                else:
                    output = torch.bmm(X.view(X.size()[0], 1, X.size()[1]), W).squeeze()
                    lqw = isotropic_gauss_loglike(W, W_mu, std_w)
                    lpw = self.prior.loglike(W)

            return output, lqw - lpw

    def extra_repr(self):
        """This method provides additional
        information about the module, useful for debugging and logging.
        """
        return "in_channel={num_inp_feats}, out_channel={num_out_feats}, with_bias={with_bias}, prior={prior}".format(
            **self.__dict__
        )
