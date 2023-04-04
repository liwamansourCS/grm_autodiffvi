import torch
from sklearn.metrics import average_precision_score
from torch.distributions import Bernoulli, Gamma, HalfNormal, Normal


def compute_avg_precision(x, y, model_params, advi_mode):
    if advi_mode == "meanfield":
        betas = model_params["beta"].mean.detach().numpy()
    elif advi_mode == "fullrank":
        betas = model_params["params"].mean.detach().numpy()

    sample = {key: model_params[key].rsample((1000,)) for key in model_params.keys()}

    if advi_mode == "meanfield":
        betas = sample["beta"]
    elif advi_mode == "fullrank":
        betas = sample["params"]

    x = x.repeat((1000, 1, 1))
    y_hat = torch.einsum("bij,bj->bi", x, betas)

    y_pred = torch.sigmoid(y_hat)
    return average_precision_score(y, y_pred.detach().numpy())


def loglike(y, x, betas, full_data_size):
    return Bernoulli(logits=x @ betas).log_prob(y).mean(0) * full_data_size


def log_prior_plus_logabsdet_J(betas):
    lp_b = Normal(0.0, 5.0).log_prob(betas).sum()
    return lp_b


def log_q(model_params, params):
    out = 0.0
    for key in model_params:
        out += model_params[key].log_q(params[key])
    return out


def elbo(y_train, x_train, y_val, x_val, model_params, full_data_size, advi_mode):
    sample = {key: model_params[key].rsample() for key in model_params.keys()}
    out = -log_q(model_params, sample)

    if advi_mode == "meanfield":
        beta = sample["beta"]
    elif advi_mode == "fullrank":
        beta = sample["params"]

    out += loglike(y_train, x_train, beta, full_data_size)
    out += log_prior_plus_logabsdet_J(beta)

    logpred = compute_avg_precision(x_val, y_val, model_params, advi_mode)

    return -out / full_data_size, logpred
