import torch
from torch.distributions import Gamma, HalfNormal, LogNormal, Normal


def compute_mse(y, x, model_params, advi_mode):
    # print(model_params["params"].rsample((1000, )))
    sample = {key: model_params[key].rsample((1000,)) for key in model_params.keys()}

    if advi_mode == "meanfield":
        betas = sample["beta"]
        sig = sample["sig"].exp()
    if advi_mode == "fullrank":
        betas = sample["params"][:, :10]
        sig = sample["params"][:, 10:].exp()

    x = x.repeat((1000, 1, 1))
    mean = torch.einsum("bij,bj->bi", x, betas)
    y_hat = Normal(mean, sig).sample().mean(0)

    return torch.square(y_hat - y).mean()


def loglike(y, x, beta, sig, full_data_size):
    return Normal(x@beta, sig.exp()).log_prob(y).mean(0) * full_data_size


def log_prior_plus_logabsdet_J(beta, sig):
    # log prior for beta, evaluated at sampled values for beta
    lp_b = Normal(0, 10).log_prob(beta).sum()

    # log prior sig + log jacobian
    lp_log_sig = (Gamma(1, 1).log_prob(sig.exp()) + sig).sum()

    return lp_b + lp_log_sig


def log_q(model_params, params):
    out = 0.0
    for key in model_params:
        out -= model_params[key].log_q(params[key])
    return out


def elbo(
    y_train, x_train, y_val, x_val, model_params, full_data_size, advi_mode="meanfield"
):
    sample = {key: model_params[key].rsample() for key in model_params.keys()}
    out = log_q(model_params, sample)

    if advi_mode == "meanfield":
        beta = sample["beta"]
        sig = sample["sig"]

    elif advi_mode == "fullrank":
        beta = sample["params"][:10]
        sig = sample["params"][10:]

    out += loglike(y_train, x_train, beta, sig, full_data_size)
    out += log_prior_plus_logabsdet_J(beta, sig)

    mse = compute_mse(y_val, x_val, model_params, advi_mode)

    return -out / full_data_size, mse
