import torch
from torch.distributions import Bernoulli, HalfNormal, LogNormal, Normal, Gamma
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_y_hat(x, betas, sigs):

    a = Normal(0, sigs[0]).rsample((4,))
    b = Normal(0, sigs[1]).rsample((4,))
    c = Normal(0, sigs[2]).rsample((16,))
    d = Normal(0, sigs[3]).rsample((51,))
    e = Normal(0, sigs[4]).rsample((5,))

    y_hat = x[:, :5]@betas
    y_hat += (
        a[x[:, 5].long() - 1]
        + b[x[:, 6].long() - 1]
        + c[x[:, 7].long() - 1]
        + d[x[:, 8].long() - 1]
        + e[x[:, 9].long() - 1]
    )

    return y_hat


def compute_avg_precision(x, y, model_params, advi_mode):
    sample = {key: model_params[key].rsample((1000,)) for key in model_params.keys()}

    if advi_mode == "meanfield":
        betas = sample["beta"]
        sigs = sample["sig"].exp()
    elif advi_mode == "fullrank":
        betas = sample["params"][:, :5]
        sigs = sample["params"][:, 5:].exp()

    a = Normal(0, sigs[:, 0]).rsample((4,)).t()
    b = Normal(0, sigs[:, 1]).rsample((4,)).t()
    c = Normal(0, sigs[:, 2]).rsample((16,)).t()
    d = Normal(0, sigs[:, 3]).rsample((51,)).t()
    e = Normal(0, sigs[:, 4]).rsample((5,)).t()

    x = x.repeat((1000, 1, 1))
    y_hat = torch.einsum("bij,bj->bi", x[:, :, :5], betas)

    y_hat += (
        a.gather(dim=1, index=x[:, :, 5].long() - 1)
        + b.gather(dim=1, index=x[:, :, 6].long() - 1)
        + c.gather(dim=1, index=x[:, :, 7].long() - 1)
        + d.gather(dim=1, index=x[:, :, 8].long() - 1)
        + e.gather(dim=1, index=x[:, :, 9].long() - 1)
    )
    y_hat = torch.sigmoid(y_hat).mean(dim=0)

    return average_precision_score(y.detach().numpy(), y_hat.detach().numpy())


def loglike(y, x, beta, sig, full_data_size):
    y_hat = compute_y_hat(x, beta, sig.exp())
    return Bernoulli(logits=y_hat).log_prob(y).mean(0) * full_data_size


def log_prior_plus_logabsdet_J(beta, sig):
    # log prior for beta, evaluated at sampled values for beta
    lp_b = Normal(0, 5).log_prob(beta).sum()

    # log prior sig + log jacobian
    lp_log_sig = (Gamma(1, 1).log_prob(sig.exp()) + sig).sum()

    return lp_b + lp_log_sig


def log_q(model_params, params):
    out = 0.0
    for key in model_params:
        out += model_params[key].log_q(params[key])
    return out


def elbo(y_train, x_train, y_val, x_val, model_params, full_data_size, advi_mode):
    n_samples = 1
    samples = [{key: model_params[key].rsample() for key in model_params.keys()} for _ in range(n_samples)]
    
    loss = 0
    score = 0
    for sample in samples:
        current_loss = -log_q(model_params, sample)

        if advi_mode == "meanfield":
            beta = sample["beta"]
            sig = sample["sig"]

        elif advi_mode == "fullrank":
            beta = sample["params"][:5]
            sig = sample["params"][5:]

        current_loss += loglike(y_train, x_train, beta, sig, full_data_size)
        current_loss += log_prior_plus_logabsdet_J(beta, sig)

        current_score = compute_avg_precision(x_val, y_val, model_params, advi_mode)

        loss += current_loss
        score += current_score

    return -loss/n_samples , score/n_samples
