import numpy as np
import torch


class FullRankParams:
    def __init__(self, size, m=None, L=None):
        if m is None:
            m = torch.zeros(size)
        if L is None:
            L = torch.eye(size)
        self.mean = m
        self.L = L

        self.mean.requires_grad = True
        self.L.requires_grad = True

    def dist(self):
        return torch.distributions.MultivariateNormal(self.mean, self.L @ self.L.t())

    def rsample(self, n=torch.Size([])):
        return self.dist().rsample(n)

    def log_q(self, value):
        return self.dist().log_prob(value).sum()


class MeanFieldParams:
    def __init__(self, size, m=None, log_s=None):
        if m is None:
            m = torch.zeros(size)
        if log_s is None:
            log_s = torch.zeros(size)
        self.mean = m
        self.log_s = log_s

        self.mean.requires_grad = True
        self.log_s.requires_grad = True

    def dist(self):
        return torch.distributions.Normal(self.mean, self.log_s.exp())

    def rsample(self, n=torch.Size([])):
        return self.dist().rsample(n)

    def log_q(self, value):
        return self.dist().log_prob(value).sum()


# def get_learning_rate(i, s, grad, tau=1, alpha=0.1):
#     s = alpha * grad**2 + (1 - alpha) * s
#     rho = learning_rate * (i ** (-0.5 + 1e-16)) / (tau + np.sqrt(s))
#     return rho, s


def train_advi(
    x_train,
    y_train,
    x_val,
    y_val,
    model_params,
    elbo,
    full_data_size,
    max_iter,
    batch_size,
    lr=0.01,
    advi_mode="meanfield",
    schedule_model="max",
    elbo_hist = [],
    logpred_hist = [],
    beta_hist = [],
    sigma_hist = [],
):
    optimizer_params = [model_params[key].mean for key in model_params]
    if advi_mode == "meanfield":
        optimizer_params += [model_params[key].log_s for key in model_params]
    elif advi_mode == "fullrank":
        optimizer_params += [model_params[key].L for key in model_params]

    optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=schedule_model, patience=200
    )

    for i in range(max_iter):
        if i % 250 == 0:
            if advi_mode == "meanfield":
                beta_hist.append(list(model_params['beta'].mean.detach().numpy()))
                sigma_hist.append(list(model_params['sig'].mean.exp().detach().numpy()))
            else:
                beta_hist.append(list(model_params['params'].mean.detach().numpy())[:5])
                sigma_hist.append(list(model_params['params'].mean.exp().detach().numpy())[5:])
        try:
            idx = np.random.choice(full_data_size, batch_size)
            loss, metric = elbo(
                y_train[idx],
                x_train[idx, :],
                y_val,
                x_val,
                model_params,
                full_data_size=full_data_size,
                advi_mode=advi_mode,
            )
            elbo_hist.append(loss.item())
            logpred_hist.append(metric.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(logpred)

            # Print progress bar
            print(
                f"{i+1}/{max_iter} loss: {loss:.3f}, metric: {metric:.3f}, lr: {optimizer.param_groups[-1]['lr']:.8f}",
                end="\r",
            )
        except KeyboardInterrupt:
            print()
            print("breaking")
            break
        

    return elbo_hist, logpred_hist, beta_hist, sigma_hist
