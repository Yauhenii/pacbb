## Overview
The `core` package provides foundational components for the PAC-Bayes framework. By assembling the pieces within `core`, you can build, train, and evaluate
probabilistic neural networks under the PAC-Bayes paradigm.

## Subpackages and Modules
- **distribution/** for modeling probability distributions over NN parameters
- **layer/** for probabilistic neural network layers
- **bound/** for PAC-Bayes bounds
- **objective/** for combining empirical loss and KL terms
- **split_strategy/** for partitioning datasets (prior/posterior/bound)
- **utils/** for miscellaneous helpers
- Higher-level modules (*loss.py*, *metric.py*, *model.py*, *risk.py*, *training.py*)
  that stitch these components together.

## PAC-Bayes Framework: Theory and Practice

**Note:** The explanations below are demonstrated using an actual working script. You can directly run it from the repository:  
[https://github.com/Yauhenii/pacbb/blob/main/scripts/generic_train.py](https://github.com/Yauhenii/pacbb/blob/main/scripts/generic_train.py)  
This script implements a complete training pipeline including prior/posterior training, data splitting, and final PAC-Bayes bound certification.

---

### Introduction

This framework provides tools to build, train, and evaluate probabilistic neural networks under the PAC-Bayes paradigm, combining theoretical guarantees with practical utility. In PAC-Bayes, we treat each neural network parameter as a random variable, so the network becomes a distribution over possible parameter values. The approach offers generalization bounds of the form:

$$
\mathbb{E}_{\theta \sim \rho} [R(\theta)] 
\;\;\leq\;\;
\underbrace{ \mathbb{E}_{\theta \sim \rho} [r(\theta)] }_{\text{empirical loss}}
\;+\;\mathrm{Complexity}\bigl(KL(\rho \| \pi),\, n\bigr),
$$

where $\rho$ is a posterior distribution over parameters, 
$\pi$ is a prior, $r(\theta)$ is the empirical loss 
(e.g., on a training or bound dataset), and 
$R(\theta)$ is the (unknown) true risk.

In practice, one typically:

1. Chooses a model architecture in PyTorch and attaches probabilistic distributions (for example, Gaussians) to its parameters.  
2. Splits data to allow for prior training, posterior training, and a dedicated set for bound evaluation.  
3. Trains the prior distribution, optionally measuring and bounding its risk.  
4. Initializes and refines the posterior distribution from the learned prior.  
5. Computes or certifies the final PAC-Bayes bound on the bound set.

Below, we break down each code component in an example pipeline. Along the way, we point out key theoretical ideas (for example, the bounded-loss requirement, data splitting for data-dependent priors, or the role of KL divergence in each objective). This corresponds to the working script `generic_train.py` from our repository.


### Losses in PAC-Bayes

A core requirement of PAC-Bayes is that the loss function must be bounded, typically in $[0,1]$. For classification, the 0-1 loss is a natural choice but is not differentiable. Instead, implementations often employ a bounded negative log-likelihood:

$$
l(f(X), Y) = \frac{-\log \max\{P(Y\mid f(X)), p_{\min}\}}{\log(1/p_{\min})}
$$

so that $l$ remains in $[0,1]$.

In practice, you might have a factory of losses:
```python
# Example: Instantiating losses and LossFactory
loss_factory = LossFactory()
losses = {
    "nll_loss": loss_factory.create("nll_loss"),
    "scaled_nll_loss": loss_factory.create("scaled_nll_loss"),
    "01_loss": loss_factory.create("01_loss")
}
```
Different variants can be plugged in, as long as they are implemented under the `AbstractLoss` interface (which ensures a forward pass that outputs a bounded value).

### Metrics

Metrics serve solely as evaluation tools, such as classification accuracy, the F1-score, or custom-defined measures. While they do not necessarily have bounded values or appear explicitly in the PAC-Bayes inequality, they provide essential insights for assessing model performance.

```python
# Example: Creating metrics for evaluation (not necessarily bounded).
metric_factory = MetricFactory()
metrics = {
    "accuracy_micro_metric": metric_factory.create("accuracy_micro_metric"),
    "accuracy_macro_metric": metric_factory.create("accuracy_macro_metric"),
    "f1_micro_metric": metric_factory.create("f1_micro_metric"),
    "f1_macro_metric": metric_factory.create("f1_macro_metric")
}
```

These can be accuracy, precision, confusion matrices, etc.

### Bounds and the Role of KL Divergence

PAC-Bayes bounds typically look like:

$$
\mathbb{E}_{\theta\sim\rho}[R(\theta)]
\,\le\,
\mathbb{E}_{\theta\sim\rho}[r(\theta)]
\;+\;
\mathrm{Complexity}(KL(\rho \| \pi),\, n, \delta),
$$

where:

- $KL(\rho \|\pi)$ measures how far the posterior $\rho$ is from the prior $\pi$.
- $n$ is the effective number of data samples used in the bound. In a data-splitting scenario, it often corresponds to the size of the bound set, and in code it is usually deduced from the `bound_loader` (for example by `len(bound_loader.dataset)`).
- $\delta$ is a confidence parameter. The bound holds with probability at least $1 - \delta$.

In the snippet below, `bound_delta` corresponds to the theoretical $\delta$. The parameter `loss_delta` is a separate hyperparameter for bounding or adjusting the loss term in practice. For instance, one might cap a negative log-likelihood at $\log(1/p_{\min})$ or include a data-driven threshold. While usage can vary by codebase, typically:

- `bound_delta` is the $\delta$ from the PAC-Bayes statement.
- `loss_delta` handles or scales losses, especially when the loss must be numerically bounded.

```python
# Example: Instantiating a PAC-Bayes Bound with delta parameters
bound_factory = BoundFactory()
bounds = {
    "kl": bound_factory.create(
        "kl",
        bound_delta=0.025,   
        loss_delta=0.01
    ),
    "mcallester": bound_factory.create(
        "mcallester",
        bound_delta=0.025,
        loss_delta=0.01
    )
}
```

### Data Splitting & Prior Selection

To build an informed prior without violating the PAC-Bayes assumptions, we split the available data into three subsets: $\mathcal{S}_{\text{prior}}$, $\mathcal{S}_{\text{bound}}$, and $\mathcal{S}_{\text{posterior}}$. Specifically:

- The prior is trained solely on $\mathcal{S}_{\text{prior}}$. 
- The PAC-Bayes bound is computed later on $\mathcal{S}_{\text{bound}}$ only, ensuring that the prior was chosen independently of these samples.
- The posterior can be refined on additional data (potentially uniting $\mathcal{S}_{\text{prior}}$ and $\mathcal{S}_{\text{posterior}}$) after the prior has been established. 

By specifying `prior_type="learnt"`, we indicate in the code below that we intend to select or train a prior distribution from the $\mathcal{S}_{\text{prior}}$ samples, rather than fixing a data-independent prior. The fraction `prior_percent=0.7` means that 70% of the `train_percent` samples go to the prior, while the remaining 30% form $\mathcal{S}_{\text{bound}}$ (with additional splitting if validation/test sets are used). Once split, $\mathcal{S}_{\text{prior}}$ is passed to a training to learn the prior distribution’s parameters.

```python
# Example: Splitting dataset into prior/posterior/bound (plus val/test).
data_loader_factory = DataLoaderFactory()
loader = data_loader_factory.create(
    "cifar10",
    dataset_path="./data/cifar10"
)

strategy = PBPSplitStrategy(
    prior_type="learnt",   # we want to learn a prior from data
    train_percent=1.0,     # use entire dataset for (prior + bound)
    val_percent=0.0,       # no separate validation here
    prior_percent=0.7,     # 70% of training data for prior, 30% for bound
    self_certified=True    # ensures the prior set is disjoint from bound set
)
strategy.split(loader, split_config={
    "batch_size": 250, 
    "dataset_loader_seed": 112,
    "seed": 111
})
```
Training the prior on $\mathcal{S}_{\text{prior}}$ is what we refer to as "Prior Selection". This data-driven selection ensures the prior distribution is tuned to some portion of the data while still preserving a separate $\mathcal{S}_{\text{bound}}$ for unbiased bound computation.

### Building the Probabilistic Model
Following [Blundell et al., 2015](https://arxiv.org/abs/1505.05424), each parameter of a PyTorch layer can be represented as a Gaussian with diagonal covariance:

$$
w_i \sim \mathcal{N}(\mu_i, \sigma_i^2).
$$

One can then reparametrize or sample from these Gaussians at each forward pass, using techniques such as the local reparametrization trick, Flipout, etc. The code below shows how we convert a deterministic model into a `ProbNN`:

```python
# Example: Attaching prior distributions to a model and converting it 
# to a Probabilistic NN.
model_factory = ModelFactory()
model = model_factory.create(
    "conv15",
    dataset="cifar10",
    in_channels=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(110)

sigma_value = 0.01
rho_init = torch.log(torch.exp(torch.Tensor([sigma_value])) - 1)

# Create a reference prior (not trainable) and an initial prior (trainable)
prior_prior = from_zeros(
    model=model,
    rho=rho_init,
    distribution=GaussianVariable,
    requires_grad=False
)

prior = from_random(
    model=model,
    rho=rho_init,
    distribution=GaussianVariable,
    requires_grad=True
)

dnn_to_probnn(model, prior, prior_prior)
model.to(device)
```

### Training the Prior

To train the prior distribution $\pi$, we minimize a PAC-Bayes-inspired objective (for instance `fquad`, `bbb`, etc.). Typically, it combines the empirical loss on the prior subset and the KL divergence to a fixed reference. Such training can yield a **data-dependent** prior while preserving correctness if the bound subset is disjoint.


```python
# Example: Prior training with a chosen PAC-Bayes objective
# (using components created in the sections above)
train_params = {
    "lr": 0.05,
    "momentum": 0.95,
    "epochs": 100,
    "seed": 1135,
    "num_samples": strategy.prior_loader.batch_size * len(strategy.prior_loader)
}

objective_factory = ObjectiveFactory()
objective = objective_factory.create(
    "fclassic",
    delta=0.025,
    kl_penalty=0.01
)

train(
    model=model,
    posterior=prior,
    prior=prior_prior,
    objective=objective,
    train_loader=strategy.prior_loader,
    val_loader=strategy.val_loader,
    parameters=train_params,
    device=device,
    wandb_params={"log_wandb": True, "name_wandb": "Prior Train"}
)
```

### Training the Prior

At this point, you could evaluate metrics or compute a PAC-Bayes bound on the prior:

```python
# Example: Prior training with a chosen PAC-Bayes objective
# (using components created in the sections above)
if strategy.test_loader is not None:
    evaluated_metrics = evaluate_metrics(
        model=model,
        metrics=metrics,
        test_loader=strategy.test_loader,
        num_samples_metric=1000,
        device=device,
        pmin=5.0e-05
    )

certified_risk = certify_risk(
    model=model,
    bounds=bounds,
    losses=losses,
    posterior=prior,
    prior=prior_prior,
    bound_loader=strategy.bound_loader,
    num_samples_loss=1000,
    device=device,
    pmin=5.0e-05
)
```

### Creating and Training the Posterior

Next, we initialize the **posterior** $\rho$ from the learned prior weights and refine it, typically on a larger dataset or the same one. We again select an objective that balances empirical loss plus KL$(\rho \|\pi)$:

```python
# Example: Posterior initialization and training via PAC-Bayes objective.
# (using components created in the sections above)
posterior_prior = from_copy(
    dist=prior, 
    distribution=GaussianVariable, 
    requires_grad=False
)
posterior = from_copy(
    dist=prior,
    distribution=GaussianVariable,
    requires_grad=True
)
update_dist(model, weight_dist=posterior, prior_weight_dist=posterior_prior)
model.to(device)

train_params = {
    "lr": 0.001,
    "momentum": 0.9,
    "epochs": 1,
    "seed": 1135,
    "num_samples": strategy.posterior_loader.batch_size *
    len(strategy.posterior_loader)
}

objective = objective_factory.create(
    "fclassic",
    delta=0.025,
    kl_penalty=1.0
)

train(
    model=model,
    posterior=posterior,
    prior=posterior_prior,
    objective=objective,
    train_loader=strategy.posterior_loader,
    val_loader=strategy.val_loader,
    parameters=train_params,
    device=device,
    wandb_params={"log_wandb": True, "name_wandb": "Posterior Train"}
)
```

### Evaluating and Certifying the Posterior

Finally, we evaluate the **posterior** $\rho$ on the test set (if available) and compute the final PAC-Bayes bound on the **bound set**:

```python
# Example: Evaluating the final posterior (metrics and bound)
# (using components created in the sections above)
if strategy.test_loader is not None:
    posterior_evaluated_metrics = evaluate_metrics(
        model=model,
        metrics=metrics,
        test_loader=strategy.test_loader,
        num_samples_metric=1000,
        device=device,
        pmin=5.0e-05
    )

posterior_certified_risk = certify_risk(
    model=model,
    bounds=bounds,
    losses=losses,
    posterior=posterior,
    prior=posterior_prior,
    bound_loader=strategy.bound_loader,
    num_samples_loss=1000,
    device=device,
    pmin=5.0e-05
)

```

### Conclusion

By aligning each stage of model building and training with PAC-Bayes theory, you obtain a rigorous pipeline for **risk certification**. These examples show how to set up and run every piece:

- **Bounded losses** ensure validity of PAC-Bayes.
- **Split strategies** maintain data-disjoint sets for prior vs bound.
- **Distribution layers** convert a PyTorch model into a probabilistic NN.
- **Objectives** combine empirical loss and $KL(\rho\|\pi)$.
- **Bounds** let you certify risk using, e.g., McAllester or Seeger’s **PAC-Bayes-kl**.

For deeper details on advanced divergences, specialized losses, or data splitting variants, consult the submodule references and the accompanying theoretical expositions.


---

## Links

- **Documentation**: [https://yauhenii.github.io/pacbb/core.html](https://yauhenii.github.io/pacbb/core.html)
- **PyPI**: [https://pypi.org/project/pacbb/](https://pypi.org/project/pacbb/)
- **Source Code**: [https://github.com/yauhenii/pacbb](https://github.com/yauhenii/pacbb)
