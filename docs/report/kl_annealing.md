# KL Annealing

This note explains the theory behind **KL annealing**, why it matters for a VAE, and exactly how we use it in this project. The theoretical references (equation numbers) point to the deck `docs/slides/Mathematical_Introduction_to_Generative_AI-47.pdf`.

## 1. Where the KL term comes from

Training a VAE means maximizing the log-likelihood of the data under the model $\tilde{\rho}_X(x)$. Because the encoder that this objective implies is intractable, we approximate the posterior $\tilde{\rho}_{Z\mid X}(z, x)$ by a multivariate independent normal (a "MIND") $\mathcal{N}(z, \mu_E(x), \sigma_E(x)^2 I)$ and apply **Jensen's inequality** to obtain the **Evidence Lower Bound (ELBO)** (eqs. 36–38). Maximizing the ELBO instead of the true objective is "the coup": it is easier to optimize and, crucially, the encoder functions $\mu_E, \sigma_E$ finally play an active role.

After fixing $\sigma_D = 1$ and applying the **reparameterization trick** $Z(x) = \mu_E(x) + \sigma_E(x) W_E$, the ELBO turns into a minimization problem (eqs. 42/48):

$$
\min_{\mu_D,\,\mu_E,\,\sigma_E}\; C + \sum_{i=1}^{N} \Big\{\, R(x_i;\,\mu_D,\mu_E,\sigma_E) \;+\; L(x_i;\,\mu_E,\sigma_E) \,\Big\}
$$

The objective is a sum of two terms with very different jobs:

- **$R$ — reconstruction error** (eqs. 43–46). With $\sigma_E = 0$ it reduces to the usual autoencoder reconstruction $\tfrac{1}{2}\lVert x - \mu_D(\mu_E(x))\rVert^2$; with $\sigma_E > 0$ it reconstructs from a *cluster* of latent points around $\mu_E(x)$, which is what makes the latent space continuous. It wants **faithful reconstructions**.
- **$L$ — latent distribution error** (eq. 47), the **KL divergence** $D_{\mathrm{KL}}\!\big(\tilde{\rho}_{Z\mid X}(\cdot, x)\,\Vert\,\mathcal{N}(0, I)\big)$:

$$
L(x;\,\mu_E,\sigma_E) \;=\; m\left(\frac{\sigma_E(x)^2 - 1}{2} - \ln \sigma_E(x)\right) + \frac{\lVert \mu_E(x)\rVert^2}{2}.
$$

A key fact (slide 58): $L \ge 0$, and $L = 0$ **iff** $\mu_E(x) = 0$ and $\sigma_E(x) = 1$. The $L$ term pushes each per-sample posterior toward the standard normal prior $\mathcal{N}(0, I)$, which is what gives the latent space its organized, dense structure.

## 2. The conflict, and posterior collapse

The two terms can **work against each other** (slides 54, 63). $R$ wants the latent code to carry as much information about $x$ as possible (sharp reconstructions); $L$ acts in an *ultra-conservative* way (slide 59), forcing every conditional $\tilde{\rho}_{Z\mid X}(\cdot, x)$ to be standard normal *regardless of $x$*. Taken alone, $L$ is overkill — a latent that ignores the input entirely.

This is exactly the **posterior collapse** failure mode. The condition $L = 0 \Leftrightarrow (\mu_E = 0,\ \sigma_E = 1)$ describes a latent that has *forgotten the input*: the encoder outputs the prior for every $x$, and those latent dimensions become useless. Early in training, when the decoder is still weak and cannot turn latent information into good reconstructions, driving $L \to 0$ is an easy way for the optimizer to lower the objective — so a strong $L$ from step one can "kill" latent dimensions before they ever become useful. An aggressively high learning rate makes this worse (it takes the path of least resistance and zeroes out the KL).

## 3. KL annealing

The standard ELBO weighs $R$ and $L$ equally. **KL annealing** introduces a scalar weight $\beta$ on the latent term,

$$
\text{loss} \;=\; R \;+\; \beta\, L,
$$

and schedules $\beta$ over training instead of fixing it at $1$:

- $\beta = 0$ at the start → the model behaves like a **plain autoencoder**, free to learn good image features and populate the latent space without regularization pressure.
- $\beta$ is then **gradually raised toward $1$**, slowly pulling the posteriors toward $\mathcal{N}(0, I)$ once the latent already carries useful structure — tightening the prior **without** collapsing dimensions prematurely.

Setting $\beta = 1$ recovers the exact ELBO of the deck. Note that $\beta$ is a *training* device: it changes the optimization path, not the quantity we ultimately care about (the true negative ELBO at $\beta = 1$).

## 4. How we use it in this project

### 4.1 In the model

`VAE.forward(x, beta=1.0)` computes the per-dimension KL and returns the $\beta$-weighted objective:

```python
# KL divergence: D_KL( q(z|x) || N(0,I) ), diagonal posterior (USE_DIFF_SIGMA_E=True)
kl = 0.5 * (mu**2 + var - logvar - 1).sum(dim=1).mean()
return recon + beta * kl
```

With `var` $=\sigma_E^2$ and `logvar` $=\ln \sigma_E^2$, the summand $\tfrac{1}{2}(\mu^2 + \sigma^2 - \ln\sigma^2 - 1)$ is precisely the per-dimension form of $L$ in eq. (47). So the code term is the slide's latent distribution error, no more and no less.

### 4.2 The schedule

We use a **step schedule** that starts at $0$ and increases $\beta$ by $0.2$ every $5$ epochs, capped at $1.0$:

```python
KL_BETA_STEP = 0.2
def kl_beta(epoch, anneal_epochs, step=KL_BETA_STEP):
    if anneal_epochs <= 0:
        return 1.0                              # constant β = 1 (no annealing)
    return min(1.0, step * (epoch // anneal_epochs))
```

| epoch | 1–4 | 5–9 | 10–14 | 15–19 | 20–24 | $\ge 25$ |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| $\beta$ | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |

So the first few epochs are a pure-autoencoder warm-up ($\beta = 0$), and the full prior pressure ($\beta = 1$) is reached at epoch 25. `train_and_evaluate` calls `beta = kl_beta(epoch, kl_anneal_epochs)` once per epoch and passes it into `train_one_epoch → VAE.forward`.

### 4.3 Fair comparison

Crucially, **evaluation always uses $\beta = 1$**: `evaluate` calls the model with the default `beta=1.0`, so the test metric is the **true negative ELBO** regardless of where the training $\beta$ currently sits. This means an annealed run and a constant-$\beta$ run are scored on the same objective, and the hyperparameter search can compare them honestly.

### 4.4 In the hyperparameter search

KL annealing is the second stage of our sequential search (`docs/hyperparameter_tuning_rationale.md`). We treat it as an A/B choice between a constant schedule and the annealed one:

```python
# Stage 2 — KL annealing: constant β=1  vs  step +0.2 every 5 epochs (0→1)
run_search(stage='KL_annealing', param='kl_anneal_epochs',
           candidates=[('constant', 0), ('step0.2_per5ep', 5)])
```

`kl_anneal_epochs = 0` means constant $\beta = 1$ (no annealing); any positive value is the step period. The winner (lower test ELBO) is locked into `best_config` and carried into the remaining stages.

## 5. Why this matters here specifically

This VAE is not an end in itself — it is the **encoder/decoder for a downstream Latent Diffusion model**. The diffusion process is built assuming the latent prior is $\mathcal{N}(0, I)$ (slide 62), and it maps most easily over a latent space that is **dense, continuous, and close to standard normal**. That makes the $L$ term — and controlling it well — directly valuable: we want the regularization that $L$ provides, but we cannot afford the posterior collapse it can cause if applied too hard, too early.

KL annealing is the lever that buys both: it lets $R$ build an informative latent first, then ramps $L$ in to shape that latent toward the prior the diffusion stage expects. As the deck notes (slide 63), VAEs tend to do better at latent organization than at direct generation, and they "excel as auxiliary tools for dimensionality reduction, making other generative algorithms more efficient" — which is exactly the role this VAE plays for the diffusion model, and exactly why a well-tuned $\beta$ schedule is worth the search.
