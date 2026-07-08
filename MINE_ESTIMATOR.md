# MINE Estimator — How the Reward↔Action MI Is Estimated

This documents the neural MI estimator in `sd35_reward_mi_diagnostic.py`. It answers
a single question: **how much does a search "control" channel `C` (prompt variant,
CFG, or per-step action) explain about the reward `R`, holding the prompt `P` fixed?**

## 1. What is estimated: conditional MI `I(R; C | P)`

We estimate the **within-prompt conditional mutual information**

$$
I(R; C \mid P) \;=\; \mathbb{E}_{P}\Big[\, \mathrm{KL}\big(\,p(R,C\mid P)\;\|\;p(R\mid P)\,p(C\mid P)\,\big)\Big]
$$

- `R` — scalar reward (composite / IR / HPSv3 …).
- `C` — a discrete control channel: variant id, cfg id, or action id.
- `P` — prompt id (the confounder we condition on).

Conditioning on `P` is the whole point: raw `I(R;C)` is dominated by prompt-level
reward differences. We want the reward variation **attributable to the action**,
with the prompt held fixed. This is achieved entirely through how negatives are
sampled (§4).

## 2. The critic network (`MineCritic`)

A single scalar critic `T_θ(r, p, c)`:

```
reward  r ──────────────► [ scalar, 1-d ]
prompt  p ──► Embedding(n_prompts, 32) ─────────┐
control c ──► Embedding(|C|, 32) ───────────────┤─► concat ─► MLP ─► T ∈ ℝ
                                                 │   Linear(in,256)→SiLU
              (one embedding per control channel)│   Linear(256,256)→SiLU
                                                 │   Linear(256,1)
```

- Input dim `= 1 + 32·(1 + #controls)` (reward is the raw scalar; prompt and each
  control are embedded, default `embedding_dim=32`, `hidden_dim=256`).
- Output is a single logit `T_θ(r,p,c)` — higher ⇒ this `(r,c)` pair looks like a
  genuine within-prompt co-occurrence rather than a shuffled one.

## 3. The variational lower bound (DV / SMILE / InfoNCE)

The estimate uses a Donsker–Varadhan (DV) style lower bound:

$$
I(R;C\mid P) \;\ge\; \underbrace{\mathbb{E}_{\text{pos}}\big[T_\theta\big]}_{\text{joint }p(r,c\mid p)} \;-\; \log \underbrace{\mathbb{E}_{\text{neg}}\big[e^{T_\theta}\big]}_{\text{product }p(r\mid p)p(c\mid p)}
$$

In code (`_critic_bound`, `_log_mean_exp`):

```
bound = t_pos.mean() − logmeanexp(t_neg)
```

Three interchangeable estimators (`--mine_estimator`, default **`smile`**):

| estimator | bound | key detail |
|---|---|---|
| `dv` | raw Donsker–Varadhan | `logmeanexp(t_neg)` unclipped — low bias, high variance |
| **`smile`** | DV with clipped critic | clip `T` to `[−τ, τ]` (τ=`mine_smile_tau`, default 5) **inside** the log-mean-exp — tames the classic MINE/DV blow-up from a single large negative logit; trades a little bias for much lower variance |
| `infonce` | block softmax (NCE) | K same-prompt samples per block, diagonal = positive; lowest variance, bounded by `log K` |

SMILE is the default: it is the DV bound with `t_neg` clamped to `[−τ,τ]` before the
`logsumexp`, which is the single most important stabilizer for near-zero channels.

## 4. Negatives = within-prompt permutation (what makes it *conditional*)

Positives are the real joint rows `(r_i, p_i, c_i)`. Negatives are built by
**permuting the control within each prompt group** (`_permute_within_prompt_ids`,
`_permute_controls_in_batch`):

```
negative_i = (r_i, p_i, c_{σ(i)})   where σ shuffles rows only among the SAME prompt
```

This draws from `p(R|P)·p(C|P)` — the product of the within-prompt conditionals —
so the bound estimates `I(R;C|P)`, not the prompt-confounded `I(R;C)`. Shuffling
across prompts would instead measure prompt effects; shuffling *within* prompt is
what isolates the action's contribution.

## 5. Training a single restart (`_train_one_restart`)

1. **3-way split** train / val / test (`--mine_val_frac`), split by row.
2. **Optimizer**: AdamW, `lr=1e-4`, gradient clip `1.0`, batch `1024`, up to
   `mine_steps=10000` steps.
3. **Per step**: sample a minibatch → positives are the real rows, negatives are a
   within-prompt permutation of the controls in that batch → `loss = −bound` →
   backprop.
4. **EMA-smoothed early stopping** (§6): every `mine_eval_every=250` steps, measure
   val MI, update an EMA `ema ← 0.7·ema + 0.3·val_mi`, and keep the **best-EMA**
   checkpoint (optional patience via `--mine_early_stop_patience`).
5. **Restarts**: `--mine_restarts` (default 3) independent inits/seeds; results are
   aggregated across restarts to reduce init variance.

## 6. Why EMA early stopping (bias control)

Selecting the single **max** val-MI checkpoint maximizes over noisy estimates → an
upward bias that is worst exactly where the true MI is near zero. Instead we track an
EMA of val MI and keep the best *smoothed* checkpoint, so a lucky spike doesn't get
locked in. (`best_ema`, `ema_val` in `_train_one_restart`.)

## 7. The final estimate (`_estimate_mi_and_auroc`)

The reported number is a **full-set single-pass** DV/SMILE evaluation on the held-out
**test** split (not the running training bound):

- Accumulate `t_pos` over **every** row, and take **one** `log-mean-exp` over **all**
  negatives from a single within-prompt permutation.
- This avoids averaging per-minibatch DV bounds, which is Jensen-biased *upward* and
  high-variance:

$$
\widehat{I} \;=\; \frac{1}{N}\sum_i T_\theta(\text{pos}_i)\;-\;\log\!\Big(\tfrac{1}{N}\sum_i e^{\,\mathrm{clip}_\tau T_\theta(\text{neg}_i)}\Big)
$$

- A companion **AUROC** (positives vs. negatives by critic score) is returned as a
  critic-quality diagnostic — AUROC ≈ 0.5 ⇒ the critic can't separate joint from
  shuffled ⇒ MI ≈ 0.

## 8. Debiasing and reporting

- **Permutation null**: the same pipeline is run on a label-permuted dataset to get
  `mi_null` (should sit at ~0). The corrected statistic is
  `mi_corrected = mi_real − mi_null` (`--mine_report_raw_corrected` drops the
  `max(0,·)` floor / `H(C)` clamp so the null genuinely centers at zero).
- **Prompt-bootstrap CI**: resample prompts to get a confidence interval.
- **Explained-variance readout** (`_mi_explained_var`): MI (nats) is mapped to a
  variance fraction via the Gaussian `I ↔ ρ²` identity

$$
\rho^2 \;=\; 1 - e^{-2\widehat{I}} \;\in [0,1)
$$

  i.e. the fraction of (CRN-residualized) reward variance the channel explains —
  cardinality-free (unlike `MI/H(C)`), `≈ 2·I` for small `I`, and directly comparable
  to the variance / η² decomposition.

## 9. Non-neural cross-checks

The same driver also computes distribution-free estimators as sanity checks against
MINE, each with a prompt-bootstrap CI (`--mine_ross_bootstrap`):

| estimator | idea |
|---|---|
| `ross` | k-NN (Ross) MI for a continuous reward vs. discrete label (`--mine_ross_k`) |
| `mm` | Miller–Madow bias-corrected plug-in on equal-frequency reward bins (`--mine_mm_bins`) |
| `omega2` | ω² — bias-debiased fraction of reward variance explained by the discrete action (distribution twin of the Gaussian ρ² map) |
| `dcor` | distance correlation |

If MINE and these agree (and the null sits at 0), the MI is trustworthy.

## 10. Key hyperparameters

| flag | default | meaning |
|---|---|---|
| `--mine_estimator` | `smile` | bound: `smile` / `dv` / `infonce` |
| `--mine_smile_tau` | `5.0` | SMILE critic clip `[−τ,τ]` |
| `--mine_hidden_dim` | `256` | critic MLP width |
| `--mine_embedding_dim` | `32` | prompt/control embedding size |
| `--mine_batch_size` | `1024` | minibatch |
| `--mine_lr` | `1e-4` | AdamW LR |
| `--mine_steps` | `10000` | max training steps per restart |
| `--mine_eval_every` | `250` | val-MI eval cadence |
| `--mine_grad_clip` | `1.0` | grad-norm clip |
| `--mine_val_frac` | `0.2` | held-out fraction |
| `--mine_restarts` | `3` | independent inits |
| `--mine_early_stop_patience` | `0` | 0 = off; else EMA-plateau patience |

## Summary (one line)

A prompt/control-embedding MLP critic is trained by a SMILE-clipped Donsker–Varadhan
bound with **within-prompt-permuted negatives** to estimate the conditional MI
`I(R;C|P)`; the reported value is a full-set single-pass test estimate, EMA-checkpoint
selected, null-subtracted and prompt-bootstrapped, and reported both in nats and as
an explained-variance fraction `ρ² = 1 − e^{−2I}`.
