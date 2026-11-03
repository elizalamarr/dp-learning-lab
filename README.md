# DP Learning Lab

How does differential privacy affect a classifier's accuracy?

This small learning project compares ordinary logistic regression with
private logistic regression on synthetic binary classification data.
We will build it in short, runnable steps and keep the code concise.

## Setup

Use Python 3.11–3.13. The dependency versions were tested on Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py
```

## Baseline

The script generates 1,200 synthetic examples with six features and two labels.
It trains ordinary logistic regression on 840 examples and evaluates accuracy
on the other 360. Fixed random seeds make the data and split reproducible.
The current baseline scores about 78.6% accuracy.

Each feature vector is clipped to L2 norm at most one: long vectors shrink,
while shorter vectors stay unchanged. This public bound prepares a fair
comparison with private models. Clipping alone does not make training
private; this baseline provides the reference accuracy without privacy noise.

## Private models

The script also trains private logistic regression at epsilon 0.1, 0.3, 1, 3,
and 10. Smaller epsilon means a stronger privacy bound. Each setting is run ten
times on the same data, with fresh privacy noise. The printed mean accuracy and
sample standard deviation show how utility varies across those noise draws;
the standard deviation is not a confidence interval.

The private implementation comes from [IBM diffprivlib](https://diffprivlib.readthedocs.io/en/latest/modules/models.html#logistic-regression)
and uses objective perturbation. We supply the public norm bound of one rather
than estimating it from data. Each fit uses its selected epsilon and delta zero
under the library's assumptions. A shared accountant tracks repeated training:
the whole sweep composes to epsilon **144**, delta **0**, rather than the epsilon
of any individual fit. The ordinary baseline is not covered by this accountant.

Only the synthetic data and split are seeded; privacy noise is not, so scores
change between runs and need not increase monotonically with epsilon. This is
a synthetic-data learning experiment: real private labels, test-set metrics,
and model selection would require additional privacy analysis.

## Tests

```bash
python -m unittest -v
```

Three tests check that clipping handles zero and signed vectors without changing
the input, that every repeated private fit contributes to the total budget, and
that separate evaluations do not accidentally share accounting state. They also
check run coverage and valid accuracy values without depending on random scores.
These checks catch implementation mistakes; passing them does not prove privacy.

## Build sequence

1. Set up the project and dependencies (done).
2. Generate synthetic data and train a non-private baseline (done).
3. Compare private models across privacy budgets (done).
4. Test feature clipping and privacy accounting (current step).
5. Save results and explain the accuracy tradeoff.

This experiment evaluates utility; an accuracy curve cannot prove differential
privacy. All examples will use synthetic, non-sensitive data.
