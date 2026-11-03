# DP Learning Lab

How does differential privacy affect a classifier's accuracy?

This small learning project will compare ordinary logistic regression with
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
comparison with private models later. Clipping alone does not make training
private; this baseline provides the reference accuracy without privacy noise.

## Build sequence

1. Set up the project and dependencies (done).
2. Generate synthetic data and train a non-private baseline (current step).
3. Compare private models across privacy budgets.
4. Test feature clipping and privacy accounting.
5. Save results and explain the accuracy tradeoff.

This experiment evaluates utility; an accuracy curve cannot prove differential
privacy. All examples will use synthetic, non-sensitive data.
