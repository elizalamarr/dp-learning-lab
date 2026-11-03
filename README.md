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
```

## Build sequence

1. Set up the project and dependencies (current step).
2. Generate synthetic data and train a non-private baseline.
3. Compare private models across privacy budgets.
4. Test feature clipping and privacy accounting.
5. Save results and explain the accuracy tradeoff.

This experiment evaluates utility; an accuracy curve cannot prove differential
privacy. All examples will use synthetic, non-sensitive data.
