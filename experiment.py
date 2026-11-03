"""Compare private and non-private classifiers on synthetic data."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.models import LogisticRegression as PrivateLogisticRegression
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

EPSILONS = (0.1, 0.3, 1.0, 3.0, 10.0)
REPEATS = 10


def clip_rows(x):
    """Limit each example's L2 norm to the public constant one."""
    return x / np.maximum(1.0, np.linalg.norm(x, axis=1, keepdims=True))


def dataset():
    x, y = make_classification(
        n_samples=1200, n_features=6, n_informative=4,
        n_redundant=0, class_sep=1.2, random_state=42,
    )
    return train_test_split(clip_rows(x), y, test_size=0.3,
                            stratify=y, random_state=42)


def evaluate(epsilons=EPSILONS, repeats=REPEATS):
    x_train, x_test, y_train, y_test = dataset()
    baseline = LogisticRegression(C=1.0, max_iter=1000).fit(x_train, y_train)
    accountant = BudgetAccountant()
    rows = []
    for epsilon in epsilons:
        for repeat in range(repeats):
            model = PrivateLogisticRegression(
                epsilon=epsilon, data_norm=1.0, C=1.0, max_iter=1000,
                accountant=accountant,
            ).fit(x_train, y_train)
            rows.append({"epsilon": epsilon, "repeat": repeat,
                         "accuracy": model.score(x_test, y_test)})
    return baseline.score(x_test, y_test), rows, accountant.total()


def main():
    baseline, rows, budget = evaluate()
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    with (output / "runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    means, stds = [], []
    print(f"Non-private accuracy: {baseline:.3f}")
    for epsilon in EPSILONS:
        scores = [row["accuracy"] for row in rows if row["epsilon"] == epsilon]
        means.append(np.mean(scores))
        stds.append(np.std(scores, ddof=1))
        print(f"epsilon={epsilon:4g}: {means[-1]:.3f} ± {stds[-1]:.3f} (sample SD)")
    print(f"Composed budget across all private fits: {budget}")
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.errorbar(EPSILONS, means, yerr=stds, marker="o", capsize=4,
                label="Private: mean ± 1 sample SD")
    ax.axhline(baseline, color="gray", linestyle="--", label="Non-private")
    ax.set(xscale="log", xlabel="Epsilon (smaller = stronger privacy)",
           ylabel="Test accuracy", ylim=(0, 1), title="Privacy versus accuracy")
    ax.legend(fontsize=8)
    fig.savefig(output / "accuracy.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
