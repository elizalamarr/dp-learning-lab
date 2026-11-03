"""Start with a non-private classifier on synthetic data."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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


def main():
    x_train, x_test, y_train, y_test = dataset()
    baseline = LogisticRegression(C=1.0, max_iter=1000).fit(x_train, y_train)
    print(f"Training examples: {len(y_train)}; test examples: {len(y_test)}")
    print(f"Non-private accuracy: {baseline.score(x_test, y_test):.3f}")


if __name__ == "__main__":
    main()
