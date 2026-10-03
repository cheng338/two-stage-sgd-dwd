"""Tiny explicit CV example; not a MNIST study or a performance comparison."""
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from two_stage_sgd_dwd import TwoStageDWDClassifier, make_search


def main():
    X, y = make_moons(n_samples=120, noise=0.18, random_state=17)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=17)
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        # A small synthetic example needs more than 5% to retain both classes.
        ("dwd", TwoStageDWDClassifier(retention_fraction=0.5, max_epochs=8,
                                       batch_size=1, random_state=17)),
    ])
    # Demonstration only: real studies must justify ranges and SGD budgets.
    grid = [
        {"dwd__kernel": ["linear"], "dwd__lambd": [5e-4, 5e-3]},
        {"dwd__kernel": ["rbf"], "dwd__lambd": [5e-4, 5e-3],
         "dwd__gamma": [0.1, 1.0]},
    ]
    with threadpool_limits(1):
        search = make_search(pipeline, grid, cv=3, random_state=17)
        search.fit(X_train, y_train)
        print("Selected parameters:", search.best_params_)
        print("Tuning CV accuracy (not final performance):", search.best_score_)
        print("Illustrative held-out accuracy:", search.score(X_test, y_test))
    return search


if __name__ == "__main__":
    main()
