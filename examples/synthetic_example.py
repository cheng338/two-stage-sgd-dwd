"""Small usage example; its output is not a dissertation benchmark."""
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from two_stage_sgd_dwd import TwoStageDWDClassifier


def main():
    X, y = make_moons(n_samples=600, noise=0.15, random_state=17)
    Xfit, Xheld, yfit, yheld = train_test_split(X, y, test_size=0.4,
                                             stratify=y, random_state=19)
    Xval, Xtest, yval, ytest = train_test_split(Xheld, yheld, test_size=0.5,
                                              stratify=yheld, random_state=23)
    with threadpool_limits(1):
        model = TwoStageDWDClassifier(lambd=5e-4, gamma=1.0,
                                      random_state=29, max_epochs=40)
        model.fit(Xfit, yfit, validation_data=(Xval, yval))
        print({"example": "synthetic_moons_not_mnist",
               "retained": model.n_retained_,
               "class_counts": model.selected_class_counts_.tolist(),
               "accuracy": model.score(Xtest, ytest),
               "model_status": model.model_status_, "stop_reason": model.stop_reason_})


if __name__ == "__main__":
    main()
