"""
The Isotonic Regression model is a type of continuous step-function where a free-form flat line
is fitted to minimize error between the observed points. Specifically it minimizes the sum of 
squared errors between predicted and observed points. Also, the model ensures that the slope of 
inputs and outputs are never in opposite sign (one can't be increasing while the other decreases),
i.e. monotonic. I chose this model just because I only looked into two models and this one seemed
more interesting and just a bit easier to understand. One of the downsides of this model is it only 
takes one feature though.

"""



import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer


def main():
    # Load breast cancer dataset from sklearn
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Use 1 feature
    X_1d = X[:, 0]

    # Simple train/test split
    split = int(0.8 * len(X_1d))
    X_train, X_test = X_1d[:split], X_1d[split:]
    y_train, y_test = y[:split], y[split:]

    # Create isotonic regression model
    model = IsotonicRegression(out_of_bounds="clip")

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Convert regression output to binary labels
    y_train_pred_class = (y_train_pred >= 0.5).astype(int)
    y_test_pred_class = (y_test_pred >= 0.5).astype(int)

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred_class)
    test_acc = accuracy_score(y_test, y_test_pred_class)

    print(f"Isotonic Regression Results")
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:     {test_acc:.3f}")


if __name__ == "__main__":
    main()

"""
The isotonic regression model performed much poorer than the logistic 
regression model (around 63% training and test accuracy for isotonic compared
to 99% for logistic). I think it probably has to do with the number of
features being reduced, but nonetheless I still think it performed much
better than expected (still more than half accurate with over 90% less features).
"""