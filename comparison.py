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
from binary_classification import load_data


def train_isotonic():
    # Load dataset
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # Convert torch tensors to numpy
    X_train = X_train.numpy()
    X_test = X_test.numpy()
    y_train = y_train.numpy()
    y_test = y_test.numpy()

    # IsotonicRegression takes 1d input (so using first feature only)
    X_train_1d = X_train[:, 0]
    X_test_1d = X_test[:, 0]

    # model
    iso_model = IsotonicRegression(out_of_bounds="clip")
    iso_model.fit(X_train_1d, y_train)

    # Predict
    y_train_pred = iso_model.predict(X_train_1d)
    y_test_pred = iso_model.predict(X_test_1d)

    # Convert regression output to binary class
    y_train_pred_class = (y_train_pred >= 0.5).astype(int)
    y_test_pred_class = (y_test_pred >= 0.5).astype(int)

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred_class)
    test_acc = accuracy_score(y_test, y_test_pred_class)

    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:     {test_acc:.3f}")

    return iso_model

model = train_isotonic()

"""
The isotonic regression model performed much poorer than the logistic 
regression model (63% training and test accuracy for isotonic compared
to 99% for logistic). I think it probably has to do with the number of
features being reduced, but nonetheless I still think it performed much
better than expected (still more than half accurate with over 90% less features).
"""