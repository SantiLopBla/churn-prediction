from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def find_best_threshold(model, X, y):
    # get predicted probabilities for the positive class (churn)
    y_probs = model.predict_proba(X)[:, 1]

    # roc_curve returns all possible thresholds and their FPR/TPR pairs
    _, _, thresholds = roc_curve(y, y_probs)

    best_threshold = 0.5
    best_f1 = 0

    # try every threshold and keep the one with the highest F1
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        score = f1_score(y, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = t

    print(f"  Best threshold: {best_threshold:.3f} — F1: {best_f1:.4f}")
    return best_threshold

def evaluate_single_model(model, model_name, X_cv, y_cv, X_test, y_test, threshold=0.5) -> None:
    print(f"\n{'='*30}")
    print(f" {model_name}")
    print(f"{'='*30}")

    # convert probabilities to binary predictions using the given threshold
    # predict_proba returns [prob_class_0, prob_class_1] — [:, 1] takes the churn probability
    # default predict() uses 0.5 — lowering it catches more churners at the cost of more false positives
    y_pred_cv   = (model.predict_proba(X_cv)[:, 1] >= threshold).astype(int)
    y_pred_test = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    # cv metrics — used to detect bias (underfitting) or variance (overfitting)
    print("CV Results:")
    print(f"  Accuracy:  {accuracy_score(y_cv, y_pred_cv):.4f}")
    print(f"  Precision: {precision_score(y_cv, y_pred_cv):.4f}")
    print(f"  Recall:    {recall_score(y_cv, y_pred_cv):.4f}")
    print(f"  F1:        {f1_score(y_cv, y_pred_cv):.4f}")

    print("-" * 15)

    # test metrics — final benchmark, should only be evaluated once
    print("Test Results:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_test):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_test):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred_test):.4f}")

    # confusion matrix shows the distribution of correct and incorrect predictions
    # top-left: true negatives (correctly predicted no churn)
    # top-right: false positives (predicted churn but stayed)
    # bottom-left: false negatives (predicted no churn but left) — most costly error
    # bottom-right: true positives (correctly predicted churn)
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.show()


def evaluate(lr, rf, xgb, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test) -> None:
    # logistic regression uses scaled data (X_cv_lr, X_test_lr)
    evaluate_single_model(lr, "Logistic Regression", X_cv_lr, y_cv, X_test_lr, y_test)

    # it calculates the best threshold for Random Forest and XGBoost
    rf_threshold  = find_best_threshold(rf,  X_cv, y_cv)
    xgb_threshold = find_best_threshold(xgb, X_cv, y_cv)

    # random forest and xgboost use unscaled data
    evaluate_single_model(rf, "Random Forest", X_cv, y_cv, X_test, y_test, threshold=rf_threshold)
    evaluate_single_model(xgb, "XGBoost", X_cv, y_cv, X_test, y_test, threshold=xgb_threshold)