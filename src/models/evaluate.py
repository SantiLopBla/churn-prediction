from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


def find_best_threshold(model, X, y):
    # get the predicted probability of churn for each row
    y_probs = model.predict_proba(X)[:, 1]

    # roc_curve returns every possible threshold the model can use
    # we loop through all of them and keep the one with the highest F1
    _, _, thresholds = roc_curve(y, y_probs)

    best_threshold = 0.5
    best_f1 = 0

    # only test thresholds in a reasonable range — extreme values like 0.01 or 0.99
    # can produce artificially high F1 on CV but fail completely on new data
    for t in thresholds[(thresholds >= 0.3) & (thresholds <= 0.7)]:
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

    # convert probabilities to binary predictions using the chosen threshold
    # predict_proba[:,1] gives the probability of churn for each row
    # any row with probability >= threshold is predicted as churn
    y_pred_cv   = (model.predict_proba(X_cv)[:, 1]   >= threshold).astype(int)
    y_pred_test = (model.predict_proba(X_test)[:, 1]  >= threshold).astype(int)

    # cv metrics — used to check if the model is underfitting or overfitting
    print("CV Results:")
    print(f"  Accuracy:  {accuracy_score(y_cv, y_pred_cv):.4f}")
    print(f"  Precision: {precision_score(y_cv, y_pred_cv):.4f}")
    print(f"  Recall:    {recall_score(y_cv, y_pred_cv):.4f}")
    print(f"  F1:        {f1_score(y_cv, y_pred_cv):.4f}")

    print("-" * 15)

    # test metrics — the final honest score, only evaluated once
    print("Test Results:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_test):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_test):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred_test):.4f}")

    # AUC-ROC measures how well the model separates churners from non-churners
    # across all possible thresholds — 1.0 is perfect, 0.5 is random guessing
    # this score does not depend on the threshold chosen above
    y_probs_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs_test)
    print(f"  AUC-ROC:   {auc:.4f}")

    # confusion matrix — shows exactly what the model got right and wrong
    # top-left:     true negatives  — correctly predicted no churn
    # top-right:    false positives — predicted churn but customer stayed
    # bottom-left:  false negatives — predicted no churn but customer left (most costly)
    # bottom-right: true positives  — correctly predicted churn
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.show()


def evaluate(lr, rf, xgb, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test) -> None:

    # logistic regression uses scaled data — threshold stays at default 0.5
    evaluate_single_model(lr, "Logistic Regression", X_cv_lr, y_cv, X_test_lr, y_test)

    # find the best threshold for RF, then evaluate immediately after
    # keeping these two calls together makes the output clean and readable
    rf_threshold = find_best_threshold(rf, X_cv, y_cv)
    evaluate_single_model(rf,  "Random Forest", X_cv, y_cv, X_test, y_test, threshold=rf_threshold)

    # same pattern for XGBoost — threshold search followed by evaluation
    xgb_threshold = find_best_threshold(xgb, X_cv, y_cv)
    evaluate_single_model(xgb, "XGBoost", X_cv, y_cv, X_test, y_test, threshold=xgb_threshold)