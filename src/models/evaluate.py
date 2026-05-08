from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_single_model(model, model_name, X_cv, y_cv, X_test, y_test, threshold=0.5) -> None:
    print(f"\n{'='*30}")
    print(f" {model_name}")
    print(f"{'='*30}")

    # use probability threshold instead of default predict() for better control over imbalanced data
    y_pred_cv   = (model.predict_proba(X_cv)[:, 1] >= threshold).astype(int)
    y_pred_test = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    # report cv metrics — used to detect bias or variance issues
    print("CV Results:")
    print(f"  Accuracy:  {accuracy_score(y_cv, y_pred_cv):.4f}")
    print(f"  Precision: {precision_score(y_cv, y_pred_cv):.4f}")
    print(f"  Recall:    {recall_score(y_cv, y_pred_cv):.4f}")
    print(f"  F1:        {f1_score(y_cv, y_pred_cv):.4f}")

    print("-" * 15)

    # report test metrics — final performance benchmark
    print("Test Results:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_test):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_test):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred_test):.4f}")

    # plot confusion matrix to understand error distribution
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.show()


def evaluate(lr, rf, X_cv_lr, X_cv, y_cv, X_test_lr, X_test, y_test) -> None:
    # evaluate logistic regression with default threshold
    evaluate_single_model(lr, "Logistic Regression", X_cv_lr, y_cv, X_test_lr, y_test)

    # evaluate random forest with lower threshold to handle class imbalance
    evaluate_single_model(rf, "Random Forest", X_cv, y_cv, X_test, y_test, threshold=0.3)