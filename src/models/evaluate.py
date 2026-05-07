from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

def evaluate(model, X_cv, y_cv, X_test, y_test) -> None:
    # generate predictions on the cross validation set
    y_pred_cv = model.predict(X_cv)

    # calculate classification metrics for cv set
    # precision and recall matter more than accuracy on imbalanced churn data
    acc_cv  = accuracy_score(y_cv, y_pred_cv)
    prec_cv = precision_score(y_cv, y_pred_cv)
    rec_cv  = recall_score(y_cv, y_pred_cv)
    f1_cv   = f1_score(y_cv, y_pred_cv)

    # generate predictions on the test set
    y_pred_test = model.predict(X_test)

    # calculate classification metrics for test set
    acc_test  = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test)
    rec_test  = recall_score(y_test, y_pred_test)
    f1_test   = f1_score(y_test, y_pred_test)

    # report cv metrics — used to detect bias or variance issues
    print("CV Results:")
    print(f"  Accuracy:  {acc_cv:.4f}")
    print(f"  Precision: {prec_cv:.4f}")
    print(f"  Recall:    {rec_cv:.4f}")
    print(f"  F1:        {f1_cv:.4f}")

    print("-" * 15)

    # report test metrics — final performance benchmark
    print("\nTest Results:")
    print(f"  Accuracy:  {acc_test:.4f}")
    print(f"  Precision: {prec_test:.4f}")
    print(f"  Recall:    {rec_test:.4f}")
    print(f"  F1:        {f1_test:.4f}")