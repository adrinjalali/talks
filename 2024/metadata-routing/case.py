# %%
import numpy as np
from sklearn.datasets import fetch_openml

# %%
credit_card = fetch_openml(data_id=1597, as_frame=True, parser="pandas")
credit_card.frame.info()

# %%
columns_to_drop = ["Class"]
data = credit_card.frame.drop(columns=columns_to_drop)
target = credit_card.frame["Class"].astype(int)

# %%
target.value_counts(normalize=True)

# %%
target.value_counts()

# %%
import matplotlib.pyplot as plt

fraud = target == 1
amount_fraud = data["Amount"][fraud]
_, ax = plt.subplots()
ax.hist(amount_fraud, bins=100)
ax.set_title("Amount of fraud transaction")
_ = ax.set_xlabel("Amount (€)")
plt.show()


# %%
def business_metric(y_true, y_pred, amount):
    mask_true_positive = (y_true == 1) & (y_pred == 1)
    mask_true_negative = (y_true == 0) & (y_pred == 0)
    mask_false_positive = (y_true == 0) & (y_pred == 1)
    mask_false_negative = (y_true == 1) & (y_pred == 0)
    legitimate_refuse = mask_false_positive.sum() * -5
    fraudulent_refuse = (mask_true_positive.sum() * 50) + amount[
        mask_true_positive
    ].sum()
    fraudulent_accept = -amount[mask_false_negative].sum()
    legitimate_accept = (amount[mask_true_negative] * 0.02).sum()
    return fraudulent_refuse + fraudulent_accept + legitimate_refuse + legitimate_accept


# %%
import sklearn
from sklearn.metrics import make_scorer

sklearn.set_config(enable_metadata_routing=True)
business_scorer = make_scorer(business_metric).set_score_request(amount=True)

# %%
amount = credit_card.frame["Amount"].to_numpy()

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test, amount_train, amount_test = (
    train_test_split(
        data, target, amount, stratify=target, test_size=0.5, random_state=42
    )
)

# %%
from sklearn.dummy import DummyClassifier

easy_going_classifier = DummyClassifier(strategy="constant", constant=0)
easy_going_classifier.fit(data_train, target_train)
benefit_cost_tolerant = business_scorer(
    easy_going_classifier, data_test, target_test, amount=amount_test
)
print(f"Benefit/cost of our easy-going classifier: {benefit_cost_tolerant:,.2f}€")

# %%
intolerant_classifier = DummyClassifier(strategy="constant", constant=1)
intolerant_classifier.fit(data_train, target_train)
benefit_cost_intolerant = business_scorer(
    intolerant_classifier, data_test, target_test, amount=amount_test
)
print(f"Benefit/cost of our intolerant classifier: {benefit_cost_intolerant:,.2f}€")

# %%
from sklearn.metrics import get_scorer

balanced_accuracy_scorer = get_scorer("balanced_accuracy")
tolerant_balanced_arruracy = balanced_accuracy_scorer(
    easy_going_classifier, data_test, target_test
)
print(
    "Balanced accuracy of our easy-going classifier: "
    f"{tolerant_balanced_arruracy:.3f}"
)
intolerant_balanced_accuracy = balanced_accuracy_scorer(
    intolerant_classifier, data_test, target_test
)
print(
    "Balanced accuracy of our intolerant classifier: "
    f"{intolerant_balanced_accuracy:.3f}"
)

# %%
accuracy_scorer = get_scorer("accuracy")
tolerant_arruracy = accuracy_scorer(
    easy_going_classifier, data_test, target_test
)
print(
    "Accuracy of our easy-going classifier: "
    f"{tolerant_arruracy:.3f}"
)
intolerant_accuracy = accuracy_scorer(
    intolerant_classifier, data_test, target_test
)
print(
    "Accuracy of our intolerant classifier: "
    f"{intolerant_accuracy:.3f}"
)


# %%
import pandas as pd
scores = pd.DataFrame(
    {"Name": [], "Benefit/Cost": [], "Balanced Accuracy": [], "Accuracy": []}
)

def add_score(name, business_score, balanced_accuracy_score, accuracy_score):
    return pd.concat(
        [scores, pd.DataFrame(
            {"Name": [name],
             "Benefit/Cost": [business_score],
             "Balanced Accuracy": [balanced_accuracy_score],
             "Accuracy": [accuracy_score]})]
    ).reset_index(drop=True)

scores = add_score(
    "Tolerant",
    benefit_cost_tolerant,
    tolerant_balanced_arruracy,
    tolerant_arruracy
)
scores = add_score(
    "Intolerant",
    benefit_cost_intolerant,
    intolerant_balanced_accuracy,
    intolerant_accuracy
)
scores

# %%
def handle_scores(model, name):
    business_score = business_scorer(
        model, data_test, target_test, amount=amount_test
    )
    balanced_accuracy_score = balanced_accuracy_scorer(model, data_test, target_test)
    accuracy_score = accuracy_scorer(model, data_test, target_test)
    global scores
    scores = add_score(name, business_score, balanced_accuracy_score, accuracy_score)
    with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 199
    ):
        print(scores)


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logistic_regression = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {"logisticregression__C": np.logspace(-6, 6, 13)}
model = GridSearchCV(
    logistic_regression, param_grid, scoring=business_scorer, n_jobs=-1
).fit(
    data_train, target_train, amount=amount_train
)

handle_scores(model, "Searched logistic regression")
model.best_estimator_

# %%
from sklearn.model_selection import TunedThresholdClassifierCV

tuned_model = TunedThresholdClassifierCV(
    estimator=model.best_estimator_,
    scoring=business_scorer,
    thresholds=40,
    n_jobs=-1,
).fit(data_train, target_train, amount=amount_train)

handle_scores(tuned_model, "Tuned searched logistic regression")
print(f"Best threshold: {tuned_model.best_threshold_}")
# %%
tuned_model = TunedThresholdClassifierCV(
    estimator=logistic_regression,
    scoring=business_scorer,
    thresholds=40,
    n_jobs=1,
)
tuned_model._response_method = "predict_proba"

param_grid = {"estimator__logisticregression__C": np.logspace(-6, 6, 13)}
model = GridSearchCV(tuned_model, param_grid, scoring=business_scorer, n_jobs=-1).fit(
    data_train, target_train, amount=amount_train
)

handle_scores(model, "Searched tuned logistic regression")
model.best_estimator_

# %%
print(f"Best threshold: {model.best_estimator_.best_threshold_}")

