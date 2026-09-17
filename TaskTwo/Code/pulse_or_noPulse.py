#imprt da librarees
import pandas as pd
#baap re
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, precision_score, recall_score, f1_score

from sklearn.impute import SimpleImputer #to impute
from sklearn.decomposition import PCA
#for da graphs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go

#read rhe csv
dataset = pd.read_csv("TaskTwo/Datasets/pulsar_data_train.csv")
dataset.columns = dataset.columns.str.strip()

# take into two sets for target and features
X = dataset.drop(columns="target_class")
y = dataset["target_class"].astype(int)

# Spliting set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=42)

## Medians learned from training data only so that test doesnt effect training
#medians = X_train.median()
##filling in medians to the null values(instead of deleting them cos omg theres so many of these)(not mean cos mean is one side shifted)
#X_train = X_train.fillna(medians)
#X_test  = X_test.fillna(medians)

#baseline: default params, both kernels (dictionary)
baseline = {}

########################################storing the base values (Myself wohoo :())

for kernel in ["linear", "rbf"]:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),  
        ("svc", SVC(kernel=kernel, class_weight= "balanced", cache_size=500, random_state=42)),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    baseline[kernel] = pipe

    #"Dignostics" ig 
    print(f"\n {kernel.upper()} (default)")
    #counts of true/false positives/negatives.
    print(confusion_matrix(y_test, pred))
    #precision, recall, F1 per class.
    print(classification_report(y_test, pred))
    #  ROC-AUC, computed from the signed distance to the decision boundary rather than hard 0/1 predictions, so it captures ranking quality.
    #what T_T the fah
    print("ROC-AUC :", round(roc_auc_score(y_test, pipe.decision_function(X_test)), 4))
    print("PR-AUC  :", round(average_precision_score(y_test, pipe.decision_function(X_test)), 4))
    print("train acc:", round(pipe.score(X_train, y_train), 4),
            "| test acc:", round(pipe.score(X_test, y_test), 4))
    print("n_support:", pipe["svc"].n_support_)


############################################################## GRIF SEARCH over C and gamma and degree and coeddicient
#using cross validation

#making 'folds' to test on
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#this is the search space
param_grid = [
    {"svc__kernel": ["linear"],
     "svc__C": [0.01, 0.1, 1, 10, 100]},
    {"svc__kernel": ["rbf"],
     "svc__C": [0.1, 1, 10, 100],
     "svc__gamma": ["scale", 0.01, 0.1, 1]}
    #{"svc__kernel": ["poly"],
    #"svc__C": [0.1, 1, 10],
    #"svc__gamma": ["scale", 0.01, 0.1, 1],
    #"svc__degree": [2, 3],
    #"svc__coef0": [0, 1]}
]

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svc", SVC(random_state=42)),
])

###########################################################ACtual SEArch################################################
grid = GridSearchCV(
    pipe, param_grid,
    #selecting by f1(harmonic mean)
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
    verbose=1,
)
grid.fit(X_train, y_train)

print("\nbest params:", grid.best_params_)
print("best CV f1 :", round(grid.best_score_, 4))

best = grid.best_estimator_
pred = best.predict(X_test)

################################################DIAGNOSTICS###############################################

print("\n BEST MODEL ON TEST ")
#making the confusuion matrix
#           predicted 1     predicted 0
# Actual 1
# Actual 0
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print("ROC-AUC :", round(roc_auc_score(y_test, best.decision_function(X_test)), 4))
print("PR-AUC :", round(average_precision_score(y_test, best.decision_function(X_test)), 4))
print("n_support:", best["svc"].n_support_)

#show the results as a table
cv_results = pd.DataFrame(grid.cv_results_)
#degree and coef0 only apply to poly, so linear/rbf rows show NaN there (thats fine, it just means "not used")
#without them every poly combo with the same C/gamma looks identical in the table
cols = ["param_svc__kernel", "param_svc__C", "param_svc__gamma",
        "mean_train_score", "mean_test_score", "rank_test_score"]

summary = (cv_results[cols]
           .sort_values("mean_test_score", ascending=False)
           .reset_index(drop=True))
print(summary.to_string())

print("\nbest per kernel:")
print(summary.loc[summary.groupby("param_svc__kernel")["mean_test_score"].idxmax()])


