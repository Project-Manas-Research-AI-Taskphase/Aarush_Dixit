#imprt da librarees
import pandas as pd
#baap re
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
#for da graphs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


#read rhe csv
dataset = pd.read_csv("TaskTwo/Datasets/gene_expression.csv")
dataset.columns = dataset.columns.str.strip()

print(dataset.head())
dataset.info()
print(dataset.describe())
print(dataset.isnull().sum())

# take into two sets for target and features
X = dataset.drop(columns="Cancer Present")
y = dataset["Cancer Present"].astype(int)
print(y.value_counts(normalize=True))

# Spliting set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=42)

#baseline: default params, both kernels (dictionary)
baseline = {}

#storing
for kernel in ["linear", "rbf"]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, random_state=42)),
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
    print("train acc:", round(pipe.score(X_train, y_train), 4),
          "| test acc:", round(pipe.score(X_test, y_test), 4))
    print("n_support:", pipe["svc"].n_support_)




#tuned: grid search over C and gamma
#using cross validation
#making 'folds' to test on
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#this is the search space
param_grid = [
    {"svc__kernel": ["linear"],
     "svc__C": [0.01, 0.1, 1, 10, 100]},
    {"svc__kernel": ["rbf"],
     "svc__C": [0.1, 1, 10, 100],
     "svc__gamma": ["scale", 0.01, 0.1, 1]},
]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(random_state=42)),
])

#actual search
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

#diagnostics again....
print("\n BEST MODEL on test ")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print("ROC-AUC :", round(roc_auc_score(y_test, best.decision_function(X_test)), 4))
print("n_support:", best["svc"].n_support_)

#show the results as a table
cv_results = pd.DataFrame(grid.cv_results_)
cols = ["param_svc__kernel", "param_svc__C", "param_svc__gamma",
        "mean_train_score", "mean_test_score", "rank_test_score"]

summary = (cv_results[cols]
           .sort_values("mean_test_score", ascending=False)
           .reset_index(drop=True))
print(summary.to_string())

print("\nbest per kernel:")
print(summary.loc[summary.groupby("param_svc__kernel")["mean_test_score"].idxmax()])



#the entire graph code::::::::::::::: (Yeah I did not make this myself wth is this T_T)

f1, f2 = X.columns[0], X.columns[1]
C0, C1 = "#3b6ea5", "#c0392b"

pad = 0.5
gx = np.linspace(X_train[f1].min() - pad, X_train[f1].max() + pad, 300)
gy = np.linspace(X_train[f2].min() - pad, X_train[f2].max() + pad, 300)
xx, yy = np.meshgrid(gx, gy)
grid = pd.DataFrame({f1: xx.ravel(), f2: yy.ravel()})

models = {
    "Linear — default\n(C=1)":            SVC(kernel="linear", random_state=42),
    "RBF — default\n(C=1, gamma=scale)":  SVC(kernel="rbf", random_state=42),
    "Linear — tuned\n(C=0.1)":            SVC(kernel="linear", C=0.1, random_state=42),
    "RBF — tuned\n(C=100, gamma=scale)":  SVC(kernel="rbf", C=100, gamma="scale", random_state=42),
}

def scatter(ax):
    for cls, c in [(0, C0), (1, C1)]:
        m = y_train == cls
        ax.scatter(X_train.loc[m, f1], X_train.loc[m, f2],
                   c=c, s=7, alpha=0.45, linewidths=0)

fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5))
axes = axes.ravel()

scatter(axes[0])
axes[0].set_title("Raw training data", fontsize=12, fontweight="bold")

for ax, (title, clf) in zip(axes[1:5], models.items()):
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", clf)]).fit(X_train, y_train)
    Z = pipe.decision_function(grid).reshape(xx.shape)
    sv = X_train.iloc[pipe["svc"].support_]

    scatter(ax)
    ax.scatter(sv[f1], sv[f2], s=34, facecolors="none",
               edgecolors="#222", linewidths=0.6, alpha=0.65)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k",
               linestyles=["--", "-", "--"], linewidths=[1.1, 1.9, 1.1])

    acc = pipe.score(X_test, y_test)
    ax.set_title(f"{title}\ntest acc {acc:.3f}  |  {len(sv)} SVs",
                 fontsize=11, fontweight="bold")

axes[5].axis("off")
handles = [
    Line2D([], [], marker="o", ls="", color=C0, label="No cancer (0)"),
    Line2D([], [], marker="o", ls="", color=C1, label="Cancer (1)"),
    Line2D([], [], marker="o", ls="", mfc="none", mec="#222", label="Support vector"),
    Line2D([], [], color="k", lw=1.9, label="Decision boundary  $f(x)=0$"),
    Line2D([], [], color="k", lw=1.1, ls="--", label="Margins  $f(x)=\\pm1$"),
]
axes[5].legend(handles=handles, loc="center", frameon=False, fontsize=12)

for ax in axes[:5]:
    ax.set_xlabel(f1); ax.set_ylabel(f2)
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1])

fig.suptitle("SVM decision boundaries — gene expression dataset",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig("svm_gene_boundaries.png", dpi=150, bbox_inches="tight")
plt.show()


#print("duplicate rows:", dataset.duplicated().sum())
#tr = set(map(tuple, X_train.values))
#leak = sum(1 for t in map(tuple, X_test.values) if t in tr)
#print(f"test rows seen in train: {leak}/{len(X_test)} ({100*leak/len(X_test):.1f}%)")
