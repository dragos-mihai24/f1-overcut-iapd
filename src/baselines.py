"""Phase 1 baselines: Dummy, keep-order rule, Gaussian Naive Bayes, Decision Tree, kNN.

Usage:  python -m src.baselines
Evaluation:
  * main dataset = green-flag duels (neutralized == 0)
  * 5-fold GroupKFold on 2018-2023, grouped by race (duels of one race never span train/test)
  * temporal hold-out: train 2018-2023, test 2024-2025
Writes reports/phase1/baselines.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import PRE_DECISION_NUMERIC, TARGET
from .sources.common import ROOT, load_config


class KeepOrderRule(BaseEstimator, ClassifierMixin):
    """Heuristic: whoever was ahead before the stops stays ahead (=> success iff A was ahead)."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.idx_ = PRE_DECISION_NUMERIC.index("a_was_ahead")
        return self

    def predict_proba(self, X):
        p = np.asarray(X)[:, self.idx_].astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def models() -> dict:
    imp = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "Dummy (majority)": DummyClassifier(strategy="most_frequent"),
        "Rule: keep order": KeepOrderRule(),
        "Gaussian NB": make_pipeline(imp(), StandardScaler(), GaussianNB()),
        "Decision Tree (d=4)": make_pipeline(imp(), DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                                                            random_state=0)),
        "kNN (k=25)": make_pipeline(imp(), StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
    }


def scores(y, pred, proba) -> dict:
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
    }


def main() -> None:
    cfg = load_config()
    d = pd.read_csv(ROOT / cfg["paths"]["processed"] / "duels.csv")
    d = d[d["neutralized"] == 0].reset_index(drop=True)
    X, y = d[PRE_DECISION_NUMERIC], d[TARGET].to_numpy()
    groups = d["season"].astype(str) + "_" + d["round"].astype(str)
    train = d["season"] <= 2023
    test = ~train
    print(f"green-flag duels: {len(d)}  (train 2018-23: {train.sum()}, test 2024-25: {test.sum()})")
    print(f"positive rate (undercut success): {y.mean():.3f}\n")

    rows = []
    for name, m in models().items():
        # grouped cross-validation on the training seasons
        cv = []
        Xt, yt, gt = X[train].to_numpy(), y[train], groups[train]
        for tr, va in GroupKFold(n_splits=5).split(Xt, yt, gt):
            m.fit(Xt[tr], yt[tr])
            cv.append(scores(yt[va], m.predict(Xt[va]), m.predict_proba(Xt[va])[:, 1]))
        cv = pd.DataFrame(cv)
        # temporal hold-out
        m.fit(Xt, yt)
        ho = scores(y[test], m.predict(X[test].to_numpy()), m.predict_proba(X[test].to_numpy())[:, 1]) \
            if test.sum() else {}
        row = {"model": name}
        row.update({f"cv_{k}": f"{cv[k].mean():.3f} ± {cv[k].std():.3f}" for k in cv.columns})
        row.update({f"test_{k}": round(v, 3) for k, v in ho.items()})
        rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    out = ROOT / "reports" / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "baselines.csv", index=False)


if __name__ == "__main__":
    main()
