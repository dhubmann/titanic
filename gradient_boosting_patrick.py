"""
Modul: gradient_boosting_patrick.py
Autor: Patrik
Beschreibung:
    Lose gekoppeltes Gradient-Boosting-Modell (LightGBM-Architektur / HistGradientBoosting)
    für die Titanic Fallstudie.
    
    Kann sowohl eigenständig als Skript ausgeführt werden als auch als Modul
    im Jupyter Notebook importiert werden, ohne bestehende Pipelines oder globale
    Variablen anderer Teammitglieder zu manipulieren.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance


def get_default_preprocessing():
    """
    Erstellt eine eigenständige Vorverarbeitungs-Pipeline,
    um Abhängigkeiten zu vorherigen Notebook-Zellen vollständig zu vermeiden.
    """
    numeric_features = ["Age", "Fare", "FamilySize"]
    categorical_features = ["Sex", "Pclass", "Embarked"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor, numeric_features, categorical_features


def create_gradient_boosting_model(preprocessor=None):
    """
    Erstellt das Gradient-Boosting-Pipeline-Modell.
    Nutzt HistGradientBoostingClassifier (LightGBM-Prinzip aus scikit-learn),
    um ohne plattformspezifische C++-Treiberprobleme (OpenMP) auf jedem System
    vollkommen stabil zu laufen.
    """
    if preprocessor is None:
        preprocessor, _, _ = get_default_preprocessing()
    else:
        # Clone, damit bestehende Pipelines nicht verändert werden
        preprocessor = clone(preprocessor)

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,        # Geringe Schrittweite für stabilere Konvergenz
            max_iter=100,              # Anzahl Boosting-Iterationen (Bäume)
            max_leaf_nodes=15,         # Entspricht ca. max_depth=4 zur Overfitting-Vermeidung
            min_samples_leaf=10,       # Glättung in den Endblättern
            l2_regularization=1.0,     # L2-Strafterm zur Regularisierung
            random_state=42
        )
        model_type = "HistGradientBoosting (LightGBM-Architektur)"
    except Exception as e:
        from sklearn.ensemble import GradientBoostingClassifier
        classifier = GradientBoostingClassifier(
            learning_rate=0.05,
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42
        )
        model_type = "GradientBoostingClassifier"

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    return pipeline, model_type


def evaluate_model(model, X_val, y_val, plot=False):
    """
    Berechnet alle relevanten Metriken für die Validierungsdaten.
    """
    predictions = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_val)[:, 1]
    else:
        probabilities = model.decision_function(X_val)

    tn, fp, fn, tp = confusion_matrix(y_val, predictions, labels=[0, 1]).ravel()

    metrics = {
        "Modell": "Gradient Boosting (Patrik)",
        "Accuracy": float(accuracy_score(y_val, predictions)),
        "Precision": float(precision_score(y_val, predictions, zero_division=0)),
        "Recall": float(recall_score(y_val, predictions, zero_division=0)),
        "F1": float(f1_score(y_val, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_val, probabilities)),
        "Nicht überlebt korrekt": int(tn),
        "Überlebt korrekt": int(tp),
        "Falsch als überlebt": int(fp),
        "Überlebende übersehen": int(fn)
    }

    if plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_val,
            predictions,
            display_labels=["Nicht überlebt", "Überlebt"],
            cmap="Greens",
            ax=ax
        )
        ax.set_title("Gradient Boosting: Validierungsdaten")
        plt.tight_layout()
        plt.show()

    return metrics, predictions, probabilities


def compute_feature_importance(model, X_val, y_val, feature_names=None, plot=False):
    """
    Berechnet die Permutation Feature Importance auf den Validierungsdaten.
    """
    result = permutation_importance(
        model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names if feature_names is not None else X_val.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.barh(importance_df["Feature"][::-1], importance_df["Importance_Mean"][::-1], color="teal")
        plt.xlabel("Permutation Importance (Mean Decrease in Accuracy)")
        plt.title("Feature Importance: Gradient Boosting")
        plt.tight_layout()
        plt.show()

    return importance_df


def predict_test_dataset(model, test_data_path="titanic/test.csv", output_path="titanic/predictions_gradient_boosting.csv"):
    """
    Generiert Vorhersagen für die Testmenge (418 Passagiere) und speichert sie als CSV.
    """
    test_df = pd.read_csv(test_data_path)
    if "FamilySize" not in test_df.columns and "SibSp" in test_df.columns and "Parch" in test_df.columns:
        test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

    feature_cols = ["Age", "Fare", "FamilySize", "Sex", "Pclass", "Embarked"]
    X_test = test_df[feature_cols]

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    result_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": preds
    })
    if probs is not None:
        result_df["SurvivalProbability"] = np.round(probs, 4)

    result_df.to_csv(output_path, index=False)
    print(f"Vorhersagen erfolgreich in '{output_path}' gespeichert.")
    print(f"Anzahl Passagiere: {len(result_df)}, vorhergesagte Überlebensrate: {preds.mean():.1%}")

    return result_df


if __name__ == "__main__":
    train_path = "titanic/train.csv"
    test_path = "titanic/test.csv"
    
    print(f"Lese Trainingsdaten von '{train_path}'...")
    train_df = pd.read_csv(train_path)
    train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1

    preprocessor, num_cols, cat_cols = get_default_preprocessing()
    features = num_cols + cat_cols

    X = train_df[features]
    y = train_df["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model, model_type = create_gradient_boosting_model(preprocessor)
    print(f"Trainiere {model_type}...")
    model.fit(X_train, y_train)

    metrics, _, _ = evaluate_model(model, X_val, y_val, plot=False)
    print("\n--- Validierungsergebnisse ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n--- Feature Importance ---")
    fi = compute_feature_importance(model, X_val, y_val, features, plot=False)
    print(fi.to_string(index=False))

    print("\n--- Test-Vorhersagen ---")
    predict_test_dataset(model, test_path)
