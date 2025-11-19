#!/usr/bin/env python3
"""
main.py

Projeto de Machine Learning para classificação da qualidade do vinho tinto.
Inclui:
- EDA mínima
- Pré-processamento
- Random Forest e SVM com GridSearchCV
- Salvamento de modelos, métricas e figuras
"""

import os
import argparse
import json
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import joblib
import matplotlib.pyplot as plt

RANDOM_STATE = 42


# ----------------------------------------------------------
# Carregar CSV com detecção automática do separador
# ----------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    # Tenta separador ';'
    df = pd.read_csv(csv_path, sep=';', engine='python')

    # Se só tem 1 coluna, tenta ','
    if df.shape[1] == 1:
        df = pd.read_csv(csv_path, sep=',', engine='python')

    # Limpa nomes das colunas
    df.columns = df.columns.str.strip().str.lower()

    return df


# ----------------------------------------------------------
# Preparar variável alvo
# ----------------------------------------------------------
def prepare_target(df: pd.DataFrame, threshold: int = 7):
    if 'quality' not in df.columns:
        raise ValueError(f"Coluna 'quality' não encontrada. Colunas disponíveis: {df.columns.tolist()}")

    X = df.drop(columns=['quality'])
    y = (df['quality'] >= threshold).astype(int)

    return X, y


# ----------------------------------------------------------
# Configurações de modelos + grids
# ----------------------------------------------------------
def get_model_grids():
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
    rf_param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }

    svm = SVC(probability=True, random_state=RANDOM_STATE)
    svm_param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['rbf', 'linear'],
        'svc__gamma': ['scale', 'auto']
    }

    return {
        'random_forest': {
            'pipeline': Pipeline([('clf', rf)]),
            'param_grid': rf_param_grid
        },
        'svm': {
            'pipeline': Pipeline([('scaler', StandardScaler()), ('svc', svm)]),
            'param_grid': svm_param_grid
        }
    }


# ----------------------------------------------------------
# Função de treinamento + GridSearchCV
# ----------------------------------------------------------
def train_and_tune(pipeline, param_grid, X_train, y_train, cv_folds=5, scoring='f1'):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    grid.fit(X_train, y_train)
    return grid


# ----------------------------------------------------------
# Avaliação do modelo
# ----------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    # Probabilidades
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["roc_auc"] = None

    return metrics


# ----------------------------------------------------------
# Plot da matriz de confusão
# ----------------------------------------------------------
def plot_confusion_matrix(cm, labels, out_path, title="Confusion Matrix"):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)

    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )

    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------
# Função principal
# ----------------------------------------------------------
def main(args):
    print(f"[INFO] Carregando dataset: {args.data_path}")

    df = load_data(args.data_path)
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Colunas: {df.columns.tolist()}")

    X, y = prepare_target(df, threshold=args.quality_threshold)
    print(f"[INFO] Classes: {y.value_counts().to_dict()}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )

    print("[INFO] Treino/Teste:", X_train.shape, X_test.shape)

    # Diretórios
    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, "models")
    reports_dir = os.path.join(args.output_dir, "reports")
    figures_dir = os.path.join(args.output_dir, "figures")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Modelos
    models = get_model_grids()
    results = {}

    for name, cfg in models.items():
        print(f"\n[INFO] Treinando modelo: {name.upper()}")

        grid = train_and_tune(cfg["pipeline"], cfg["param_grid"],
                              X_train, y_train,
                              cv_folds=args.cv_folds,
                              scoring=args.scoring)

        best_model = grid.best_estimator_

        print(f"[INFO] Best params: {grid.best_params_}")
        print(f"[INFO] Best CV Score: {grid.best_score_}")

        metrics = evaluate_model(best_model, X_test, y_test)

        # Salvar modelo
        model_path = os.path.join(models_dir, f"{name}.joblib")
        joblib.dump(best_model, model_path)

        # Salvar matriz de confusão
        cm = metrics["confusion_matrix"]
        cm_path = os.path.join(figures_dir, f"confusion_matrix_{name}.png")
        plot_confusion_matrix(cm, ["not_good", "good"], cm_path)

        # Salvar métricas
        metrics_path = os.path.join(reports_dir, f"metrics_{name}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        results[name] = {
            "best_params": grid.best_params_,
            "best_cv": grid.best_score_,
            "metrics": metrics
        }

    # Salvar resumo geral
    summary_path = os.path.join(reports_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n✔ FINALIZADO! Todos os artefatos foram salvos em:", args.output_dir)


# ----------------------------------------------------------
# Execução via terminal
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="f1")
    parser.add_argument("--quality_threshold", type=int, default=7)

    args = parser.parse_args()
    main(args)
