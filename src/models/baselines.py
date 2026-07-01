import json
import logging
import os

import joblib
import mlflow
import mlflow.sklearn as mlflow_sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Importação dos módulos internos desenvolvidos por nós
from src.data import build_feature_transformer, load_raw_data, preprocess_data, validate_raw

# Configuração de logs para monitorização da execução
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_log_pipeline(model_name: str, model, X_train, X_test, y_train, y_test) -> None:
    """
    Constrói um Pipeline fresco (transformer + modelo), executa o treino,
    avalia as métricas no conjunto de teste, regista tudo no MLflow e persiste o artefacto físico.
    """
    # Transformer instanciado aqui para garantir estado isolado por pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', build_feature_transformer()),
        ('classifier', model)
    ])

    # Inicialização do ciclo de monitorização do MLflow
    with mlflow.start_run(run_name=model_name):
        logging.info(f"A iniciar o treino do pipeline para: {model_name}...")
        pipeline.fit(X_train, y_train)

        # Predições seguras utilizando o pipeline estruturado (evita discrepâncias de colunas)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Cálculo das métricas regulamentadas no projeto
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        # Registo de metadados e performance no MLflow tracking
        mlflow.log_metric("F1_Score", float(f1))
        mlflow.log_metric("ROC_AUC", float(roc_auc))
        mlflow.log_metric("PR_AUC", float(pr_auc))
        mlflow.log_param("algoritmo", type(model).__name__)

        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # Registo do artefacto de pipeline completo no ecossistema MLflow
        mlflow_sklearn.log_model(pipeline, f"pipeline_{model_name}")

        # Persistência física do pipeline estruturado para entrega em produção
        caminho_joblib = f"models/{model_name}.joblib"
        joblib.dump(pipeline, caminho_joblib)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": float(f1),
            "AUC-ROC":  float(roc_auc),
            "PR-AUC":   float(pr_auc),
        }
        with open(f"models/{model_name}_metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

        logging.info(f"=== RESULTADOS: {model_name} ===")
        logging.info(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1-Score: {f1:.4f}")
        logging.info(f"✅ Pipeline completo (Features + Modelo) guardado em: {caminho_joblib}")
        logging.info("\n" + str(classification_report(y_test, y_pred)))
        logging.info("-" * 60)


def main():
    # Criação das pastas de destino de dados e modelos caso não existam
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Configuração de infraestrutura do MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("TechChallenge_TelcoChurn")

    # 2. Pipeline de Ingestão e Limpeza de dados (Módulos 1 e 2)
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df_raw = load_raw_data(url)
    validate_raw(df_raw)
    df_clean = preprocess_data(df_raw)

    # Salvamento local do snapshot de dados limpos
    df_clean.to_csv("data/telco_churn_preprocessed.csv", index=False)
    logging.info("Cópia dos dados processados guardada em: data/telco_churn_preprocessed.csv")

    # 3. Segregação de variáveis independentes (X) e variável dependente target (y)
    X = df_clean.drop(columns=["target"], errors="ignore")
    y = df_clean["target"]

    # 4. Divisão de Treino e Teste com estratificação para mitigar o desbalanceamento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Execução e avaliação dos Modelos Baseline
    # Baseline 1: Dummy Classifier
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    train_and_log_pipeline("baseline_dummy_model", dummy, X_train, X_test, y_train, y_test)

    # Baseline 2: Regressão Logística
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    train_and_log_pipeline("baseline_logistic_regression", lr, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()


