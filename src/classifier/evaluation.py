import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, labels=None):
    """
    Calcula as principais métricas de classificação.
    
    Args:
        y_true: Array com os rótulos verdadeiros.
        y_pred: Array com os rótulos preditos pelo modelo.
        labels (list, opcional): Lista com os nomes das classes (para o relatório).
        
    Returns:
        dict: Um dicionário contendo 'accuracy', 'f1_macro', 'f1_weighted' e 'report'.
    """

    metrics = {}

    # 1. Métricas Escalares (Número únicos para comparação rápida)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')

    # 2. Relatório Completo
    if labels:
        labels = [str(l) for l in labels]
        metrics['report'] = classification_report(y_true, y_pred, target_names=labels)
    else:
        metrics['report'] = classification_report(y_true, y_pred)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels, filepath=None):
    """
    Gera e salva a matriz de confusão como uma imagem.
    """

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=labels,
        yticklabels=labels)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.title('Matriz de confusão')

    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)
        plt.close()

def save_results(metrics, experiment_info, filepath):
    """
    Salva as métricas e informações do experimento em um arquivo CSV

    Args:
        metrics (dict): Dicionário retornado por evaluate_model.
        experiment_info (dict): Infos extras (nome do dataset, modelo, parâmetros).
        filepath (str): Caminho para o arquivo .csv de log.
    """
        
    data = experiment_info.copy()
    data['accuracy'] =  metrics['accuracy']
    data['f1_macro'] =  metrics['f1_macro']
    data['f1_weighted'] = metrics['f1_weighted']

    df = pd.DataFrame([data])

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    file_exists = os.path.isfile(filepath)
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)
    print(f"Resultados salvos em: {filepath}")

