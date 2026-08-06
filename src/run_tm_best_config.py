import yaml
import json
import argparse 
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.data_loader import ROOT 
from src.topic_modeling.data_prepare import load_stopwords, prepare_data
from src.topic_modeling.embeddings import generate_embeddings
from src.topic_modeling.optimization import grid_search
from src.topic_modeling.model import train_topic_model, evaluate_model
from src.topic_modeling.export import export_topic_dictionary, export_visualizations


def load_config(config_path: str = "config.yaml") -> dict:
    """Lê o arquivo YAML e retorna um dicionário."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_model_name(model_name: str) -> str:
    """Transforma nome do modelo em nome de pasta seguro (sem barras)."""
    return model_name.replace("/", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Topic Modeling")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
        help="Nome do arquivo de configuração YAML (ex: config_teste.yaml)"
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    print(f"Carregando configurações de: {config_path}")
    config = load_config(config_path)

    input_path = ROOT / config["paths"]["input_path"]
    stopwords_path = ROOT / config["paths"]["stopwords"]
    cache_embeddings_dir = ROOT / config["paths"]["cache_embeddings_dir"]
    dir_reports_base = ROOT / config["paths"]["dir_reports"]
    sample_frac = config["sampling"]["frac"]
    embedding_names = config["model"]["embedding_names"]

    stopwords = load_stopwords(stopwords_path)

    print("| 1. Carregando amostra das mensagens...")
    documents_unique, df_unique, df_full = prepare_data(
        file_path=input_path,
        sample_frac=sample_frac
    )

    # 3. Carrega embeddings
    print("Carregando embeddings...")
    model_name = config["model"]["embedding_names"][0]
    cache_path = ROOT / config["paths"]["cache_embeddings_dir"] / f"{model_name}.npy"
    embeddings = generate_embeddings(
        documents_unique, 
        cache_path=cache_path,
        model_name=model_name
    )

    # 4. Extrai os parâmetros definidos no yaml
    best_params = {k: v[0] for k, v in config["grid_search"].items()}
    
    # 5. Treina o Modelo
    print("Treinando BERTopic final...")
    topic_model, topics = train_topic_model(
        documents=documents_unique,
        embeddings=embeddings,
        stopwords=stopwords,
        **best_params
    )

    reduce_outliers_flag = config.get("model", {}).get("reduce_outliers", False)

    if reduce_outliers_flag:
        print("Reduzindo Outliers (Tópico -1) usando embeddings...")
    
        new_topics = topic_model.reduce_outliers(
            documents=documents_unique, 
            topics=topics, 
            strategy="embeddings", 
            embeddings=embeddings
        )
        
        topic_model.update_topics(documents_unique, topics=new_topics)
        topics = new_topics
    
    # 6. Salva tudo (JSON, Parquet com id+channel e os 3 HTMLs)
    output_dir = ROOT / config["paths"]["dir_reports"] / model_name.replace("/", "_")
    export_topic_dictionary(topic_model, topics, df_unique, df_full, output_dir)
    export_visualizations(topic_model, output_dir)

    print(f"Resultados salvos em: {output_dir}")

if __name__ == "__main__":
    main()