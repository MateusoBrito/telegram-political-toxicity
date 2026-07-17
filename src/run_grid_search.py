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

    cache_embeddings_dir.mkdir(parents=True, exist_ok=True)

    for i, model_name in enumerate(embedding_names, start=1):
        model_slug = sanitize_model_name(model_name)
        print(f"\n{'#'*60}")
        print(f"[{i}/{len(embedding_names)}] Modelo de embedding: {model_name}")
        print(f"{'#'*60}")

        cache_path = cache_embeddings_dir / f"{model_slug}.pkl"

        print("| 2. Gerando embeddings...")
        embeddings = generate_embeddings(
            documents_unique, 
            cache_path=cache_path,
            model_name=model_name
        )

        # Diretório de saída específico por modelo
        dir_reports_model = dir_reports_base / model_slug
        dir_reports_model.mkdir(parents=True, exist_ok=True)

        print("| 3. Busca em Grade de hiperparâmetros...")
        df_results = grid_search(
            documents_unique, 
            embeddings, 
            stopwords, 
            param_grid=config["grid_search"], 
            output_dir=dir_reports_model
        )

        print(f"Resultados salvos em: {dir_reports_model}")

if __name__ == "__main__":
    main()