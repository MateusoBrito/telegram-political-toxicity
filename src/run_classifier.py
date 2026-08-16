import pandas as pd
import argparse
import yaml
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.topic_modeling.embeddings import generate_embeddings
from src.classifier.models import get_classifier
from src.classifier.validation import get_validation_strategy
from src.classifier.classic_pipeline import run_classic_pipeline

ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str) -> dict:
    """Lê o arquivo YAML e retorna um dicionário."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_model_name(model_name: str) -> str:
    """Transforma nome do modelo em nome de pasta seguro (sem barras)."""
    return model_name.replace("/", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Avaliação de Classificadores com Embeddings"
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Caminho para o arquivo YAML de configuração"
    )
    args = parser.parse_args()
    config_path = ROOT / args.config

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Erro ao carregar configuração: {e}")
        return

    # --- Extração de variáveis do YAML ---
    text_path = ROOT / config['paths']['text_path']
    topics_path = ROOT / config['paths']['topics_path']
    cache_embeddings_dir = ROOT / config['paths']['cache_embeddings_dir']
    output_dir = ROOT / config['paths']['output_dir']

    merge_keys = config['data']['merge_keys']
    text_col = config['data']['text_col']
    label_col = config['data']['label_col']

    embedding_names = config['embeddings']['models']
    train_cfg = config['training']
    classifiers_cfg = config['classifiers']

    cache_embeddings_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Carregamento dos Dados ---
    try:
        df_text = pd.read_parquet(text_path)
        df_topics = pd.read_parquet(topics_path)
    except Exception as e:
        print(f"Erro ao carregar datasets: {e}")
        return

    df_merged = pd.merge(df_text, df_topics, on=merge_keys, how="inner")
    documents = df_merged[text_col].tolist()

    print(f"Dataset carregado: {len(documents)} documentos")
    print(f"Embeddings a testar: {embedding_names}")
    print(f"Classificadores a testar: {list(classifiers_cfg.keys())}")

    # --- Label Encoding ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_merged[label_col])
    class_names = [str(cls) for cls in le.classes_]
    print(f"Classes encontradas: {class_names}")

    # --- Validação Cruzada ---
    splitter = get_validation_strategy(
        train_cfg['validation'],
        n_splits=train_cfg['n_splits']
    )

    # --- Loop: Embedding Models × Classificadores ---
    for i, emb_name in enumerate(embedding_names, start=1):
        emb_slug = sanitize_model_name(emb_name)
        print(f"\n{'#'*60}")
        print(f"[{i}/{len(embedding_names)}] Modelo de embedding: {emb_name}")
        print(f"{'#'*60}")

        cache_path = cache_embeddings_dir / f"{emb_slug}.pkl"

        print("| Gerando/carregando embeddings...")
        embeddings = generate_embeddings(
            documents,
            cache_path=cache_path,
            model_name=emb_name
        )

        for clf_name, param_grid in classifiers_cfg.items():
            print(f"\n{'='*50}")
            print(f"  Classificador: {clf_name.upper()} | Embedding: {emb_name}")
            print(f"{'='*50}")

            model_class = get_classifier(clf_name)
            clf_output_dir = output_dir / emb_slug / clf_name

            run_classic_pipeline(
                X=embeddings,
                y_encoded=y_encoded,
                splitter=splitter,
                model_class=model_class,
                param_grid=param_grid,
                model_name=clf_name,
                embedding_name=emb_name,
                dataset_path=str(text_path),
                output_dir=str(clf_output_dir),
                class_names=class_names
            )

    print(f"\n{'#'*60}")
    print(f"Pipeline concluído! Resultados em: {output_dir}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()