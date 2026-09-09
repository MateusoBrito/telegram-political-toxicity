import pandas as pd
import argparse
import yaml
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.topic_modeling.embeddings import generate_embeddings
from src.classifier.models import get_classifier
from src.classifier.trainer import ModelTrainer 
from src.run_compare_classifieds import load_config, sanitize_model_name

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Treinamento do melhor modelo para produção"
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Caminho para o arquivo YAML de configuração"
    )
    parser.add_argument(
        '--embeddings', type=str, default="all-distilroberta-v1",
        help="Modelo de embedding escolhido"
    )
    parser.add_argument(
        '--model', type=str, default="lr",
        help="Classificador escolhido"
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

    corpus_cfg = config.get('corpus', {})
    corpus_filter = corpus_cfg.get('filter', None)
    col_filter = corpus_cfg.get('col', None)
    filter_category = corpus_cfg.get('category', None)

    merge_keys = config['data']['merge_keys']
    text_col = config['data']['text_col']
    label_col = config['data']['label_col']

    # --- Definição dos Campeões ---
    emb_name = args.embeddings
    clf_name = args.model
    
    # Extrai apenas o grid de parâmetros do classificador selecionado
    PARAMS_GRID = config['classifiers'][clf_name]

    cache_embeddings_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        text_cols_to_load = list(set(merge_keys + [text_col]))

        topics_cols = merge_keys + [label_col]
        if corpus_filter == "just_politic" and col_filter:
            topics_cols.append(col_filter)
            
        topics_cols_to_load = list(set(topics_cols))

        print(f"Lendo texto com colunas: {text_cols_to_load}")
        df_text = pd.read_parquet(text_path, columns=text_cols_to_load)
        
        print(f"Lendo tópicos com colunas: {topics_cols_to_load}")
        df_topics = pd.read_parquet(topics_path, columns=topics_cols_to_load)
        
    except Exception as e:
        print(f"Erro ao carregar datasets: {e}")
        return

    df_merged = pd.merge(df_text, df_topics, on=merge_keys, how="inner")
    print(f"Dataset carregado: {len(df_merged)} documentos")

    if corpus_filter == "just_politic" and col_filter:
        print(f"Filtrando corpus para: {col_filter} == {filter_category}")
        df_merged = df_merged[df_merged[col_filter] == filter_category].copy()
    else:
        print("Nenhum filtro de corpus ativado. Usando a base completa.")

    documents = df_merged[text_col].tolist()
    print(f"Dataset carregado pós filtro: {len(documents)} documentos")
    
    print(f"Modelo Vencedor: {clf_name.upper()} com {emb_name}")

    # --- Label Encoding ---
    #le = LabelEncoder()
    #y = le.fit_transform(df_merged[label_col]) 
    #class_names = [str(cls) for cls in le.classes_]
    # --- Extração Direta dos Rótulos (Sem LabelEncoder) ---
    y = df_merged[label_col].astype(str).to_numpy()
    
    class_names = df_merged[label_col].unique().tolist()
    print(f"Classes encontradas para treino: {class_names}")

    emb_slug = sanitize_model_name(emb_name)
    cache_path = cache_embeddings_dir / f"{emb_slug}.pkl"

    print("| Gerando/carregando embeddings...")
    X = generate_embeddings(
        documents,
        cache_path=cache_path,
        model_name=emb_name
    )

    model_class = get_classifier(clf_name)

    print(f"\nOtimizando hiperparâmetros com GridSearchCV em 100% da amostra...")
    
    # Usamos cv=5 (ou o valor que preferir para a calibração final)
    trainer = ModelTrainer(model_class=model_class, param_grid=PARAMS_GRID, cv=5)
    
    # Otimiza e acha os melhores parâmetros no dataset completo
    trainer.optimize_hyperparameters(X, y)
    
    # Treino final (O GridSearchCV já faz isso via refit, mas garantimos aqui)
    trainer.train(X, y)

    # 4. Salva o modelo final compactado com joblib usando o método nativo do trainer
    # Define o caminho do artefato final
    model_path = output_dir / f"{clf_name}_{emb_slug}_producao.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    trainer.save(model_path)
    print(f"Pipeline de treino concluído. Modelo salvo para inferência em: {model_path}")

if __name__ == "__main__":
    main()