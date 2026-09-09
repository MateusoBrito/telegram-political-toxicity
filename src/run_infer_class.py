import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import joblib
import argparse
import yaml
from pathlib import Path
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType

from src.utils.data_loader import get_spark_session, load_data, ROOT
from src.run_compare_classifieds import load_config, sanitize_model_name

INPUT_PATH = ROOT / "data" / "processed" / "messages_preprocessed"
OUTPUT_PATH = ROOT / "data" / "processed" / "messages_classified_topics"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

PK_COLS = ["id", "group_name"]
SAVE_FULL_ROW = True


# ────────────────────────────────────────────────────────────────
# PREDIÇÃO POR PARTIÇÃO (roda nos workers)
# ────────────────────────────────────────────────────────────────
def make_predict_partition_iterator(model_bc, embedding_model_name, text_col):
    """
    Fábrica de iteradores para o mapInPandas. 
    Recebe o modelo via broadcast e os parâmetros textuais do config.
    """
    def predict_partition_iterator(iterator):
        from sentence_transformers import SentenceTransformer

        # Carrega o modelo de embeddings na memória do worker
        embedder = SentenceTransformer(embedding_model_name)
        model = model_bc.value
        batch_size = 256

        for pdf in iterator:
            if pdf.empty:
                yield pdf[PK_COLS + ["predicted_class"]] if not SAVE_FULL_ROW else pdf
                continue

            texts = pdf[text_col].fillna("").astype(str).tolist()
            preds = []
            
            # Processamento em lotes para evitar estouro de memória
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                embeddings = embedder.encode(batch_texts, show_progress_bar=False)
                preds.extend(model.predict(embeddings))

            pdf["predicted_class"] = preds
            yield pdf if SAVE_FULL_ROW else pdf[PK_COLS + ["predicted_class"]]

    return predict_partition_iterator


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Inferência Distribuída (PySpark)"
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
    text_col = config['data']['text_col']
    output_dir = ROOT / config['paths']['output_dir']

    emb_name = args.embeddings
    clf_name = args.model
    emb_slug = sanitize_model_name(emb_name)

    # --- Carregamento do Modelo de Produção ---
    model_path = output_dir / f"{clf_name}_{emb_slug}_producao.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Erro: Modelo de produção não encontrado em {model_path}. Execute o script de treino primeiro.")
        
    print(f"Carregando modelo treinado de {model_path}...")
    model = joblib.load(model_path)

    # --- Inicialização do Spark ---
    spark = get_spark_session("classifies_messages")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    # Transmite o modelo treinado para todos os workers
    model_bc = spark.sparkContext.broadcast(model)

    print("Carregando base de dados para inferência...")
    columns_ip = PK_COLS + [text_col, "datetime", "year", "month"]
    df = load_data(spark, INPUT_PATH, columns_ip)

    # Define o schema de saída
    if SAVE_FULL_ROW:
        output_schema = df.schema.add(StructField("predicted_class", StringType(), True))
    else:
        output_schema = StructType(
            [StructField(c, StringType(), True) for c in PK_COLS] + 
            [StructField("predicted_class", StringType(), True)]
        )

    print("Mapeando meses disponíveis para inferência...")
    meses_unicos = df.select("year", "month").distinct().orderBy("year", "month").collect()
    print(f"Total de blocos mensais a processar: {len(meses_unicos)}")

    for row in meses_unicos:
        ano, mes = row["year"], row["month"]
        if ano is None or mes is None:
            continue

        partition_dir = OUTPUT_PATH / f"year={ano}" / f"month={mes}"
        if partition_dir.exists() and any(partition_dir.glob("*.parquet")):
            print(f"Pulando Lote: {ano}-{mes:02d} (já processado anteriormente)")
            continue

        print(f"\n--- Processando Lote: {ano}-{mes:02d} ---")
        df_mes = df.filter((col("year") == ano) & (col("month") == mes))
        
        # Passa as variáveis para a fábrica de iteradores
        df_result = df_mes.mapInPandas(
            make_predict_partition_iterator(model_bc, emb_name, text_col), 
            schema=output_schema
        )

        try:
            (
                df_result.write
                .mode("overwrite")
                .partitionBy("year", "month")
                .parquet(str(OUTPUT_PATH))
            )
            print(f"Lote {ano}-{mes:02d} salvo com sucesso!")
        except Exception as e:
            print(f"ERRO CRÍTICO no lote {ano}-{mes:02d}: {e}")

    print("\nClassificação em batches finalizada!")
    spark.stop()


if __name__ == "__main__":
    main()