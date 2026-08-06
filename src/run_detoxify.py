import os
import sys 

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pathlib import Path
import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, FloatType

from src.utils.data_loader import get_spark_session, load_data, ROOT

INPUT_PATH  = ROOT / "data" / "processed" / "messages_preprocessed"
OUTPUT_PATH = ROOT / "data" / "processed" / "toxicity"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def analyze_partition_iterator(iterator):
    """
    Processa os dados em lotes por partição.
    O iterador recebe e retorna Pandas DataFrames.
    """
    import torch
    import pandas as pd
    from detoxify import Detoxify
    
    # Otimização crucial: impede que o PyTorch sufoque os cores do worker
    torch.set_num_threads(1)
    
    # Inicializa o modelo (mude device='cuda' se tiver GPU nos workers)
    model = Detoxify('original', device='cpu')
    batch_size = 256 

    for pdf in iterator:
        if pdf.empty:
            yield pdf
            continue

        texts = pdf['clean_text'].fillna("").astype(str).tolist()

        results = {
            'toxicity': [], 'severe_toxicity': [], 'obscene': [],
            'threat': [], 'insult': [], 'identity_attack': []
        }

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            if not batch or all(t.strip() == "" for t in batch):
                for k in results.keys():
                    results[k].extend([0.0] * len(batch))
                continue

            predictions = model.predict(batch)

            for key in results.keys():
                results[key].extend(predictions[key])
        
        for key, values in results.items():
            pdf[key] = values
            
        yield pdf


if __name__ == "__main__":
    spark = get_spark_session("processDetoxifyDynamic")
    
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    # A mágica: Sobrescreve apenas as partições filtradas, sem apagar a pasta toda
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    
    print("Carregando base de dados...")
    columns_ip = ["id", "group_name", "clean_text", "datetime", "year", "month"]
    df = load_data(spark, INPUT_PATH, columns_ip)

    # 1. Definir o Schema de saída 
    output_schema = df.schema
    new_columns = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
    for col_name in new_columns:
        output_schema = output_schema.add(StructField(col_name, FloatType(), True))

    # 2. Coletar partições disponíveis
    print("Mapeando meses disponíveis para inferência...")
    meses_unicos = df.select("year", "month").distinct().orderBy("year", "month").collect()
    
    print(f"Total de blocos mensais a processar: {len(meses_unicos)}")
    
    # 3. Processamento Incremental
    for row in meses_unicos:
        ano = row['year']
        mes = row['month']
        
        if ano is None or mes is None:
            continue
            
        partition_dir = OUTPUT_PATH / f"year={ano}" / f"month={mes}"
        if partition_dir.exists() and any(partition_dir.glob("*.parquet")):
            print(f"Pulando Lote: {ano}-{mes:02d} (Já processado anteriormente)")
            continue
            
        print(f"\n--- Processando Lote: {ano}-{mes:02d} ---")

        # Filtra os dados apenas para o mês atual
        df_mes = df.filter((col("year") == ano) & (col("month") == mes))
        
        # Aplica o modelo
        df_result = df_mes.mapInPandas(analyze_partition_iterator, schema=output_schema)
        
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
            
    print("\nProcessamento total do Detoxify finalizado!")
    spark.stop()