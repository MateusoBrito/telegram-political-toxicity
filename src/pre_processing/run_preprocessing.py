from pathlib import Path
import pandas as pd
from pyspark.sql.functions import col, pandas_udf, from_unixtime, year, month, length
from pyspark.sql.types import StringType
import nltk

from src.utils.data_loader import get_spark_session, load_data, ROOT
from src.pre_processing.preprocessing import TextPreprocessor

INPUT_PATH  = ROOT / "data" / "processed" / "messages_with_language"
OUTPUT_PATH = ROOT / "data" / "processed" / "messages_preprocessed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 1. Força o download do NLTK apenas UMA vez no nó principal (Driver)
def setup_nltk_driver():
    print("Verificando recursos do NLTK no Driver...")
    preprocessor_dummy = TextPreprocessor()
    preprocessor_dummy._download_nltk_resources()
    print("Recursos NLTK prontos!")

# 2. A UDF agora apenas processa, sem tentar baixar nada da internet
@pandas_udf(StringType())
def preprocess_udf(texts: pd.Series) -> pd.Series:
    # A classe não deve mais ter o _download no __init__
    preprocessor = TextPreprocessor()
    return texts.apply(preprocessor.preprocess)

if __name__ == "__main__":
    setup_nltk_driver()

    spark = get_spark_session("preprocessMessagesMonthly")

    # Configurações agressivas de memória e particionamento dinâmico
    spark.conf.set("spark.sql.shuffle.partitions", "200")
    spark.conf.set("spark.sql.adaptive.enabled", "true")          
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    # ESSENCIAL: Permite sobrescrever apenas o mês que está sendo processado
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    print("Carregando base de dados...")
    df = load_data(spark, INPUT_PATH)
    
    # Filtra apenas inglês
    df_en = df.filter(col("language") == "en")

    # Cria as colunas de ano e mês a partir do timestamp original
    df_en = df_en.withColumn("date_type", from_unixtime(col("timestamp")))
    df_en = df_en.withColumn("year", year(col("date_type")))
    df_en = df_en.withColumn("month", month(col("date_type")))

    # Coleta quais meses existem na base (processamento rápido no Parquet)
    print("Mapeando meses disponíveis...")
    meses_unicos = df_en.select("year", "month").distinct().orderBy("year", "month").collect()
    
    print(f"Total de blocos mensais a processar: {len(meses_unicos)}")

    # 3. Processamento Incremental (Mês a Mês)
    for row in meses_unicos:
        ano = row["year"]
        mes = row["month"]
        
        # O Spark frequentemente gera nulos em conversões de datas inválidas
        if ano is None or mes is None:
            continue
            
        print(f"\n--- Processando Lote: {ano}-{mes:02d} ---")
        
        # Filtra o DataFrame apenas para aquele mês específico
        df_mes = df_en.filter((col("year") == ano) & (col("month") == mes))
        
        # Aplica o NLP
        df_mes_clean = df_mes.withColumn("clean_text", preprocess_udf(col("text")))

        df_mes_clean = df_mes_clean.filter(
            col("clean_text").isNotNull() &
            (col("clean_text") != "") &
            (length(col("clean_text")) > 20)
        )

        try:
            # Salva o mês específico. O mode("overwrite") + dynamic garante que
            # ele só substitua os arquivos deste exato ano/mês, sem tocar no resto.
            (
                df_mes_clean.write
                .mode("overwrite")
                .partitionBy("year", "month")
                .parquet(str(OUTPUT_PATH))
            )
            print(f"Lote {ano}-{mes:02d} salvo com sucesso!")
            
        except Exception as e:
            print(f"ERRO CRÍTICO no lote {ano}-{mes:02d}: {e}")
            # Se um mês quebrar, o script apenas avisa e segue para o próximo mês

    print("\nProcessamento incremental finalizado!")
    spark.stop()                                                                        