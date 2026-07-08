from src.utils.data_loader import get_spark_session, load_data, ROOT
from pyspark.sql.functions import col, year, month, count, concat_ws, when, rand
from pathlib import Path

INPUT_PATH = ROOT / "data" / "processed" / "messages_preprocessed"
OUTPUT_PATH = ROOT / "data" / "processed" / "stratified_sample"

TOTAL_SAMPLE_SIZE = 1000000 

if __name__ == "__main__":
    print("Fase 1: Amostragem Estratificada - Extraindo apenas IDs...")
    print("="*50)
    
    spark = get_spark_session("stratifiedSampleIDs")

    df = spark.read.parquet(str(INPUT_PATH)).select("id", "datetime", "group_name")

    total_msg = df.count()
    print(f"Total de mensagens: {total_msg}")

    df_filtered = df.filter(
        (month(col("datetime")) >= 7) & 
        (month(col("datetime")) <= 12) & 
        (year(col("datetime")) == 2024)
    ).withColumn("msg_month", month(col("datetime")))

    # Cria um estrato (mês_grupo) de cada mensagem
    df_filtered = df_filtered.withColumn("strat_key", concat_ws("_", col("msg_month"), col("group_name")))

    total_msg_filtered = df_filtered.count()
    print(f"Total de mensagens filtradas: {total_msg_filtered}")

    # Conta quantas mensagens tem em cada estrato (Quantas mensagens tem em cada mês de cada grupo)
    counts_df = df_filtered.groupBy("strat_key").agg(count("*").alias("channel_month_count"))
    
    from pyspark.sql.functions import round

    counts_df = counts_df.withColumn(
        "sample_size",
        round(
            (col("channel_month_count") / total_msg_filtered)
            * TOTAL_SAMPLE_SIZE
        ).cast("int")
    )

    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number
    from pyspark.sql.functions import rand

    df_joined = df_filtered.join(
        counts_df.select("strat_key", "sample_size"),
        on="strat_key"
    )

    window = Window.partitionBy("strat_key").orderBy(rand(seed=42))

    df_joined = df_joined.withColumn(
        "rn",
        row_number().over(window)
    )
    
    df_sample = df_joined.filter(
        col("rn") <= col("sample_size")
    )

    OUTPUT_IDS_PATH = ROOT / "data" / "processed" / "stratified_sample_ids"
    df_sample_ids = (
        df_sample
        .select("id", "group_name")
    )
    
    sample_count = df_sample_ids.count()
    print(f"Amostra final: {sample_count:,} mensagens")
    
    df_sample_ids.coalesce(10) \
        .write \
        .mode("overwrite") \
        .parquet(str(OUTPUT_IDS_PATH))