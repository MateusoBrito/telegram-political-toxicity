from pathlib import Path
from datetime import datetime
from langdetect import detect, DetectorFactory

from pyspark.sql.functions import (col,length,udf,from_unixtime,year,month, regexp_replace)
from pyspark.sql.types import StringType

from src.utils.data_loader import (get_spark_session,load_data,ROOT)

OUTPUT_PATH = ROOT / "data" / "processed" / "messages_with_language"

DetectorFactory.seed = 0


def initialize_spark(app_name):
    spark = get_spark_session(app_name)

    df_mensagens = load_data(
        spark,
        columns=["id","group_name","text","timestamp", "file_path"])

    return df_mensagens

def filter_messages(df):
    df = df.withColumn(
        "text_clean",
        regexp_replace(col("text"), r"http\S+", "")
    )

    df = df.withColumn(
        "text_clean",
        regexp_replace(col("text_clean"), r"@\w+", "")
    )

    df = df.filter(
        col("text_clean").isNotNull() &
        (length(col("text_clean")) >= 20)
    )

    return df
    

def detect_language(text):

    if not text or not str(text).strip():
        return "unknown"

    try:
        return detect(str(text))

    except:
        return "unknown"

detect_language_udf = udf(detect_language, StringType())

if __name__ == "__main__":

    spark = get_spark_session("filterMessagesByLanguage")
    df = initialize_spark("filterMessagesByLanguage")
    df = filter_messages(df)
    
    total_filtrado = df.count()
    print(f"Mensagens após pré processamento básico: {total_filtrado:,}")


    df = df.withColumn("datetime",from_unixtime(col("timestamp")))
    df = df.withColumn("year",year(col("datetime")))
    df = df.withColumn("month",month(col("datetime")))

    meses = (df
        .select("year", "month")
        .distinct()
        .orderBy("year", "month")
        .collect())

    print(f"\nTotal de meses encontrados: {len(meses)}")


    for row in meses:

        ano = row["year"]
        mes = row["month"]

        print(f"\nProcessando {ano}-{mes:02d}")

        batch = df.filter(
            (col("year") == ano) &
            (col("month") == mes)
        )

        batch = batch.repartition(100)
        batch = batch.withColumn(
            "language",
            detect_language_udf(col("text_clean"))
        )

        (
            batch
            .write
            .mode("append")
            .partitionBy("year", "month")
            .parquet(str(OUTPUT_PATH))
        )

        print(f"Lote {ano}-{mes:02d} salvo com sucesso.")

    print("\nProcessamento finalizado.")