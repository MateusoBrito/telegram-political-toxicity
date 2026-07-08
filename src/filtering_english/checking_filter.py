from src.utils.data_loader import get_spark_session, ROOT

from pyspark.sql.functions import col

INPUT_PATH = ROOT / "data" / "processed" / "messages_with_language"


if __name__ == "__main__":

    spark = get_spark_session("ReadFilteredMessages")

    print(f"Lendo dados de: {INPUT_PATH}")

    df = spark.read.parquet(str(INPUT_PATH))

    print("\nEsquema do dataset:")
    df.printSchema()

    total_mensagens = df.count()

    print(f"\nTotal de mensagens processadas: {total_mensagens:,}")

    idiomas = (
        df.groupBy("language")
        .count()
        .orderBy(col("count").desc())
    )

    print("\nDistribuição de idiomas:")
    idiomas.show(20, truncate=False)

    ingles = df.filter(col("language") == "en").count()

    print(f"\nTotal de mensagens em inglês: {ingles:,}")

    percentual = (ingles / total_mensagens) * 100

    print(f"Percentual em inglês: {percentual:.2f}%")

    print("\nExemplos de mensagens em inglês:")

    (
        df.filter(col("language") == "en")
        .select("group_name", "text_clean")
        .show(20, truncate=2000)
    )

    print("\nExemplos de mensagens em outro idioma:")

    (
        df.filter(col("language") != "en")
        .select("group_name", "text_clean")
        .show(20, truncate=2000)
    )