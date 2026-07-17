from src.utils.data_loader import get_spark_session, ROOT
from pyspark.sql.functions import broadcast

INPUT_PATH = ROOT / "data" / "processed" / "messages_preprocessed"
IDS_PATH = ROOT / "data" / "processed" / "stratified_sample_ids"
OUTPUT_PATH = ROOT / "data" / "processed" / "stratified_sample_full"

spark = get_spark_session("materializeSample")

df_full = spark.read.parquet(str(INPUT_PATH))
df_ids = spark.read.parquet(str(IDS_PATH))

df_sample_full = df_full.join(broadcast(df_ids), on=["id", "group_name"], how="left_semi")

print(f"Amostra completa: {df_sample_full.count():,} mensagens")

df_sample_full.coalesce(10) \
    .write \
    .mode("overwrite") \
    .parquet(str(OUTPUT_PATH))