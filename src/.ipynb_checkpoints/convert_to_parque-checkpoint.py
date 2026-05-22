import os
import sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, BooleanType, IntegerType

# ============================== FORÇAR VERSÃO DO PYTHON ==============================
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# ============================== CONFIG ==============================
# Lendo da sua pasta antiga que já tem os textos em inglês!
INPUT_DIR = "../../messages_july_en"
OUTPUT_DIR = "../../messages_july_en_parquet"

print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando Spark com máxima memória...")

spark = SparkSession.builder \
    .appName("convert_csv_to_parquet") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ============================== SCHEMA COMPLETO (18 Colunas) ==============================
# Inclui o file_path e o lang que você já havia gerado no script original
schema = StructType([
    StructField("id", LongType(), True),    
    StructField("user_id", LongType(), True),
    StructField("text", StringType(), True),
    StructField("timestamp", LongType(), True),                  
    StructField("bot_flag", BooleanType(), True),                
    StructField("via_bot_id", LongType(), True),             
    StructField("via_business_bot_id", LongType(), True),    
    StructField("reply_to_msg_id", LongType(), True),        
    StructField("fwd_flag", BooleanType(), True),                
    StructField("fwd_from_id", LongType(), True),            
    StructField("media_type", StringType(), True),               
    StructField("views", IntegerType(), True),                   
    StructField("forwards", IntegerType(), True),                
    StructField("replies", IntegerType(), True),                 
    StructField("reactions", IntegerType(), True),       
    StructField("reaction_json", StringType(), True),
    StructField("file_path", StringType(), True),
    StructField("lang", StringType(), True)
])

print(f"[{datetime.now().strftime('%H:%M:%S')}] Lendo os dados já filtrados em inglês...")

try:
    # 1. Leitura do seu diretório CSV antigo
    df_csv = spark.read.option("header", True) \
        .option("delimiter", "\t") \
        .option("multiline", True) \
        .option("quote", '"') \
        .option("escape", '"') \
        .schema(schema) \
        .csv(INPUT_DIR)  # <--- Deixe apenas a variável!

    # 2. Remoção das duplicatas (Para consertar os 35M -> 28M)
    # Extraímos o channel_id do file_path caso não esteja explícito
    from pyspark.sql.functions import col, regexp_extract
    df_csv = df_csv.withColumn("channel_id_extracted", regexp_extract(col("file_path"), r"channel_(\d+)-", 1))
    
    df_clean = df_csv.dropDuplicates(["id", "channel_id_extracted"])

    # Removemos a coluna auxiliar para deixar o dado limpo
    df_final = df_clean.drop("channel_id_extracted")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Removendo duplicatas e convertendo para Parquet...")

    # 3. Salvamento direto em Parquet
    (
        df_final.write
        .mode("overwrite")
        .parquet(OUTPUT_DIR)
    )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCESSO! Conversão finalizada.")

except Exception as e:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ERRO: {str(e)}")

spark.stop()