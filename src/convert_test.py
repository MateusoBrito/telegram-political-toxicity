"""
Este código converte os dados dos canais filtrados para .parquet de forma segura (em lotes) e tolerante a falhas.
"""

import pandas as pd
import os
import math

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import DataFrame

DIR = "/home/students/moliveira/"
BASE_DIR = DIR + "telegram_2024/extracted"
GROUPS_FILE = "../reports/group_to_keep.txt"
OUTPUT_DIR = DIR + "telegram-political-toxicity/data/raw"
ERROS_FILE = DIR + "telegram-political-toxicity/data/lotes_com_erro.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def peak_filter_chats(fp: str = GROUPS_FILE) -> list:
    print("1. Carregando lista de canais filtrados...")
    df_chats = pd.read_csv(fp, header=None, names=['channel'])
    canais_permitidos = set(df_chats['channel'].astype(str).tolist())
    print(f"   -> Canais a serem convertidos: {len(canais_permitidos)}")
    return list(canais_permitidos)

def load_schema(spark: SparkSession, file_paths: list):
    # SCHEMA SEGURO: IDs como LongType para não perder dados
    schema = StructType([
        StructField("id", LongType(), False),    
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
    ])

    df = (spark.read.option("header", True)
      .option("delimiter", "\t")
      .option("quote", '"')
      .option("multiline", True)
      .option("escape", '"')
      .schema(schema)
      .csv(file_paths)
      .withColumn("file_path", input_file_name()))

    df = df.withColumn("group_name", regexp_extract(col("file_path"), r"/extracted/([^/]+)/", 1))

    return df

def save_in_parquet(df: DataFrame, modo: str):
    (
        df.write
        .mode(modo)
        .parquet(OUTPUT_DIR)
    )

if __name__ == '__main__':
    chats = peak_filter_chats()
    fp_chats = [BASE_DIR + "/" + chat + "/*.tsv.gz" for chat in chats]

    print("2. Preparando ambiente Spark...")
    spark = (
        SparkSession.builder
        .appName("convert_to_parquet_batch")
        .config("spark.driver.memory", "12g")    # Reduzido para garantir que a JVM ligue
        .config("spark.executor.memory", "12g")  # Reduzido
        .config("spark.sql.files.ignoreCorruptFiles", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print("   -> Spark pronto!")

    CHUNK_SIZE = 500 
    total_lotes = math.ceil(len(fp_chats) / CHUNK_SIZE)

    for i in range(total_lotes):
        idx_start = i * CHUNK_SIZE
        idx_finish = idx_start + CHUNK_SIZE
        lote_arquivos = fp_chats[idx_start:idx_finish]
        
        print(f"\n--- Processando Lote {i+1} de {total_lotes} ({len(lote_arquivos)} canais) ---")

        try:
            df = load_schema(spark, lote_arquivos)
            modo_gravacao = "overwrite" if i == 0 else "append"
            save_in_parquet(df, modo_gravacao)
            print(f"   -> Lote {i+1} salvo com sucesso!")
            
        except Exception as e:
            erro_msg = f"Erro no lote {i+1}: {str(e)}"
            print(erro_msg)
            with open(ERROS_FILE, "a", encoding="utf-8") as f:
                f.write(erro_msg + "\n")
                
    print("\nProcessamento em lotes finalizado!")
    spark.stop()