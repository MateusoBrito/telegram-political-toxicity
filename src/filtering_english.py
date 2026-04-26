import glob
import os
import math
from tqdm import tqdm
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# ============================== CONFIG ==============================
INPUT_DIR = "/home/reis/ic/eleicoes/eleicoes_final/messages_july"
OUTPUT_DIR = "/home/reis/ic/eleicoes/eleicoes_final/messages_july_en"
LOG_FILE = "/home/reis/ic/eleicoes/eleicoes_final/language_filter.log"

BATCH_SIZE = 20   # processa 20 arquivos por vez

# ============================== LOG ==============================


def log(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = f"[{t}] {msg}"
    print(linha)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(linha + "\n")

# ============================== SPARK ==============================

spark = SparkSession.builder \
    .appName("telegram_language_filter") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

log("Ambiente Spark preparado!")

# ============================== SCHEMA ==============================

schema = StructType([
    StructField("id", IntegerType(), False),    
    StructField("user_id", IntegerType(), True),
    StructField("text", StringType(), True),
    StructField("timestamp", LongType(), True),                  
    StructField("bot_flag", BooleanType(), True),                
    StructField("via_bot_id", BooleanType(), True),             
    StructField("via_business_bot_id", BooleanType(), True),    
    StructField("reply_to_msg_id", IntegerType(), True),         
    StructField("fwd_flag", BooleanType(), True),               
    StructField("fwd_from_id", IntegerType(), True),             
    StructField("media_type", StringType(), True),               
    StructField("views", IntegerType(), True),                   
    StructField("forwards", IntegerType(), True),                
    StructField("replies", IntegerType(), True),                 
    StructField("reactions", IntegerType(), True),       
    StructField("reaction_json", StringType(), True),   
    ])

# ============================== LANGUAGE FUNCTION ==============================

def detect_language(text):
    if text is None:
        return "unknown"

    text = str(text).strip()

    if text == "":
        return "unknown"

    try:
        return detect(text)
    except:
        return "unknown"

lang_udf = udf(detect_language, StringType())

# ============================== FILES ==============================

file_paths = sorted(glob.glob(f"{INPUT_DIR}/*.tsv.gz"))

total_files = len(file_paths)
total_batches = math.ceil(total_files / BATCH_SIZE) #Retorna o maior número inteiro ex: 7.001 -> 8

log(f"Arquivos encontrados: {total_files}")
log(f"Lotes totais: {total_batches}")

# cria pasta output se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================== CONTAGENS ==============================

total_mensagens = 0
total_mensagens_filtradas = 0
total_mensagens_ingles = 0
total_erro_lotes = 0

# ============================== PROCESSAMENTO EM LOTES ==============================

for batch_num in tqdm(range(total_batches), desc="Processando Canais"):

    # Calcula qual os canais que serao exeucatos no lote
    ini = batch_num * BATCH_SIZE
    fim = ini+BATCH_SIZE

    batch_files = file_paths[ini:fim]

    try:
        # ============================== LOAD ==============================
        # Load all files into a single DataFrame, adding a column for the file path
        df = (spark.read.option("header", True)  # Read with header
            .option("delimiter", "\t")         # Set tab delimiter
            .option("quote", '"')              # Set quote character
            .option("multiline", True)         # Allow multiline fields
            .option("escape", '"')             # Disable escape
            .schema(schema)
            .csv(batch_files)                  # Read CSV files
            .withColumn("file_path", input_file_name()))  # Add file path column
        
        # Extract the folder from the path
        df = df.withColumn(
            "channel_id",
            regexp_extract(col("file_path"), r"channel_(\d+)-", 1)
        )

       # ============================== FILTROS ==============================
        total_lote = df.count()
        total_mensagens += total_lote
        df = df.filter(col("text").isNotNull())
        df = df.filter(length(col("text")) >= 20)
        filtrado_lote = df.count()
        total_mensagens_filtradas += filtrado_lote

        # ============================== LANGUAGE ==============================
        df = df.withColumn("lang", lang_udf(col("text")))
        df_en = df.filter(col("lang") == "en")

        ingles_lote = df_en.count()
        total_mensagens_ingles += ingles_lote

        # ============================== SAVE ==============================

        (
            df_en.write
            .mode("append")   
            .partitionBy("channel_id")
            .option("header", True)
            .option("delimiter", "\t")
            .option("compression", "gzip")
            .csv(OUTPUT_DIR)
        )

        log(f"Lote {batch_num+1} salvo com sucesso.")
    except Exception as e:
        log(f"ERRO no lote {batch_num+1}: {str(e)}")
        total_erro_lotes+=1

log(f"Total mensagens lidas: {total_mensagens}")
log(f"Total após filtros: {total_mensagens_filtradas}")
log(f"Total em inglês: {total_mensagens_ingles}")
log(f"Total lotes com erro: {total_erro_lotes}")

log("=" * 60)
log("Processo finalizado!")
spark.stop()