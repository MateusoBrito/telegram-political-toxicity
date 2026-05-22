import os
import sys
import re
import glob
import math
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, input_file_name, regexp_extract, length
from pyspark.sql.types import StructType, StructField, StringType, LongType, BooleanType, IntegerType
from langdetect import detect, DetectorFactory

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# ============================== CONFIG ==============================
INPUT_DIR = "../../messages_july"
OUTPUT_DIR = "../../messages_july_en_parquet"
LOG_FILE = "../reports/language_filter_parquet.log"

# ============================== LOG ==============================

def log(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linha = f"[{t}] {msg}"
    print(linha)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(linha + "\n")

# ============================== SPARK ==============================
log("Iniciando ambiente Spark...")

spark = (
    SparkSession.builder
    .appName("telegram_language_filter_parquet")
    .config("spark.driver.memory", "12g")
    .config("spark.executor.memory", "12g")
    .config("spark.sql.shuffle.partitions", "200") 
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

log("Ambiente Spark distribuído preparado!")

# ============================== SCHEMA CORRIGIDO ==============================
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

# ============================== LANGUAGE FUNCTION ==============================
def detect_language(text):
    DetectorFactory.seed = 0
    
    if not text or not str(text).strip():
        return "unknown"
    try:
        return detect(str(text))
    except:
        return "unknown"

lang_udf = udf(detect_language, StringType())
# ============================== ARQUIVOS PROCESSADOS ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

canais_processados = set()
if os.path.exists(OUTPUT_DIR):
    for pasta in os.listdir(OUTPUT_DIR):
        if pasta.startswith("channel_id="):
            cid = pasta.split("=")[1]
            canais_processados.add(cid)

log(f"Encontrados {len(canais_processados)} canais já processados no destino.")

todos_arquivos = glob.glob(f"{INPUT_DIR}/*.tsv.gz")
arquivos_pendentes = []

for arquivo in todos_arquivos:
    match = re.search(r"channel_(\d+)-", arquivo)
    if match:
        cid = match.group(1)
        if cid not in canais_processados:
            arquivos_pendentes.append(arquivo)

log(f"Arquivos pendentes para processar: {len(arquivos_pendentes)}")

if len(arquivos_pendentes) == 0:
    log("Todos os arquivos já foram processados!")
    spark.stop()
    sys.exit(0)

BATCH_SIZE = 500
total_lotes = math.ceil(len(arquivos_pendentes) / BATCH_SIZE)

# ============================== PROCESSAMENTO EM LOTE ==============================

for i in range(0, len(arquivos_pendentes), BATCH_SIZE):
    lote_atual = arquivos_pendentes[i:i+BATCH_SIZE]
    num_lote = (i // BATCH_SIZE) + 1
    
    log(f"--- Iniciando Lote {num_lote}/{total_lotes} ({len(lote_atual)} arquivos) ---")

    try:
        df = (spark.read.option("header", True)
              .option("delimiter", "\t")
              .option("quote", '"')
              .option("multiline", True)
              .option("escape", '"')
              .option("mode", "DROPMALFORMED") 
              .schema(schema)
              .csv(lote_atual))

        # Criação da coluna de origem (faltava esta linha)
        df = df.withColumn("file_path", input_file_name())
        
        # Extração do ID
        df = df.withColumn("channel_id", regexp_extract(col("file_path"), r"channel_(\d+)-", 1))
        
        # Filtros
        df = df.filter(col("text").isNotNull() & (length(col("text")) >= 20))

        # Idioma
        df = df.withColumn("lang", lang_udf(col("text")))
        df_en = df.filter(col("lang") == "en")

        # Deduplicação (cria a variável df_final)
        df_final = df_en.dropDuplicates(["id", "channel_id"])

        (
            df_final.write
            .mode("append") 
            .partitionBy("channel_id")
            .parquet(OUTPUT_DIR)
        )

        log(f"Lote {num_lote} salvo com sucesso!")
    except Exception as e:
        log(f"ERRO no Lote {num_lote}. Detalhes: {str(e)}")

log("=" * 60)
log("Processamento incremental finalizado!")
spark.stop()