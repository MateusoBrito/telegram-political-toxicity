from pathlib import Path
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, BooleanType, IntegerType

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "raw"

def get_spark_session(app_name="TelegramAnalysis"):
    """
    Inicia uma sessão Spark padronizada com memória otimizada e Arrow ativado
    para análises e UDFs rápidas.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "12g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_data(
    spark: SparkSession, 
    file_path=INPUT_PATH,
    columns: list = None
):
    """
    Carrega toda a base de dados Parquet instantaneamente de forma global.
    """
    print(f"Lendo base de dados de: {file_path}")
    
    if isinstance(file_path, list):
        caminhos_texto = [str(p) for p in file_path]
    else:
        caminhos_texto = str(file_path)
        
    df = spark.read.parquet(caminhos_texto)

    if columns is not None:
        df = df.select(columns)

    return df

if __name__ == "__main__":
    spark = get_spark_session()
    df_mensagens = load_data(spark)
    
    print("\nEsquema carregado automaticamente do Parquet:")
    df_mensagens.printSchema()
    
    print(f"\nTotal de mensagens carregadas: {df_mensagens.count():,}")