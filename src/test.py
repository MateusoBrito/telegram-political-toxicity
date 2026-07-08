from src.utils.data_loader import get_spark_session, ROOT
from pyspark.sql.functions import col

# Caminhos dos diretórios (ajuste se os nomes das pastas forem diferentes no seu projeto)
INPUT_ORIGINAL_PATH = ROOT / "data" / "processed" / "messages_preprocessed"
INPUT_IDS_PATH = ROOT / "data" / "processed" / "stratified_sample_ids"
OUTPUT_FINAL_PATH = ROOT / "data" / "processed" / "stratified_sample_completo"

if __name__ == "__main__":
    print("Fase 2: Filtrando base original com base nos IDs amostrados...")
    print("="*50)
    
    # Inicia a sessão usando a mesma função configurada com 12GB
    spark = get_spark_session("extractFullMessages")
    
    # 1. Lê a lista de IDs gerada na Fase 1 (Super leve)
    print("Lendo os IDs da amostra...")
    df_ids = spark.read.parquet(str(INPUT_IDS_PATH))
    total = df_ids.count()
    print(total)
    
    # 2. Lê a base original completa (com os textos pesados)
    print("Lendo a base de dados original...")
    df_original = spark.read.parquet(str(INPUT_ORIGINAL_PATH))
    
    # 3. Aplica o LEFT_SEMI join usando a coluna de ID
    # IMPORTANTE: Se a coluna de ID se chamar algo diferente de 'id' (ex: 'msg_id'),
    # altere o nome na linha abaixo em on="..."
    print("Filtrando mensagens completas via Left-Semi Join...")
    df_final_completo = df_original.join(df_ids, on=["id","group_name"], how="left_semi")
    total = df_final_completo.count()
    print(total)
    
    # 4. Salva a amostra final com todas as colunas originais
    print(f"Salvando base final completa em: {OUTPUT_FINAL_PATH}")
    df_final_completo.write.mode("overwrite").parquet(str(OUTPUT_FINAL_PATH))
    
    print("="*50)
    print("Processo finalizado com sucesso!")
    print(f"Amostra final armazenada em: {OUTPUT_FINAL_PATH}")