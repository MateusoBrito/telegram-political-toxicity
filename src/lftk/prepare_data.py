import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path

tqdm.pandas(desc="Limpando mensagens")

MESSAGES_PATH = Path("../data/processed/messages_preprocessed")
POLITICAL_PATH = Path("../data/processed/messages_classified")
OUTPUT_DIR = Path("../data/processed/messages_pre_lftk")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PK = ["id", "group_name"]
TEXT_COLUMN = "text"
COLUMN = "political_category"
CATEGORY = "Politic"

print("Carregando datasets...")
df_messages = pd.read_parquet(MESSAGES_PATH) 
df_political = pd.read_parquet(POLITICAL_PATH)

print(f"Filtrando mensagens em {POLITICAL_PATH}: {COLUMN} == {CATEGORY}")
df_political_keys = df_political.loc[df_political[COLUMN] == CATEGORY, PK]
print(f"Quantidade de mensagens do tipo {CATEGORY}: {len(df_political_keys)}")

print(f"Inner Join entre {POLITICAL_PATH} e {MESSAGES_PATH}")
df_filtered = pd.merge(df_messages, df_political_keys, on=PK, how="inner")

def clean_for_spacy(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs (que confundem o tokenizador)
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Substitui quebras de linha e tabulações por espaço
    text = re.sub(r'[\n\r\t]+', ' ', text)
    # Remove espaços múltiplos
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Aplicando limpeza nos textos...")
df_filtered[TEXT_COLUMN] = df_filtered[TEXT_COLUMN].progress_apply(clean_for_spacy)

df_filtered.to_parquet(
    OUTPUT_DIR, 
    partition_cols=['year', 'month'], 
    engine='pyarrow', 
    index=False
)

print(f"Sucesso! Dataset de {len(df_filtered)} linhas salvo e particionado em: {OUTPUT_DIR}")