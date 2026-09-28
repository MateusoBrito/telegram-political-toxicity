import os
import re
import shutil
import gc
from pathlib import Path
from time import perf_counter
import math

import pandas as pd
import numpy as np
import spacy
import lftk
from tqdm import tqdm

from src.lftk.data_loader import load_and_clean_data

BASE_DIR = Path(__file__).resolve().parents[2]

FILE_PATH = BASE_DIR / "data/processed/ads_deputados_preprocessed.parquet"
OUTPUT_PATH = BASE_DIR / "data/processed/lftk_output.parquet"
OUTPUT_PATH_SPACY = BASE_DIR / "outputs/examples_spacy.txt"
TEMP_DIR = BASE_DIR / "data/processed/temp_partitions"

COLUMN = "text_clean"
BATCH_SIZE = 32   
CHUNK_SIZE = 1000  

def stream_lftk_features(texts: list, nlp: spacy.Language, features: list) -> pd.DataFrame:
    """Extrai features em formato de fluxo (stream) para economizar MUITA memória RAM."""
    results = []
    
    # O nlp.pipe atua como um gerador. O doc não fica salvo numa lista global.
    doc_generator = nlp.pipe(texts, batch_size=BATCH_SIZE, disable=["ner"])
    
    for doc in tqdm(doc_generator, total=len(texts), desc="spaCy + LFTK"):
        extractor = lftk.Extractor(docs=doc)
        extractor.customize(stop_words=True, punctuations=True)   
        feats = extractor.extract(features=features)
        results.append(feats)
        
    return pd.DataFrame(results)


if __name__ == "__main__": 
    df = load_and_clean_data(FILE_PATH, COLUMN) 

    if OUTPUT_PATH.exists():
        print(f"Arquivo existente encontrado: {OUTPUT_PATH}")

        df_done = pd.read_parquet(OUTPUT_PATH)
        processed_texts = set(df_done[COLUMN].astype(str))
        original_size = len(df)
        df = df[~df[COLUMN].astype(str).isin(processed_texts)].reset_index(drop=True)

        if len(df) == 0:
            print("Nenhum texto novo para processar.")
            exit()

        print(f"    -> Já processados: {len(processed_texts):,}")
        print(f"    -> Restantes:      {len(df):,}")
        print(f"    -> Ignorados:      {original_size - len(df):,}")
    else:
        df_done = None

    print("[2/5] Carregando modelo Transformer pt_core_news_trf...")
    try:
        nlp = spacy.load("pt_core_news_lg")
    except OSError:
        raise OSError("Modelo não encontrado. Instale as dependências.")
    
    all_features = lftk.search_features(return_format="list_key")
    print(f"    -> Total de features configuradas: {len(all_features)}")

    # 4. Processamento em Lotes (Chunks)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_files = []
    total_chunks = math.ceil(len(df) / CHUNK_SIZE)
    
    print(f"[4/5] Iniciando processamento dividido em {total_chunks} lotes...")

    for i in range(total_chunks):
        print(f"\n--- Processando Lote {i+1}/{total_chunks} ---")
        
        start_idx = i * CHUNK_SIZE
        chunk_df = df.iloc[start_idx : start_idx + CHUNK_SIZE].reset_index(drop=True)
        chunk_texts = chunk_df[COLUMN].tolist()
        
        start_time = perf_counter()
        df_feats = stream_lftk_features(chunk_texts, nlp, all_features)
        print(f"Lote finalizado em {perf_counter() - start_time:.2f}s")
        
        chunk_final = pd.concat([chunk_df, df_feats], axis=1)
        temp_file = TEMP_DIR / f"part_{i:04d}.parquet"
        chunk_final.to_parquet(temp_file)
        temp_files.append(temp_file)
        
        del chunk_df, chunk_texts, df_feats, chunk_final
        gc.collect() 

    print(f"\n[5/5] Consolidando partições e salvando resultado final...")
    """
    df_final = pd.concat([pd.read_parquet(f) for f in temp_files], ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_PATH)
    
    print(f"    -> Arquivo final salvo em: {OUTPUT_PATH}")
    print(f"    -> Shape final: {df_final.shape}")

    # Limpa pasta temporária
    shutil.rmtree(TEMP_DIR)
    print("Processo concluído com sucesso e arquivos temporários limpos.") 
    """
    print(f"\n[5/5] Consolidando partições e salvando resultado final...")

    df_new = pd.concat(
        [pd.read_parquet(f) for f in temp_files],
        ignore_index=True
    )

    if df_done is not None:
        df_final = pd.concat(
            [df_done, df_new],
            ignore_index=True
        )
    else:
        df_final = df_new

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_PATH)

    print(f"    -> Arquivo final salvo em: {OUTPUT_PATH}")
    print(f"    -> Shape final: {df_final.shape}")