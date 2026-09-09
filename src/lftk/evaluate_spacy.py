import re
from pathlib import Path
from collections import Counter
from time import perf_counter
import random

import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm

from src.lftk.data_loader import load_and_clean_data

# ==========================================
# CONFIGURAÇÕES
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

FILE_PATH = BASE_DIR / "data/processed/ads_deputados_preprocessed.parquet"
CLEAN_OUTPUT_PATH = BASE_DIR / "data/processed/ads_deputados_nlp_ready.parquet"
REPORT_OUTPUT_PATH = BASE_DIR / "outputs/spacy_quality_report.txt"
EXAMPLES_OUTPUT_PATH = BASE_DIR / "outputs/spacy_examples.txt"

COLUMN = "text_clean"
BATCH_SIZE = 64

EVALUATE_FULL_CORPUS = True  


# ==========================================
# RELATORIO
# ==========================================

def generate_spacy_report(texts: list, nlp: spacy.Language, output_path: Path, full_corpus: bool):
    """Gera um relatório de qualidade do texto usando spaCy."""
    print(f"\n[3/3] Gerando relatório spaCy em: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not full_corpus and len(texts) > 10000:
        print("    -> Modo Amostra: Selecionando 10.000 textos aleatórios para o relatório...")
        import random
        # Garante a reprodutibilidade da amostra
        random.seed(42) 
        texts_to_process = random.sample(texts, 10000)
    else:
        print(f"    -> Modo Full Corpus: Processando todos os {len(texts):,} textos...")
        texts_to_process = texts

    total_tokens, propn_count, verb_count, noun_count, bad_tokens = 0, 0, 0, 0, 0
    pos_counter = Counter()
    lemma_examples = {}
    lengths = []

    start_time = perf_counter()
    
    # Processamento em lote para agilizar
    doc_generator = nlp.pipe(texts_to_process, batch_size=BATCH_SIZE, disable=["ner"])
    
    for doc in tqdm(doc_generator, total=len(texts_to_process), desc="Avaliando Textos"):
        lengths.append(len(doc))
        for token in doc:
            total_tokens += 1
            pos_counter[token.pos_] += 1
            if token.pos_ == "PROPN": propn_count += 1
            elif token.pos_ == "VERB": verb_count += 1
            elif token.pos_ == "NOUN": noun_count += 1
            elif token.pos_ in ["X", "SPACE"]: bad_tokens += 1
            
            # Coleta exemplos de lematização vazia ou problemática
            if token.lemma_ in ["-PRON-", ""] and len(lemma_examples) < 40:
                lemma_examples[token.text] = token.lemma_

    # Salva o txt
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"RELATÓRIO SPACY ({'Corpus Completo' if full_corpus else 'Amostra de 10k'})\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total de documentos avaliados: {len(texts_to_process)}\n")
        f.write(f"Total de tokens: {total_tokens}\n")
        f.write(f"Média de tokens por doc: {np.mean(lengths):.2f}\n\n")

        f.write("Distribuição de POS (Classes Gramaticais):\n")
        for k, v in pos_counter.most_common():
            f.write(f"  {k:<10}: {v}\n")

        if total_tokens > 0:
            f.write("\nProporções importantes:\n")
            f.write(f"  VERB   : {verb_count / total_tokens:.2%}\n")
            f.write(f"  NOUN   : {noun_count / total_tokens:.2%}\n")
            f.write(f"  PROPN  : {propn_count / total_tokens:.2%}\n")
            f.write(f"  RUÍDO (X/SPACE) : {bad_tokens / total_tokens:.2%}\n")

    print(f"    -> Relatório salvo com sucesso em {perf_counter() - start_time:.2f} segundos.")


def save_examples_spacy(docs, output_path: Path):
    """
    Salva exemplos processados pelo spaCy com formatação tabular.
    (O sample de N exemplos já é feito antes de passar os docs para cá)
    """
    print(f"\n[3/4] Salvando exemplos práticos em: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write("\n" + "="*80 + "\n")
            f.write("TEXTO ORIGINAL:\n")
            f.write(doc.text + "\n")
            f.write("="*80 + "\n")

            f.write(f"\n{'IDX':<4} | {'TOKEN':<20} | {'POS':<10} | {'LEMMA':<20}\n")
            f.write("-"*80 + "\n")

            for i, token in enumerate(doc):
                f.write(
                    f"{i:<4} | "
                    f"{token.text:<20} | "
                    f"{token.pos_:<10} | "
                    f"{token.lemma_:<20}\n"
                )
            f.write("\n\n")

# ==========================================
# EXECUÇÃO
# ==========================================
if __name__ == "__main__": 
    # 1. Carrega e limpa os dados
    df_clean = load_and_clean_data(FILE_PATH, COLUMN)
    
    # 3. Carrega o modelo de NLP
    print("\nCarregando modelo pt_core_news_lg para avaliação...")
    try:
        nlp = spacy.load("pt_core_news_lg")
    except OSError:
        raise OSError("Modelo não encontrado. Instale as dependências.\n")
    
    # 4. Avalia a qualidade
    texts_list = df_clean[COLUMN].tolist()
    generate_spacy_report(texts_list, nlp, REPORT_OUTPUT_PATH, EVALUATE_FULL_CORPUS)

    random_texts = random.sample(texts_list, min(5, len(texts_list)))
    sample_docs = list(nlp.pipe(random_texts, disable=["ner"]))
    
    save_examples_spacy(sample_docs, EXAMPLES_OUTPUT_PATH)
    
    print("\nProcesso de preparação concluído. Agora você pode rodar o script do LFTK no arquivo limpo.")