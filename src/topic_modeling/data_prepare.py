from src.utils.data_loader import ROOT
from pathlib import Path
import pandas as pd

INPUT_PATH = ROOT / "data" / "processed" / "stratified_sample_full"

def load_stopwords(path: Path) -> set:
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def prepare_data(
    file_path=INPUT_PATH,
    columns: list = None,
    sample_frac: float = 1.0,
    random_state: int = 42
):
    try:
        df = pd.read_parquet(file_path, columns=columns)  # filtra colunas direto na leitura
        print(f"   -> Foram encontrados {len(df)} mensagens provenientes de {df['group_name'].nunique()} grupos.")

        if sample_frac < 1.0:
            df_sample = df.sample(frac=sample_frac, random_state=random_state)
            print(f"   -> Foram amostrados {len(df_sample)} mensagens provenientes de {df_sample['group_name'].nunique()} grupos.")
        else:
            df_sample = df
            print(f"   -> Usando 100% da base ({len(df_sample)} mensagens).")

        df_unique = df_sample.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
        documents_unique = df_unique["clean_text"].tolist()
        print(f"   -> Textos únicos para treinamento: {len(df_unique)}")
    except Exception as e:
        raise ValueError(f"Erro na leitura de {file_path}: {e}")

    return documents_unique, df_unique, df_sample
    

if __name__ == "__main__":
    documents_unique, df_unique, df  = prepare_data(INPUT_PATH)
    
    print("\nEsquema carregado automaticamente do Parquet:")
    print(df_unique)
    print(df_unique.columns)
    
    print(f"\nTotal de mensagens únicas carregadas: {len(documents_unique):,}")

    print(documents_unique[:10])

