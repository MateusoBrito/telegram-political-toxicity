import re
import ast
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.data_loader import ROOT

INPUT_DIR = ROOT / "reports" / "topic_modeling" / "grid_search"

LOG_PATH_1 = ROOT / "outputs" / "grid_log.txt"
LOG_PATH_2 = ROOT / "outputs" / "grid_log2.txt"

OUTPUT_FILE = ROOT / "configs" / "best_config.yaml"

def parse_grid_log(log_path: Path) -> pd.DataFrame:
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
 
    rows = []
    current_model = None
    pending = None
 
    combo_re = re.compile(r"^\[(\d+)/(\d+)\]\s+(\{.*\})\s*$")
    model_re = re.compile(r"^\[(\d+)/(\d+)\]\s+Modelo de embedding:\s+(.*)$")
    outliers_re = re.compile(r"Outliers \(ruído\):\s*(\d+)\s*\(([\d.]+)%\)")
    silhouette_re = re.compile(r"Silhoueta:\s*([-\d.]+|nan)")
    diversity_re = re.compile(r"Diversidade:\s*([-\d.]+|nan)")
    coherence_re = re.compile(r"Coerência \(c_v\):\s*([-\d.]+|nan)")
    falhou_re = re.compile(r"Falhou:\s*(.*)$")
 
    for line in lines:
        stripped = line.strip()
 
        m_model = model_re.match(stripped)
        if m_model:
            current_model = m_model.group(3).strip()
            continue
 
        m_combo = combo_re.match(stripped)
        if m_combo:
            if pending is not None:
                rows.append(pending)
            idx = int(m_combo.group(1))
            params_str = m_combo.group(3)
            try:
                params = ast.literal_eval(params_str)
            except Exception:
                params = {}
            
            pending = {"embedding_model": current_model, "combo_idx": idx, **params}
            continue
 
        if pending is not None:
            m_out = outliers_re.search(line)
            if m_out:
                pending["n_outliers"] = int(m_out.group(1))
                pending["outliers_pct"] = float(m_out.group(2))
                continue
            m_sil = silhouette_re.search(line)
            if m_sil:
                val = m_sil.group(1)
                pending["silhouette"] = float(val) if val != "nan" else float("nan")
                continue
            m_div = diversity_re.search(line)
            if m_div:
                val = m_div.group(1)
                pending["diversity"] = float(val) if val != "nan" else float("nan")
                continue
            m_coh = coherence_re.search(line)
            if m_coh:
                val = m_coh.group(1)
                pending["coherence_cv"] = float(val) if val != "nan" else float("nan")
                continue
            m_fail = falhou_re.search(line)
            if m_fail:
                pending["error"] = m_fail.group(1).strip()
                continue
 
    if pending is not None:
        rows.append(pending)
 
    df = pd.DataFrame(rows)
 
    if "outliers_pct" in df.columns:
        incompletas = df["outliers_pct"].isna() & (
            df["error"].isna() if "error" in df.columns else True
        )
        if incompletas.any():
            print(f"Aviso: {incompletas.sum()} combinação(ões) incompleta(s) — removidas.")
            df = df[~incompletas].reset_index(drop=True)
 
    return df

def add_normalized_scores(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Normaliza silhouette, diversity e coherence_cv (0-1) e calcula scores combinados."""
    df = df.copy()
    for col in ["silhouette", "diversity", "coherence_cv"]:
        min_v, max_v = df[col].min(), df[col].max()
        if max_v > min_v:
            df[f"{col}_norm{suffix}"] = (df[col] - min_v) / (max_v - min_v)
        else:
            df[f"{col}_norm{suffix}"] = 0.0

    df[f"score_sdc{suffix}"] = df[[f"silhouette_norm{suffix}", f"diversity_norm{suffix}", f"coherence_cv_norm{suffix}"]].mean(axis=1)
    df[f"score_sc{suffix}"] = df[[f"silhouette_norm{suffix}", f"coherence_cv_norm{suffix}"]].mean(axis=1)
    return df

def save_best_config(df: pd.DataFrame, metric_col: str, output_name: str, multiplier: int = 10):
    """
    Encontra a melhor linha para a métrica informada, multiplica o min_cluster_size
    e exporta um arquivo YAML pronto para rodar.
    """
    # 1. Ordena o DataFrame para pegar o maior valor da métrica desejada
    best_row = df.sort_values(by=metric_col, ascending=False).iloc[0]
    
    # 2. Como as colunas de merge viraram string, precisamos converter de volta
    n_neighbors = int(float(best_row["n_neighbors"]))
    n_components = int(float(best_row["n_components"]))
    min_dist = float(best_row["min_dist"])
    min_cluster_size = int(float(best_row["min_cluster_size"]))
    min_samples = int(float(best_row["min_samples"]))
    
    # 3. Multiplica o tamanho mínimo do cluster pelo fator de escala da base
    new_min_cluster_size = min_cluster_size * multiplier

    # 4. Estrutura o novo YAML
    config = {
        "model": {
            "embedding_names": [best_row["embedding_model"]]
        },
        "paths": {
            "input_path": "data/processed/stratified_sample_full",
            "stopwords": "data/processed/stopwords.txt",
            "cache_embeddings_dir": "data/cache/embeddings_grid",
            "dir_reports": f"reports/topic_modeling/best_{metric_col}"
        },
        "sampling": {
            "frac": 1.0
        },
        "grid_search": {
            "n_neighbors": [n_neighbors],
            "n_components": [n_components],
            "min_dist": [min_dist],
            "metric": [best_row["metric"]],
            "min_cluster_size": [new_min_cluster_size],
            "min_samples": [min_samples],
            "hdbscan_metric": [best_row["hdbscan_metric"]],
            "cluster_selection_method": [best_row["cluster_selection_method"]],
            "prediction_data": [True]
        }
    }

    # 5. Salva no disco
    out_path = ROOT / "configs" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        
    print(f"Salvo: {output_name}")
    print(f"  -> Métrica ({metric_col}): {best_row[metric_col]:.4f}")
    print(f"  -> min_cluster_size ajustado: {min_cluster_size} -> {new_min_cluster_size}")
    print(f"  -> Relatórios irão para: {config['paths']['dir_reports']}\n")
    
if __name__ == "__main__":

    dfs = []
    for model_dir in INPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        csv_path = model_dir / "grid_search_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["embedding_model"] = model_dir.name
        dfs.append(df)
    
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total de linhas carregadas: {len(df_all)} de {df_all['embedding_model'].nunique()} modelos")
    
    df_log_1 = parse_grid_log(LOG_PATH_1)
    df_log_2 = parse_grid_log(LOG_PATH_2)
    df_log = pd.concat([df_log_1, df_log_2], ignore_index=True)
    
    print(df_log["embedding_model"].value_counts())
    
    # 1. Defina quais colunas de hiperparâmetros serão usadas para bater os dados do CSV com o Log
    param_cols = [
        "n_neighbors", "n_components", "min_dist", "metric",
        "min_cluster_size", "min_samples", "hdbscan_metric",
        "cluster_selection_method", "prediction_data"
    ]
    
    # Filtra colunas que de fato existem nos dois lados
    join_cols = ["embedding_model"] + [c for c in param_cols if c in df_all.columns and c in df_log.columns]
    
    # 2. Conversão de tipos para garantir igualdade perfeita na hora de mesclar
    for col in join_cols:
        if col in df_all.columns and col in df_log.columns:
            # Força o mesmo tipo nos dois DataFrames (removendo diferenças entre float/int/str)
            df_all[col] = df_all[col].astype(str).str.strip()
            df_log[col] = df_log[col].astype(str).str.strip()
    
    # 3. Realiza o merge trazendo apenas as novas colunas que você quer recuperar do Log
    df_all = df_all.merge(
        df_log[join_cols + ["outliers_pct", "n_outliers"]],
        on=join_cols,
        how="left"
    )
    
    print(f"Total de linhas final: {len(df_all)}")

    df_all = add_normalized_scores(df_all, suffix="_global")
    
    print("\n" + "="*50)
    print("GERANDO AS MELHORES CONFIGURAÇÕES (MULTIPLICADOR x10)")
    print("="*50)

    # Gera os 4 arquivos baseados nas colunas corretas do seu DataFrame
    
    # 1. SCORE GLOBAL (Média combinada das três normalizadas)
    save_best_config(df_all, "score_sdc_global", "config_best_score_global.yaml", multiplier=10)
    
    # 2. SILHOUETTE GLOBAL
    save_best_config(df_all, "silhouette", "config_best_silhouette.yaml", multiplier=10)
    
    # 3. COHERENCE GLOBAL
    save_best_config(df_all, "coherence_cv", "config_best_coherence.yaml", multiplier=10)
    
    # 4. DIVERSITY GLOBAL
    save_best_config(df_all, "diversity", "config_best_diversity.yaml", multiplier=10)