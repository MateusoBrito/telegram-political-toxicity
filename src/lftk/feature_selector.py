import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
import lftk
import json
import os

class GiniFeatureSelector:
    def __init__(self, X: pd.DataFrame, 
                 y: pd.Series, 
                 n_thresholds: int = None, 
                 valid_labels: list = None):
        self.X = X.select_dtypes(include=[np.number]).reset_index(drop=True)
        if valid_labels is not None:
            mask = y.isin(valid_labels)
            self.X = X[mask].select_dtypes(include=[np.number]).reset_index(drop=True)
            self.y = y[mask].reset_index(drop=True)
        else:
            self.y = y.reset_index(drop=True)
        
        self.n_thresholds = n_thresholds
        self.ranking_df = None

    @staticmethod
    def _gini_impurity(y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)

    @staticmethod
    def _gini_gain(args):
        X_vals, y_vals, feature_name, feature_idx, threshold = args
        parent_gini = GiniFeatureSelector._gini_impurity(y_vals)
        col = X_vals[:, feature_idx]
        left_mask = col <= threshold
        y_left, y_right = y_vals[left_mask], y_vals[~left_mask]
        left_weight = len(y_left) / len(y_vals)
        right_weight = len(y_right) / len(y_vals)
        gini_after = (left_weight  * GiniFeatureSelector._gini_impurity(y_left) +
                      right_weight * GiniFeatureSelector._gini_impurity(y_right))
        return feature_name, threshold, parent_gini - gini_after

    def compute_ranking(self) -> pd.DataFrame:
        """Calcula o Gini Gain para todos os atributos e retorna o ranking completo."""
        tasks = []
        X_vals = self.X.values
        y_vals = self.y.values

        for feature_name in self.X.columns:
            idx = self.X.columns.get_loc(feature_name)
            unique_vals = np.unique(X_vals[:, idx])
            if self.n_thresholds and len(unique_vals) > self.n_thresholds:
                thresholds = np.percentile(unique_vals, np.linspace(0, 100, self.n_thresholds))
            else:
                thresholds = unique_vals
            for thr in thresholds:
                tasks.append((X_vals, y_vals, feature_name, idx, thr))

        #with Pool(cpu_count()) as pool:
        #    results = pool.map(self._gini_gain, tasks)
        
        with Pool(cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self._gini_gain, tasks),
                    total=len(tasks),
                    desc="Gini Gain"
                )
            )

        results.sort(key=lambda x: x[2], reverse=True)
        self.ranking_df = pd.DataFrame(results, columns=['attribute', 'split', 'gini_gain'])
        return self.ranking_df

    @staticmethod
    def _load_lftk_feature_map() -> dict:
        """
        Lê o feature_map.json embutido no pacote lftk instalado e retorna
        um dict {key: {'language': 'general' | 'en', 'name': ..., 'family': ..., ...}}.
 
        O LFTK só possui duas categorias em 'language':
          - 'general' : funciona com qualquer pipeline spaCy (qualquer idioma)
          - 'en'      : depende de lookup tables exclusivas do inglês
                        (Kuperman, Brysbaert, SubtlexUS) e portanto só é
                        válido quando o texto/pipeline é em inglês.
        """
        lftk_dir = os.path.dirname(lftk.__file__)
        path = os.path.join(lftk_dir, "resources", "feature_map.json")
 
        feature_map = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    feature_map[d["key"]] = d
        return feature_map
 
    def select_language_features(self, lang: str = "en") -> list:
        """
        Retorna a lista de colunas de self.X que são válidas para o idioma informado.
 
        Regra (LFTK só distingue 'general' vs 'en'):
          - lang == 'en'   -> todas as features são válidas (general + en),
                              pois o texto está em inglês.
          - lang != 'en'   -> apenas as features 'general' são válidas,
                              pois as 'en'-only (ex: a_kup_ps, t_subtlex_us_zipf,
                              cole, fkre, entidades nomeadas, etc.) dependem de
                              recursos linguísticos exclusivos do inglês e não
                              produzem valores confiáveis em outros idiomas
                              (ex: português, usando pt_core_news_lg).
 
        Parameters
        ----------
        lang : str
            Idioma do texto/pipeline spaCy utilizado para extrair as features
            do LFTK. Use 'en' para inglês; qualquer outro valor (ex: 'pt')
            é tratado como "não-inglês".
 
        Returns
        -------
        list[str]
            Lista de nomes de colunas de self.X consideradas válidas para o
            idioma informado. Colunas de self.X que não existem no
            feature_map.json do LFTK são mantidas por padrão (ver `unknown_policy`
            comment abaixo) -- ajuste se preferir excluí-las.
        """
        feature_map = self._load_lftk_feature_map()
 
        valid_features = []
        for col in self.X.columns:
            feat_info = feature_map.get(col)
 
            if feat_info is None:
                # Coluna não reconhecida pelo LFTK (ex: feature customizada,
                # como features de tópicos/embeddings). Mantemos por padrão.
                valid_features.append(col)
                continue
 
            feat_lang = feat_info["language"]
 
            if lang == "en":
                # Em inglês, tanto 'general' quanto 'en' são válidas
                valid_features.append(col)
            else:
                # Em qualquer outro idioma, só 'general' é válida
                if feat_lang == "general":
                    valid_features.append(col)
 
        return valid_features
    
    def filter_by_language(self, lang: str = "en") -> pd.DataFrame:
        """
        Conveniência: retorna self.X já filtrado, contendo apenas as colunas
        válidas para o idioma informado (ver select_language_features).
        """
        valid_cols = self.select_language_features(lang=lang)
        return self.X[valid_cols]

    # ------------------------------------------------------------------
    # Seleção com filtro de correlação de Spearman
    # ------------------------------------------------------------------
    def select_top(self, n: int = 10, corr_threshold: float = 0.7) -> pd.DataFrame:
        if self.ranking_df is None:
            raise RuntimeError("Rode compute_ranking() antes de select_top().")

        best_per_feature = (
            self.ranking_df
            .sort_values("gini_gain", ascending=False)
            .drop_duplicates(subset="attribute")
            .reset_index(drop=True)
        )

        # top N antes da correlação
        top_before = best_per_feature.head(n)["attribute"].tolist()
        print(f"\nTop {n} antes da correlação de Spearman:")
        for i, feat in enumerate(top_before, 1):
            print(f"   {i:2}. {feat}")

        corr_matrix = self.X.corr(method="spearman").abs()
        selected = []
        removed = set()

        for _, row in best_per_feature.iterrows():
            feature = row["attribute"]
            if feature in removed:
                continue
            selected.append(row)
            correlated = corr_matrix.index[corr_matrix[feature] > corr_threshold].tolist()
            removed.update(f for f in correlated if f != feature)
            if len(selected) == n:
                break

        top_after = [r["attribute"] for r in selected]
        print(f"\nTop {n} após correlação de Spearman (threshold={corr_threshold}):")
        for i, feat in enumerate(top_after, 1):
            marker = "  <-- novo" if feat not in top_before else ""
            print(f"   {i:2}. {feat}{marker}")

        replaced = [f for f in top_before if f not in top_after]
        added    = [f for f in top_after  if f not in top_before]
        if replaced:
            print(f"\nRemovidos por correlação: {replaced}")
            print(f"Substituídos por:         {added}")
        else:
            print("\nNenhuma substituição — top N idêntico antes e depois.")

        return pd.DataFrame(selected).reset_index(drop=True)


# ------------------------------------------------------------------
# Uso
# ------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    import argparse
    from pathlib import Path
    from src.lftk.data_loader import load__all_and_clean_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Caminho para o config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    BASE_DIR = Path(__file__).resolve().parents[2]

    # Paths
    paths       = cfg["paths"]
    base        = Path(BASE_DIR) / paths["base_dir"]
    #topics_path = Path(base) / paths["topics_file"]
    #lftk_path   = Path(base) / paths["lftk_file"]
    file_path = Path(base) / paths["file"]
    output_full  = paths["output_ranking"]
    output_top10 = paths["output_top10"]

    # Colunas
    col_text = cfg["columns"]["text"]
    col_target = cfg["columns"]["target"]
    col_pk = cfg["columns"]["primary_key"]
    num_topics = cfg["columns"]["num_topics"]
    cols_lftk = cfg["columns"]["lftk_columns"]

    # Parâmetros Gini
    n_thresholds = cfg["gini"]["n_thresholds"]
    top_n = cfg["gini"]["top_n"]
    lang = cfg['gini']['lang']
    corr_threshold = cfg["gini"]["corr_threshold"]

    # Pipeline
    #df_topics = load__all_and_clean_data(topics_path, col_text)
    #df_lftk   = load__all_and_clean_data(lftk_path, col_text)
    df = load__all_and_clean_data(file_path, col_text)
    #df = pd.merge(df_topics, df_lftk, on=col_pk, how="inner")
    print(f"Shape após merge: {df.shape}")

    df = df[df[col_target].isin(num_topics)]
    print(f"Shape após filtro de target: {df.shape}")

    X = df[cols_lftk]
    Y = df[col_target]
    selector = GiniFeatureSelector(X, Y, valid_labels=num_topics, n_thresholds=n_thresholds)
    #selector = GiniFeatureSelector(X, Y, valid_labels=num_topics)

    X = selector.filter_by_language(lang=lang)
    selector.X = X

    print("Calculando o gini gain de cada métrica...")
    selector.compute_ranking()

    print("Selecionando o top10 métricas mais representativas...")
    top10 = selector.select_top(n=top_n, corr_threshold=corr_threshold)
    print(top10[['attribute', 'split', 'gini_gain']])

    selector.ranking_df.to_csv(output_full, index=False)
    top10.to_csv(output_top10, index=False)