import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from scipy.stats import ttest_rel
import seaborn as sns

class ResultsComparator:
    def __init__(self, base_dir='results'):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"Diretório de resultados não encontrado: {self.base_dir}")
        
    def get_model_data(self, dataset_name, representation, model_name=None):
        # AJUSTE: Se model_name for None ou vazio, busca direto na pasta da representação
        # Isso permite ler results/dataset/bert/experiment_log.csv
        if model_name:
            file_path = os.path.join(self.base_dir, representation, model_name, 'experiment_log.csv')
        else:
            file_path = os.path.join(self.base_dir, representation, 'experiment_log.csv')
        
        if not os.path.exists(file_path):
            return None

        try:
            df = pd.read_csv(file_path)
        except Exception:
            return None
        
        # Filtra pelo nome do dataset (ex: 'mpqa' ou 'SMSSpamCollection.csv')
        subset = df[df['dataset'] == dataset_name]
        if subset.empty: return None

        subset = subset.sort_values('fold')
        
        # Tenta pegar f1_weighted, se não tiver vai de macro
        f1_col = 'f1_weighted' if 'f1_weighted' in subset.columns else 'f1_macro'

        return {
            'accuracy': subset['accuracy'].values,
            'f1': subset[f1_col].values
        }

    def plot_combined_bars(self, results_dict, title_suffix='', output_path=None):
        """
        Gera um gráfico de barras agrupadas (Accuracy e F1 lado a lado).
        """
        methods = list(results_dict.keys())
        metrics = ['accuracy', 'f1']
        
        # Calcula médias e desvios
        means = {m: [np.mean(results_dict[method][m]) for method in methods] for m in metrics}
        stds = {m: [np.std(results_dict[method][m]) for method in methods] for m in metrics}

        x = np.arange(len(methods))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        # Barra 1: Accuracy (à esquerda)
        rects1 = ax.bar(x - width/2, means['accuracy'], width, yerr=stds['accuracy'], 
                        capsize=5, label='Accuracy', alpha=0.8, color='#4e79a7')
        
        # Barra 2: F1 (à direita)
        rects2 = ax.bar(x + width/2, means['f1'], width, yerr=stds['f1'], 
                        capsize=5, label='F1-Score', alpha=0.8, color='#f28e2b')

        ax.set_ylabel('Scores')
        ax.set_title(f'Performance: {title_suffix}')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.set_ylim(0, 1.1) # Margem para ver o topo
        ax.legend(loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Adiciona labels no topo das barras
        ax.bar_label(rects1, fmt='%.2f', padding=3, fontsize=9)
        ax.bar_label(rects2, fmt='%.2f', padding=3, fontsize=9)

        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    def plot_pairwise_differences(self, results_dict, metric='accuracy', title_suffix='', output_path=None):
        """
        Gera um gráfico de barras horizontal mostrando a diferença de performance
        entre os pares. Colore de VERDE se for significativo e CINZA se for empate.
        """
        methods = list(results_dict.keys())
        n = len(methods)
        if n < 2: return

        pairs = list(itertools.combinations(methods, 2))
        
        # Configuração Bonferroni
        num_comparisons = len(pairs)
        alpha_corr = 0.05 / num_comparisons
        
        # Listas para o plot
        labels = []
        diffs = []
        colors = []
        p_values = []
        
        for m1, m2 in pairs:
            scores1 = results_dict[m1][metric]
            scores2 = results_dict[m2][metric]
            
            # Diferença média (Positivo = m1 ganhou, Negativo = m2 ganhou)
            diff = np.mean(scores1) - np.mean(scores2)
            
            # Teste T
            t_stat, p_val = ttest_rel(scores1, scores2)
            is_sig = p_val < alpha_corr
            
            labels.append(f"{m1.upper()} vs {m2.upper()}")
            diffs.append(diff)
            p_values.append(p_val)
            
            # Cor: Verde (Significativo) ou Cinza (Não Significativo)
            # Se diff for negativo (m2 ganhou), podemos usar vermelho ou manter verde
            if is_sig:
                colors.append('#2ca02c') # Verde (Significativo)
            else:
                colors.append('#bdbdbd') # Cinza (Empate)

        # --- PLOTAGEM ---
        plt.figure(figsize=(8, len(pairs) * 1.5 + 1)) # Altura dinâmica
        
        y_pos = np.arange(len(labels))
        bars = plt.barh(y_pos, diffs, color=colors, alpha=0.8, edgecolor='black')
        
        # Linha vertical no zero
        plt.axvline(0, color='black', linewidth=1, linestyle='--')
        
        plt.yticks(y_pos, labels, fontsize=11)
        plt.xlabel(f"Diferença média de {metric.capitalize()}", fontsize=10)
        plt.title(f"Comparação Pareada - {title_suffix}\n(Verde = Diferença Real, Cinza = Empate)", fontsize=12)
        
        # Adiciona textos nas barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            p_text = "*" if p_values[i] < alpha_corr else "ns"
            val_text = f"{width:+.4f} ({p_text})"
            
            # Posiciona o texto um pouco ao lado da barra
            offset = 0.0005 if width >= 0 else -0.0005
            ha = 'left' if width >= 0 else 'right'
            
            plt.text(width + offset, bar.get_y() + bar.get_height()/2, 
                     val_text, va='center', ha=ha, fontsize=10, fontweight='bold')

        # Ajusta limites para caber o texto
        max_limit = max(abs(min(diffs)), abs(max(diffs))) * 1.3
        plt.xlim(-max_limit, max_limit)
        
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            plt.close()

    def save_statistical_report(self, results_dict, title_suffix='', output_path=None):
        """
        Salva um relatório de texto (.txt) com os detalhes do Teste T.
        """
        if not output_path: return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        methods = list(results_dict.keys())
        pairs = list(itertools.combinations(methods, 2))
        num_comparisons = len(pairs)
        alpha = 0.05
        alpha_corr = alpha / num_comparisons
        
        with open(output_path, 'w') as f:
            f.write(f"=== RELATÓRIO ESTATÍSTICO: {title_suffix} ===\n")
            f.write(f"Alpha Original: {alpha}\n")
            f.write(f"Correção Bonferroni: {alpha_corr:.6f} (para {num_comparisons} comparações)\n\n")
            
            for metric in ['accuracy', 'f1']:
                f.write(f"--- Métrica: {metric.upper()} ---\n")
                
                for m1, m2 in pairs:
                    scores1 = results_dict[m1][metric]
                    scores2 = results_dict[m2][metric]
                    
                    if len(scores1) != len(scores2):
                        f.write(f"[ERRO] Tamanhos diferentes: {m1} vs {m2}\n")
                        continue

                    t_stat, p_val = ttest_rel(scores1, scores2)
                    is_sig = p_val < alpha_corr
                    
                    result_txt = "SIGNIFICATIVO" if is_sig else "Não Significativo"
                    
                    f.write(f"{m1.upper()} vs {m2.upper()}:\n")
                    f.write(f"   Média {m1.upper()}: {np.mean(scores1):.4f}\n")
                    f.write(f"   Média {m2.upper()}: {np.mean(scores2):.4f}\n")
                    f.write(f"   P-Value: {p_val:.4e} -> {result_txt}\n\n")
            
            f.write("Fim do relatório.\n")
        
        print(f"Relatório salvo em: {output_path}")