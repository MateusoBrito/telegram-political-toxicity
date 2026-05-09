import pandas as pd
import glob
import os
import json

def carregar_dados_canal(caminho_canal):
    """
    Lê todos os arquivos mensais de um único canal e os consolida.
    """
    arquivos_mensais = glob.glob(os.path.join(caminho_canal, "*.tsv.gz"))
    lista_df = []
    
    # Extrai o ID do canal a partir do nome da pasta
    canal_id = os.path.basename(caminho_canal)
    
    for arquivo in sorted(arquivos_mensais):
        try:
            # O pandas identifica automaticamente a compressão .gz
            df_mes = pd.read_csv(arquivo, sep='\t', compression='gzip')
            lista_df.append(df_mes)
        except Exception as e:
            print(f"Erro ao ler {arquivo}: {e}")
            
    if not lista_df:
        return pd.DataFrame()
        
    df_completo = pd.concat(lista_df, ignore_index=True)
    
    # Dicas do README: remover duplicatas pelo ID da mensagem e ordenar
    df_completo = df_completo.drop_duplicates(subset=['id'])
    df_completo = df_completo.sort_values(by='timestamp')
    
    # Adiciona o ID do canal para referência global
    df_completo['channel_folder_id'] = canal_id
    
    return df_completo

def analisar_reacoes(reaction_json_str):
    """Exemplo de como processar a coluna reaction_json para análise de sentimento."""
    if pd.isna(reaction_json_str) or reaction_json_str == "None":
        return {}
    try:
        return json.loads(reaction_json_str)
    except:
        return {}

# Exemplo de execução para os 5 primeiros canais listados no seu terminal
diretorio_raiz = "../../telegram_2024/extracted/"
canais = [os.path.join(diretorio_raiz, d) for d in os.listdir(diretorio_raiz) if os.path.isdir(os.path.join(diretorio_raiz, d))]

# Exemplo processando apenas o primeiro canal da sua lista
df_canal_exemplo = carregar_dados_canal(canais[0])
print(df_canal_exemplo.head())