import os
import glob
import gzip
import csv
import multiprocessing
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def processar_arquivo_diario(caminho_arquivo):
    import gzip
    import csv
    import os
    from datetime import datetime
    from collections import Counter
    
    contagem_diaria = Counter()
    ids_unicos = set()
    erros_linha = 0
    
    try:
        with gzip.open(caminho_arquivo, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            next(reader, None)
            
            for row in reader:
                if row:
                    try:
                        ts = int(row[3])
                        if ts > 10**11: 
                            dt_object = datetime.fromtimestamp(ts / 1000.0)
                        else:
                            dt_object = datetime.fromtimestamp(ts)
                        
                        msg_id = row[0]
                        if msg_id not in ids_unicos:
                            ids_unicos.add(msg_id)
                            dia = dt_object.strftime('%Y-%m-%d')
                            contagem_diaria[dia] += 1
                    except (ValueError, TypeError):
                        erros_linha += 1
            print(f"No arquivo {caminho_arquivo}, {erros_linhas} deu erro la contagem.")
        return contagem_diaria
    except Exception:
        return contagem_diaria

if __name__ == "__main__":
    base_path = "/home/students/moliveira/telegram_2024/extracted/"
    
    # 1. Mapeamento de arquivos
    arquivos_totais = glob.glob(os.path.join(base_path, "*", "20*.tsv.gz"))
    chats = pd.read_csv("../reports/group_to_keep.txt", header=None, names=['channel'])
    canais_permitidos = set(chats['channel'].astype(str).tolist())
    
    arquivos_filtrados = [
        arq for arq in arquivos_totais 
        if os.path.basename(os.path.dirname(arq)) in canais_permitidos
    ]

    # 2. Processamento Paralelo
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        print("Processando Total Diário...")
        res_total = pool.map(processar_arquivo_diario, arquivos_totais)
        
        print("Processando Filtrado Diário...")
        res_filtrado = pool.map(processar_arquivo_diario, arquivos_filtrados)

    # 3. Agregação dos Counters
    final_total = Counter()
    for c in res_total:
        final_total.update(c)
        
    final_filtrado = Counter()
    for c in res_filtrado:
        final_filtrado.update(c)

    # 4. Preparação dos dados para JSON e Gráfico
    # Criamos um DataFrame para organizar as datas
    df_total = pd.DataFrame.from_dict(final_total, orient='index', columns=['total'])
    df_filtrado = pd.DataFrame.from_dict(final_filtrado, orient='index', columns=['filtrado'])
    
    # Unir e ordenar por data
    df_final = df_total.join(df_filtrado, how='outer').fillna(0).sort_index()
    df_final.index.name = 'data'

    # 5. Salvar em JSON
    df_final.to_json("relatorio_diario.json", orient='index', indent=4)
    print("JSON diário salvo.")