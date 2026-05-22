import os
import glob
import gzip
import csv
import multiprocessing
import json
import pandas as pd
from collections import defaultdict

def processar_arquivo(caminho_arquivo):
    import gzip
    import csv
    import os
    
    ids_unicos = set()
    nome_base = os.path.basename(caminho_arquivo)
    mes = nome_base.split('.')[0]
    
    try:
        if os.path.getsize(caminho_arquivo) == 0:
            print(f"Aviso: {caminho_arquivo} está vazio.")
            return (mes, 0)
            
        with gzip.open(caminho_arquivo, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            next(reader, None)
            
            for row in reader:
                if row:
                    ids_unicos.add(row[0])
                    
        return (mes, len(ids_unicos))
    except Exception as e:
        print(f"Erro em {caminho_arquivo}:{e}")
        return (mes, 0)

def agregar_resultados(lista_resultados):
    totais = defaultdict(int)
    for mes, conta in lista_resultados:
        totais[mes] += conta
    return totais

if __name__ == "__main__":
    base_path = "/home/students/moliveira/telegram_2024/extracted/"
    
    arquivos_totais = glob.glob(os.path.join(base_path, "*", "20*.tsv.gz"))
    
    chats = pd.read_csv("../reports/group_to_keep.txt", header=None, names=['channel'])
    canais_permitidos = set(chats['channel'].astype(str).tolist())
    
    arquivos_filtrados = [
        arq for arq in arquivos_totais 
        if os.path.basename(os.path.dirname(arq)) in canais_permitidos
    ]

    num_cores = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        print("Processando todos os arquivos (Total)...")
        res_total = pool.map(processar_arquivo, arquivos_totais)
        
        print("Processando arquivos dos chats selecionados (Filtrado)...")
        res_filtrado = pool.map(processar_arquivo, arquivos_filtrados)

    # 3. Agregação por mês
    dict_total = agregar_resultados(res_total)
    dict_filtrado = agregar_resultados(res_filtrado)

    # 4. Unir os dados em uma estrutura final
    todos_os_meses = sorted(set(list(dict_total.keys()) + list(dict_filtrado.keys())))
    
    relatorio_final = {}
    for mes in todos_os_meses:
        relatorio_final[mes] = {
            "total": dict_total.get(mes, 0),
            "filtrado": dict_filtrado.get(mes, 0)
        }
        print(f"Mês {mes} -> Total: {relatorio_final[mes]['total']} | Filtrado: {relatorio_final[mes]['filtrado']}")

    # 5. Salvar em JSON
    with open("relatorio_mensagens.json", "w", encoding="utf-8") as jf:
        json.dump(relatorio_final, jf, indent=4, ensure_ascii=False)

    print("\nRelatório salvo com sucesso em 'relatorio_mensagens.json'")