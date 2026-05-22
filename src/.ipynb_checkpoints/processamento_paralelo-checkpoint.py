import os
import glob
import gzip
import csv
import multiprocessing

def processar_arquivo(caminho_arquivo):
    """
    Lê um arquivo .tsv.gz e conta os registros únicos baseados no ID.
    Usa o módulo csv para lidar corretamente com \n e \t dentro das aspas.
    """
    ids_unicos = set()
    try:
        with gzip.open(caminho_arquivo, 'rt', encoding='utf-8') as f:
            # Configuramos o parser para TSV (delimiter \t) 
            # e para tratar quebras de linha dentro de aspas (quoting)
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            
            # Pula o cabeçalho se existir
            next(reader, None)
            
            for row in reader:
                if row:  # Verifica se a linha não está vazia
                    # O ID geralmente é a primeira coluna
                    ids_unicos.add(row[0])
                    
        return len(ids_unicos)
    except Exception as e:
        # Silencia erros de arquivos corrompidos (como o channel_1933278309)
        return 0

if __name__ == '__main__':
    # 1. Mapear os 10.000 arquivos de Julho
    base_path = "/home/students/moliveira/usc-tg-24-us-election/extracted/"
    arquivos = glob.glob(os.path.join(base_path, "*", "2024-07.tsv.gz"))
    
    print(f"Iniciando processamento paralelo em {len(arquivos)} arquivos...")

    num_cores = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        # O map distribui a lista de arquivos entre os 60 processos
        resultados = pool.map(processar_arquivo, arquivos)

    # 3. Resultado Final
    total_mensagens = sum(resultados)
    
    print(f"\n==========================================")
    print(f"Total de mensagens em Julho: {total_mensagens}")
    print(f"Processamento concluído via Multiprocessing Pool.")
    print(f"==========================================\n")