import pandas as pd
import os
import zlib
import json
import glob

def decompress_content(compressed_data):
    """
    O artigo menciona que os objetos são JSON serializados e comprimidos com zlib.
    Se o CSV salvou o dado binário como string, pode ser necessário um passo extra.
    """
    if pd.isna(compressed_data) or not isinstance(compressed_data, (bytes, str)):
        return None
    try:
        # Tenta descomprimir se for zlib puro
        return zlib.decompress(compressed_data).decode('utf-8')
    except Exception:
        return compressed_data

def analisar_arquivos_csv(diretorio_base):
    # Procura por todos os CSVs (ajuste a extensão se for .csv.gz, etc)
    arquivos = glob.glob(os.path.join(diretorio_base, "**/*.csv*"), recursive=True)
    print(arquivos)
    for caminho_arquivo in arquivos:
        print(f"Processando: {os.path.basename(caminho_arquivo)}")
        
        # O pandas lida automaticamente com compressões (gzip, zip, bz2) 
        # se a extensão estiver correta.
        try:
            # Usando chunksize para não estourar a memória (o dataset é massivo)
            for chunk in pd.read_csv(caminho_arquivo, chunksize=10000):
                
                # Exemplo: Descomprimir a coluna 'message' se ela seguir o padrão do artigo
                if 'message' in chunk.columns:
                    # Aplique a função de descompressão se necessário
                    # chunk['message_plain'] = chunk['message'].apply(decompress_content)
                    pass
                
                # Aqui você pode aplicar seus filtros (ex: por mês ou por palavra-chave)
                # Exemplo: Filtrar mensagens de um mês específico se houver coluna de data
                # data_filtrada = chunk[chunk['timestamp'].str.contains('2024-11')]
                
                # Processamento de amostra
                print(f"Lidas {len(chunk)} linhas de {os.path.basename(caminho_arquivo)}")
                break # Remova o break para processar o arquivo inteiro
                
        except Exception as e:
            print(f"Erro ao ler {caminho_arquivo}: {e}")

# Uso:
analisar_arquivos_csv('../../telegram_2024/extracted/')