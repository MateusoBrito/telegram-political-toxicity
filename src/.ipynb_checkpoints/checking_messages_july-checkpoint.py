import os
import glob

def validar_processamento_julho(diretorio_original, diretorio_july):
    # 1. Mapear o que DEVERIA existir (Base Original)
    # Estrutura: .../channel_ID/2024-07.tsv.gz
    print("Mapeando base original...")
    arquivos_originais = glob.glob(os.path.join(diretorio_original, "**/2024-07.tsv.gz"), recursive=True)
    
    # Extrai o nome da pasta pai (ex: channel_1000154686)
    ids_esperados = {os.path.basename(os.path.dirname(f)) for f in arquivos_originais}
    
    # 2. Mapear o que VOCÊ TEM (Sua pasta filtrada)
    # Estrutura: channel_ID-2024-07.tsv.gz
    print("Mapeando sua pasta de julho...")
    arquivos_filtrados = glob.glob(os.path.join(diretorio_july, "*.tsv.gz"))
    
    # Extrai a primeira parte do nome do arquivo antes do primeiro '-'
    ids_processados = set()
    for f in arquivos_filtrados:
        nome_arquivo = os.path.basename(f)
        canal_id = nome_arquivo.split('-')[0] # Pega 'channel_ID'
        ids_processados.add(canal_id)
    
    # 3. Comparação Logística
    faltantes = ids_esperados - ids_processados
    extras = ids_processados - ids_esperados
    
    # Relatório Final
    print(f"\n--- Relatório de Integridade (Julho 2024) ---")
    print(f"Total de canais na origem com dados de julho: {len(ids_esperados)}")
    print(f"Total de canais processados no seu destino: {len(ids_processados)}")
    
    if not faltantes:
        print("Sucesso: Todos os canais esperados estão presentes.")
    else:
        print(f"Erro: Faltam {len(faltantes)} canais na sua pasta filtrada.")
        print(f"Exemplos de IDs que não foram encontrados: {list(faltantes)[:5]}")

    if extras:
        print(f"Nota: Existem {len(extras)} arquivos no destino que não constam na busca da origem.")

# Configuração dos caminhos conforme seu ambiente Jupyter
diretorio_origem = "../../telegram_2024/extracted/"
diretorio_destino = "../../messages_july"

validar_processamento_julho(diretorio_origem, diretorio_destino)