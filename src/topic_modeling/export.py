from typing import List
import numpy as np
from bertopic import BERTopic
from pathlib import Path
import pandas as pd
import random
import json

def export_topic_dictionary(
    topic_model: BERTopic,
    topics: List[int],
    df_unique: pd.DataFrame, 
    df_full: pd.DataFrame,
    output_dir: Path
):
    print("3. Exportando Dicionário de Tópicos Enriquecido...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cria o mapa base ligando o texto limpo ao número do tópico
    topic_mapping = dict(zip(df_unique["clean_text"], topics))
    
    # 2. Anexa temporariamente o tópico ao df_full para puxar os textos originais
    df_full_temp = df_full.copy()
    df_full_temp["topic"] = df_full_temp["clean_text"].map(topic_mapping)

    # 3. Pega a tabela de informações básicas do BERTopic (Contém o -1 dos outliers)
    df_topic_info = topic_model.get_topic_info()
    topics_data = []

    # 4. Constrói o JSON de descrição de todos os tópicos (incluindo o -1)
    for _, row in df_topic_info.iterrows():
        t = row['Topic']
        
        topic_dict = {
            "Topic": t,
            "Count": row["Count"],
            "Name": row["Name"],
            "Representation": row["Representation"], # <-- Aqui vão as 10 palavras c-TF-IDF do tópico!
            "Representative_Docs": [],
            "Sample_Docs": []
        }

        # Filtra apenas os posts reais deste tópico específico
        df_topico = df_full_temp[df_full_temp['topic'] == t].dropna(subset=['clean_text'])
        
        if len(df_topico) > 0:
            # Tradutor de texto limpo para texto original com formatação original
            tradutor = dict(zip(df_topico["clean_text"], df_topico["text"]))

            # A. Extrai os 3 Documentos Representativos (Se houver - outliers não costumam ter representativos formais no BERTopic)
            try:
                rep_docs_limpos = topic_model.get_representative_docs(t)
                rep_docs_limpos = rep_docs_limpos[:3] if rep_docs_limpos else []
                rep_docs_originais = [tradutor.get(doc, doc) for doc in rep_docs_limpos]
                topic_dict["Representative_Docs"] = rep_docs_originais
            except Exception:
                # O BERTopic pode reclamar se tentarmos buscar representativos oficiais do -1
                topic_dict["Representative_Docs"] = []

            # B. Extrai amostras aleatórias para vermos o que caiu lá (Funciona excelente para Outliers!)
            pool_amostra_original = df_topico["text"].dropna().drop_duplicates().tolist()
            
            # Evita duplicar representativos na amostragem
            pool_amostra = [doc for doc in pool_amostra_original if doc not in topic_dict["Representative_Docs"]]
            
            # Sorteia até 7 exemplos reais (seja tópico normal ou outlier)
            n_samples = min(7, len(pool_amostra))
            sample_docs = random.sample(pool_amostra, n_samples) if n_samples > 0 else []
            topic_dict["Sample_Docs"] = sample_docs

        topics_data.append(topic_dict)

    # Salva o JSON final super formatado
    with open(output_dir / "topic_info.json", "w", encoding="utf-8") as f:
        json.dump(topics_data, f, indent=4, ensure_ascii=False)

    # 5. Salva a relação final ID + Channel -> Tópico
    colunas_identificacao = ["id", "channel"] if "channel" in df_full.columns else ["id"]
    df_mapping = df_full[colunas_identificacao + ["clean_text"]].copy()
    df_mapping["topic"] = df_mapping["clean_text"].map(topic_mapping)
    df_mapping = df_mapping.drop(columns=["clean_text"])
    
    # Salva o Parquet que você usará no LFTK
    df_mapping.to_parquet(output_dir / "post_topics.parquet", index=False)

    print(f"   -> Dicionário e Parquet salvos em: {output_dir}")

def export_visualizations(topic_model: BERTopic, output_dir: Path):
    """
    Gera e salva os relatórios interativos em HTML.
    """
    print("4. Gerando Visualizações HTML...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_topic_info = topic_model.get_topic_info()
    n_clusters_total = len(df_topic_info) - 1 # Ignora o tópico -1 (outliers)

    # Gráfico de Barras (Assinaturas Linguísticas)
    fig_words = topic_model.visualize_barchart(
        top_n_topics=n_clusters_total, 
        n_words=10,
        title="Assinaturas Linguísticas do Ecossistema (c-TF-IDF)"
    )
    fig_words.write_html(output_dir / "visualizacao_topicos.html")

    # Mapa Topológico 2D
    fig_2d = topic_model.visualize_topics(
        title="Topologia do Ecossistema: Distância entre Clusters Ideológicos"
    )
    fig_2d.write_html(output_dir / "mapa_2d_topicos.html")

    print("   -> Gerando Árvore Hierárquica...")
    fig_hierarchy = topic_model.visualize_hierarchy(
        custom_labels=True, 
        title="Árvore Hierárquica dos Subreddits"
    )
    fig_hierarchy.write_html(output_dir / "hierarquia_topicos.html")
    
    print(f"   -> Visualizações salvas em: {output_dir}")