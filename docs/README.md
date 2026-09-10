# Documentação do pipeline

Este diretório documenta o fluxo usado para transformar as mensagens brutas do Telegram em dados classificados por tema e com escores de toxicidade.

## Ordem geral

1. `convert_to_parquet.py`: converte os TSV compactados dos canais selecionados para Parquet.
2. `filtering_english.py`: remove textos muito curtos, limpa URLs/menções e detecta o idioma.
3. `run_preprocessing.py`: mantém o inglês e aplica a normalização linguística.
4. `stratified_sample_channel.py`: cria uma amostra estratificada por mês e canal.
5. `stratified_sample_full.py`: materializa as linhas completas correspondentes aos IDs amostrados.
6. `run_grid_search.py`: testa combinações de embeddings, UMAP e HDBSCAN.
7. `select_best_tm_gs.py`: seleciona as melhores configurações e gera YAMLs.
8. `run_tm_best_config.py`: treina o BERTopic final e exporta os tópicos/macrotópicos.
9. Classificação em dois níveis:
   - político versus não político;
   - macrotópico dentro das mensagens políticas.
10. `run_detoxify.py`: calcula os escores de toxicidade.

O fluxo completo, os caminhos de entrada e saída e os comandos estão em [PIPELINE.md](PIPELINE.md).

## Pré-requisitos

- Python com PySpark, pandas, PyArrow, sentence-transformers, BERTopic, NLTK e Detoxify instalados.
- Dados brutos em TSV compactado (`.tsv.gz`).
- Execução a partir da raiz do repositório, para que `src.utils.data_loader.ROOT` resolva os caminhos corretamente.
- Memória suficiente para os jobs Spark e para carregar os modelos de embedding.