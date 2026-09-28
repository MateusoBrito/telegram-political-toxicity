# Pipeline de processamento e análise

## Visão geral

O projeto processa mensagens do Telegram em etapas incrementais, usando Parquet como formato intermediário e particionando os jobs Spark por ano e mês quando o volume é grande.

```text
TSV.GZ bruto
  -> data/raw
  -> messages_with_language
  -> messages_preprocessed
  -> stratified_sample_ids
  -> stratified_sample_full
  -> topic modeling
  -> classificação político/macrotópico
  -> toxicity
```

As etapas são independentes no armazenamento: cada uma lê a saída da anterior e grava uma nova pasta. Isso permite reprocessar um mês ou uma etapa sem refazer necessariamente todo o pipeline.

`src/processamento_paralelo.py` é um utilitário auxiliar de diagnóstico: conta IDs únicos nos arquivos de julho usando `multiprocessing`. Ele não produz uma saída consumida pelas etapas seguintes.

## 1. Conversão para Parquet

**Script:** `src/convert_to_parquet.py`

O script lê os canais listados em `reports/group_to_keep.txt`, procurando arquivos `*.tsv.gz` dentro de `telegram_2024/extracted`. O Spark usa um schema explícito para preservar IDs e interpretar corretamente TSVs com aspas e quebras de linha.

Características importantes:

- processa os canais em lotes de 500;
- ignora arquivos corrompidos configurando `spark.sql.files.ignoreCorruptFiles`;
- grava o primeiro lote com `overwrite` e os seguintes com `append`;
- adiciona `file_path` e extrai `group_name` do caminho do arquivo;
- registra falhas em `data/lotes_com_erro.txt`.

**Entrada:** TSV compactado dos canais extraídos e `reports/group_to_keep.txt`.

**Saída:** `data/raw/`.

```bash
python -m src.convert_to_parquet
```

Os caminhos de origem e destino estão definidos no próprio script e podem precisar ser ajustados para o ambiente de execução.

## 2. Filtragem de idioma

**Script:** `src/filtering_english/filtering_english.py`

Primeiro, o script cria `text_clean`, removendo URLs e menções. Em seguida, descarta mensagens sem texto ou com menos de 20 caracteres. A detecção de idioma é feita com `langdetect`, usando uma semente fixa para tornar o resultado reproduzível.

O processamento é feito mês a mês. Para cada partição, o script cria:

- `datetime`, derivado de `timestamp`;
- `year`;
- `month`;
- `language`.

**Entrada:** `data/raw/`.

**Saída:** `data/processed/messages_with_language/`, particionada por `year` e `month`.

```bash
python -m src.filtering_english.filtering_english
```

## 3. Pré-processamento textual

**Scripts:** `src/pre_processing/run_preprocessing.py` e `src/pre_processing/preprocessing.py`

`run_preprocessing.py` seleciona apenas as mensagens com `language == "en"` e aplica `TextPreprocessor` em uma UDF Pandas do Spark. A limpeza textual realiza:

- conversão para minúsculas;
- remoção de URLs, hashtags e menções;
- remoção de pontuação e números;
- redução de repetições de caracteres;
- tokenização por espaço;
- remoção de stopwords em inglês;
- remoção de palavras com menos de três caracteres;
- lematização com POS tagging do NLTK.

Mensagens vazias ou com até 20 caracteres após a limpeza são descartadas. O resultado é escrito incrementalmente por ano e mês.

**Entrada:** `data/processed/messages_with_language/`.

**Saída:** `data/processed/messages_preprocessed/`, com a coluna `clean_text`.

```bash
python -m src.pre_processing.run_preprocessing
```

Na primeira execução, o script verifica e baixa os recursos necessários do NLTK no driver.

## 4. Amostragem estratificada

### 4.1 Seleção dos IDs

**Script:** `src/stratified_sample_channel.py`

O script considera mensagens de 2024 e cria o estrato `mês_canal`. O tamanho da amostra de cada estrato é proporcional à quantidade de mensagens daquele mês e canal. A seleção é aleatória, mas usa `seed=42` para reprodutibilidade.

O tamanho total desejado é definido por `TOTAL_SAMPLE_SIZE`, atualmente igual a `1_000_000`.

**Entrada:** `data/processed/messages_preprocessed/`.

**Saída:** `data/processed/stratified_sample_ids/`, contendo `id` e `group_name`.

```bash
python -m src.stratified_sample_channel
```

### 4.2 Materialização das mensagens

**Script:** `src/stratified_sample_full.py`

Faz um `left_semi join` entre a base pré-processada e os IDs selecionados. Assim, recupera todas as colunas das mensagens sem trazer IDs que não pertencem à amostra.

**Entrada:** `messages_preprocessed` e `stratified_sample_ids`.

**Saída:** `data/processed/stratified_sample_full/`.

```bash
python -m src.stratified_sample_full
```

## 5. Busca em grade do topic modeling

**Script:** `src/run_grid_search.py`

Para cada modelo de embedding configurado, o script:

1. carrega `clean_text` da amostra;
2. remove textos duplicados para o treinamento;
3. gera ou reutiliza embeddings em cache;
4. testa combinações de UMAP e HDBSCAN;
5. calcula métricas como silhueta, diversidade e coerência $c_v$;
6. salva os resultados por modelo.

**Entrada principal:** `data/processed/stratified_sample_full/`.

**Configuração:** um arquivo YAML com as seções `paths`, `sampling`, `model` e `grid_search`.

**Saídas:** CSVs de resultados em `reports/topic_modeling/grid_search/` e embeddings em `data/cache/embeddings_grid/`.

```bash
python -m src.run_grid_search --config configs/grid_search_config.yaml
```

O caminho acima é o padrão esperado pela documentação; confirme o nome do YAML disponível no ambiente antes de executar.

## 6. Seleção da melhor configuração

**Script:** `src/select_best_tm_gs.py`

Este script combina os CSVs da busca em grade com os logs de execução, recupera métricas de outliers e normaliza silhueta, diversidade e coerência. Depois, gera configurações YAML para diferentes critérios:

- score global combinado;
- silhueta;
- coerência;
- diversidade.

Durante a exportação, `min_cluster_size` é multiplicado por 10 para a escala da base final.

**Entradas:** resultados em `reports/topic_modeling/grid_search/` e `outputs/grid_log*.txt`.

**Saídas:** arquivos `config_best_*.yaml` em `configs/`.

```bash
python -m src.select_best_tm_gs
```

## 7. Treinamento final do BERTopic

**Script:** `src/run_tm_best_config.py`

Lê uma configuração escolhida, gera ou carrega os embeddings, treina o BERTopic com os parâmetros selecionados e exporta:

- dicionário de tópicos;
- resultados associados às mensagens;
- visualizações HTML;
- dados usados para identificar os tópicos e macrotópicos.

Algumas configurações também podem ativar `reduce_outliers`, que tenta reassociar documentos inicialmente classificados no tópico `-1` usando embeddings.

**Entrada:** `stratified_sample_full`, stopwords e um YAML selecionado.

**Saída:** pasta definida por `paths.dir_reports`, normalmente dentro de `reports/topic_modeling/`.

```bash
python -m src.run_tm_best_config --config configs/config_best_score_global.yaml
```

O resultado de macrotópicos é produzido nesta etapa e serve de base para a classificação posterior. A criação/interpretação dos macrotópicos não é uma etapa separada deste repositório de execução.

## 8. Classificação em dois níveis

Os classificadores usam embeddings de sentence-transformers e modelos configurados em YAML. O treinamento combina o texto com os rótulos por `id` e `group_name`.

### 8.1 Nível 1: político ou não político

**Configuração:** `configs/classifier_config.yaml`

O rótulo esperado é `political_category`, e o corpus completo é usado. A avaliação testa os embeddings e classificadores definidos no YAML com validação cruzada estratificada de cinco partes.

Para comparar modelos:

```bash
python -m src.run_compare_classifieds \
  --config configs/classifier_config.yaml
```

Para treinar o modelo final de produção:

```bash
python -m src.train_best_classifier \
  --config configs/classifier_config.yaml \
  --embeddings all-distilroberta-v1 \
  --model lr
```

O artefato final é salvo em `reports/classifier/` como `<modelo>_<embedding>_producao.joblib`.

### 8.2 Nível 2: macrotópico das mensagens políticas

**Configuração:** `configs/classifier_topic_config.yaml`

Neste nível, o rótulo é `macro_topic` e o corpus é filtrado para `political_category == "Politic"`. Assim, o segundo classificador tenta diferenciar os macrotópicos apenas entre as mensagens já identificadas como políticas.

Para avaliar combinações:

```bash
python -m src.run_compare_classifieds \
  --config configs/classifier_topic_config.yaml
```

Para treinar o modelo final:

```bash
python -m src.train_best_classifier \
  --config configs/classifier_topic_config.yaml \
  --embeddings all-distilroberta-v1 \
  --model lr
```

O diretório de saída é definido em `paths.output_dir` no YAML. O arquivo de entrada com os rótulos precisa conter `macro_topic`, normalmente produzido a partir dos resultados do topic modeling.

### 8.3 Inferência distribuída

**Script:** `src/run_infer_class.py`

Carrega o modelo `.joblib`, transmite-o aos workers Spark e classifica a base mês a mês usando `mapInPandas`. A previsão é escrita na coluna `predicted_class`.

```bash
python -m src.run_infer_class \
  --config configs/classifier_config.yaml \
  --embeddings all-distilroberta-v1 \
  --model lr
```

Para executar o segundo nível, use o YAML e o modelo treinados para macrotópicos. A saída padrão do script é `data/processed/messages_classified_topics/`; para evitar misturar os níveis, confirme o `OUTPUT_PATH` antes da execução.

## 9. Detoxify

**Script:** `src/run_detoxify.py`

O script aplica o modelo `Detoxify('original')` em CPU, em lotes de 256 textos, por partição Spark. Ele usa `clean_text` e adiciona seis colunas de probabilidade:

- `toxicity`;
- `severe_toxicity`;
- `obscene`;
- `threat`;
- `insult`;
- `identity_attack`.

O processamento é incremental por ano e mês. Partições já existentes são ignoradas para permitir retomada.

**Entrada:** `data/processed/messages_preprocessed/`.

**Saída:** `data/processed/toxicity/`, particionada por `year` e `month`.

```bash
python -m src.run_detoxify
```

## Conferência rápida

Antes de iniciar uma etapa, confirme que a saída anterior existe:

```text
data/raw/
data/processed/messages_with_language/
data/processed/messages_preprocessed/
data/processed/stratified_sample_ids/
data/processed/stratified_sample_full/
```

Para reprocessar uma partição, remova somente a pasta `year=YYYY/month=MM` correspondente da saída da etapa. Os scripts usam sobrescrita dinâmica por partição em vez de apagar a base inteira.

## Pontos de atenção

- Os scripts usam caminhos absolutos no conversor para Parquet; ajuste-os quando executar fora do ambiente original.
- Os classificadores dependem de um arquivo de tópicos com as chaves `id` e `group_name`.
- A inferência e o Detoxify pulam partições já processadas.
- O processamento Spark consome bastante memória e pode carregar o modelo de embedding em cada worker.
- O filtro de idioma e o pré-processamento são específicos para inglês.