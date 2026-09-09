#!/bin/bash

# Aborta o script inteiro se qualquer comando falhar
set -e

# Cria a pasta de logs caso ela não exista
mkdir -p outputs/classifier

# Captura o horário de início para unificar os nomes dos dois arquivos
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Define os nomes dos dois arquivos de log separadamente
#LOG_TRAIN="outputs/classifier/train_${TIMESTAMP}.log"
LOG_INFER="outputs/classifier/infer_${TIMESTAMP}.log"

echo "==================================================="
echo "Iniciando o pipeline de Classificação..."
#echo "Logs de Treino: $LOG_TRAIN"
echo "Logs de Inferência: $LOG_INFER"
echo "==================================================="

echo -e "\n--- 1. Treinando o modelo final ---"
# Salva a saída apenas no LOG_TRAIN
#python -u -m src.train_best_classifier \
#    --config configs/classifier_config.yaml \
#    --embeddings "all-distilroberta-v1" \
#    --model "lr" 2>&1 | tee "$LOG_TRAIN"

echo -e "\n--- 2. Executando a inferência no Spark ---"
# Salva a saída apenas no LOG_INFER (sem o -a, pois é um arquivo novo)
python -u -m src.run_infer_class \
    --config configs/classifier_config.yaml \
    --embeddings "all-distilroberta-v1" \
    --model "lr" 2>&1 | tee "$LOG_INFER"

echo "==================================================="
echo "Processo finalizado com sucesso!"
echo "==================================================="