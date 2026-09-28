#!/bin/bash

# Aborta o script inteiro se qualquer comando falhar
set -e

# Cria a pasta de logs caso ela não exista
mkdir -p outputs/detoxify

# Captura o horário de início para unificar os nomes dos dois arquivos
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Define os nomes dos dois arquivos de log separadamente
#LOG_TRAIN="outputs/classifier/train_${TIMESTAMP}.log"
LOG_INFER="outputs/detoxify/infer_${TIMESTAMP}.log"

echo "==================================================="
echo "Iniciando o pipeline de Inferência de Toxicidade..."
echo "Logs de Inferência: $LOG_INFER"
echo "==================================================="

# Salva a saída apenas no LOG_INFER (sem o -a, pois é um arquivo novo)
python -u -m src.run_detoxify 2>&1 | tee "$LOG_INFER"

echo "==================================================="
echo "Processo finalizado com sucesso!"
echo "==================================================="