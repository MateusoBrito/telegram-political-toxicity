#!/bin/bash

LOG_FILE="outputs/classifier_topic/classifier_$(date +%Y%m%d_%H%M%S).log"

echo "==================================================="
echo "Iniciando a comparação dos classificadores..."
echo "Acompanhe os logs aqui ou no arquivo: $LOG_FILE"
echo "==================================================="

# Executa o script em modo unbuffered (-u) e salva a saída e os erros (2>&1) no log
python -u -m src.run_compare_classifieds --config configs/classifier_topic_config.yaml 2>&1 | tee "$LOG_FILE"

echo "==================================================="
echo "Processo finalizado! Log salvo em: $LOG_FILE"
echo "==================================================="