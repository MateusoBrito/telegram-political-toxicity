#!/bin/bash

echo "================================================="
echo "Iniciando as execuções dos 4 melhores modelos..."
echo "================================================="

# 1. Roda a configuração com o melhor SCORE GLOBAL
echo "-> Rodando: Melhor SCORE GLOBAL"
python -u -m src.run_tm_best_config --config configs/config_best_score_global.yaml | tee "outputs/tm_best_score.log"

# 2. Roda a configuração com a melhor SILHOUETTE
echo "-> Rodando: Melhor SILHOUETTE"
python -u -m src.run_tm_best_config --config configs/config_best_silhouette.yaml  | tee "outputs/tm_best_silhouette.log"

# 3. Roda a configuração com a melhor COHERENCE
echo "-> Rodando: Melhor COHERENCE"
python -u -m src.run_tm_best_config --config configs/config_best_coherence.yaml  | tee "outputs/tm_best_coherence.log"

# 4. Roda a configuração com a melhor DIVERSITY
echo "-> Rodando: Melhor DIVERSITY"
python -u -m src.run_tm_best_config --config configs/config_best_diversity.yaml  | tee "outputs/tm_best_diversity.log"

echo "================================================="
echo "Todas as execuções foram concluídas com sucesso!"
echo "================================================="