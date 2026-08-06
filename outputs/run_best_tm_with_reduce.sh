#!/bin/bash

python -u -m src.run_tm_best_config --config configs/best_config_reduce.yaml | tee "outputs/tm_best_reduced.log"
