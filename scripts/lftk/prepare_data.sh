#!/bin/bash

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_INFER="outputs/lftk/prepare_data_${TIMESTAMP}.log"

python -u -m src.lftk.prepare_data 2>&1 | tee "$LOG_INFER"