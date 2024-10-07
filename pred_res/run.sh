#!/bin/bash

# setup logging
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/main_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# run main.py with logging
python main.py 2>&1 | tee "$LOG_FILE"