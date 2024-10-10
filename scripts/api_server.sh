#!/bin/bash
source .env

MODEL=$DEFAULT_MODEL
HOST=$DEFAULT_HOST
PORT=$DEFAULT_PORT
DEVICE=$DEFAULT_DEVICE

SERVER_PATH="-m vllm.entrypoints.openai.api_server"
LOG_LEVEL="info"

python $SERVER_PATH \
    --model $MODEL \
    --host $HOST \
    --port $PORT \
    --device $DEVICE \
    --enable-prefix-caching \
    --uvicorn-log-level $LOG_LEVEL
