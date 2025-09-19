#!/bin/bash

# Select model to preload
# MODEL="qwen3:0.6b"
# MODEL="qwen3:1.7b"
# MODEL="gemma3:1b"
# MODEL="deepseek-r1:1.5b"
# MODEL="qwen2.5:0.5b-instruct"
# MODEL="gemma3n:e2b"
# MODEL="qwen3:4b-instruct"
MODEL="qwen3:4b-instruct-2507-q4_K_M"

# Start the Ollama server in the background
ollama serve &
pid=$!

# Wait for the server to be ready
sleep 5

# Check if the model is already installed
if ! ollama list | grep -q "^$MODEL"; then
  echo "Model $MODEL not found. Pulling..."
  ollama pull "$MODEL"
else
  echo "Model $MODEL already present. Skipping pull."
fi

# Wait for the Ollama server process to finish
wait $pid
