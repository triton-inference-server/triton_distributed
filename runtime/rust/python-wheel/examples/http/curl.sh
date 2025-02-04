#!/bin/bash

curl -X 'POST' \
  'http://localhost:9992/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-model",
    "messages": [
      {
        "role":"user",
        "content":"Hello! How are you?"
      },
      {
        "role":"assistant",
        "content":"Hi! I am quite well, how can I help you today?"
      },
      {
        "role":"user",
        "content":"Can you write me a song? Use as many emojis as possible."
      }
    ],
    "max_tokens": 64,
    "stream": true
  }'
