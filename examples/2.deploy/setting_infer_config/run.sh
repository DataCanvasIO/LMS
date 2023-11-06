#!/bin/bash

api_key=default
port=18991

lms deploy --model_name opt-125m --gpu 0 --port 18991 --api_key $api_key --infer_config infer_config.json




curl -v --noproxy '*' --location 'http://127.0.0.1:18991/prediction' \
--header 'Api_key: default' \
--header 'Content-Type: application/json' \
--data '{
  "messages": [
    {
      "role": "User",
      "content": "hello"
    }
  ],
  "repetition_penalty": 1.2,
  "top_k": 40,
  "top_p": 0.5,
  "temperature": 0.7,
  "max_new_tokens": 100
}'



lms undeploy --model_name opt-125m
