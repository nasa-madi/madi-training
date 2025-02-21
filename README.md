# madi-training
Repository containing training code and adapter files for models trained under the madi project

## Downloading custom model

`cd Models`

`git clone https://huggingface.co/ssbclarke/20250210-llama3.2-3b-lora-aim`

## Setting up a custom Ollama model using Docker

`cd ollama`

`docker compose up -d` to start up an ollama docker container

`docker exec -it ollama sh` to get access to the container to run commands

In the docker container run

`ollama create customModel` which will convert the huggingface safetensors into the ollama format

`ollama run customModel` to chat with the custom model


## Models

### 20250210-llama3.2-3b-lora-aim
https://huggingface.co/ssbclarke/20250210-llama3.2-3b-lora-aim 

trained using only one document from the faa website, the Aeronatical Information Manual. This is a test model to ensure loading and running works correctly through ollama.