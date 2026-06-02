# Projeto All Eyes

## Instruções:

Aqui você encontrará as instruções para configurar e executar o projeto All Eyes, que utiliza a biblioteca MediaPipe e o repositório L2CS para detecção e rastreamento facial.

### Instalação

1. Clone o repositório do projeto:
   ```bash
   git clone git@github.com:bpmsilva/all-eyes.git
    ```
2. Navegue até o diretório do projeto:
    ```bash
    cd all-eyes
    ```
3. Crie um ambiente virtual (opcional, mas recomendado):
    ```bash
    python3 -m venv main-venv
    source main-venv/bin/activate  # No Windows: main-venv\Scripts\activate
    ```
4. Instale as dependências necessárias:
    ```bash
    pip3 install -r requirements.txt
    ```

### Arquivos

- `ola_mundo.py`: Script simples que verifica se a câmera está acessível.
- `media_pipe.py`: Script que utiliza a biblioteca MediaPipe para detectar e rastrear pontos faciais. Exige o uso do arquivo `face_landmarker.task` da pasta `models`.
- `eye-tracker.py`: Script que utiliza o repositório L2CS para detecção e rastreamento facial. Exige o uso do arquivo `face_landmarker.task` da pasta `models`.

### Execução

Para executar os scripts, use os seguintes comandos no terminal:
- Para verificar a câmera:
    ```bash
    python3 ola_mundo.py # Você precisa estar na pasta do projeto para acessar o arquivo ola_mundo.py
    ```
- Para usar o MediaPipe:
    ```bash
    python3 media_pipe.py # Você precisa estar na pasta do projeto para acessar o arquivo media_pipe.py
    ```
- Para usar o L2CS:
    ```bash
    python3 eye-tracker.py # Você precisa estar na pasta l2cs para acessar o arquivo eye-tracker.py
    ```

