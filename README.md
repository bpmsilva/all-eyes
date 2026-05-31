# Projeto All Eyes

## Instruções:


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
