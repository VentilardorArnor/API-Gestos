# Sistema de Reconhecimento de Gestos

Sistema de reconhecimento de gestos em tempo real usando visão computacional e aprendizado de máquina.

## Funcionalidades

- Detecção de mãos em tempo real
- Reconhecimento de 5 gestos diferentes:
  - Apontando (pointing)
  - Punho fechado (fist)
  - Palma aberta (open_palm)
  - Pinça (pinch)
  - Paz (peace)
- Interface web para visualização
- API REST para integração

## Requisitos

- Python 3.8+
- OpenCV
- MediaPipe
- FastAPI
- Scikit-learn
- Outras dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/sistema-gestos.git
cd sistema-gestos
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure o ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

## Uso

1. Inicie a API:
```bash
python api.py
```

2. Em outro terminal, inicie o servidor web:
```bash
python serve.py
```

3. Acesse a interface web:
```
http://localhost:8080
```

## API

A API está disponível em `http://localhost:8000` com os seguintes endpoints:

- `GET /`: Verifica se a API está online
- `GET /status`: Retorna o status do sistema
- `WS /ws`: WebSocket para reconhecimento em tempo real

Documentação da API disponível em:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Estrutura do Projeto

```
/
├── api.py                 # API principal
├── hand_detector.py       # Detector de mãos
├── ai_model.py           # Modelo de IA
├── gesture_recognizer.py  # Reconhecimento de gestos
├── requirements.txt       # Dependências
├── README.md             # Documentação
├── .env.example          # Exemplo de configuração
├── gunicorn_config.py    # Configuração do Gunicorn
├── serve.py             # Servidor de arquivos estáticos
├── training_data/        # Dados de treinamento
└── static/              # Arquivos estáticos
    └── index.html       # Interface web
```

## Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes. 