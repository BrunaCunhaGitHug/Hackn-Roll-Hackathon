from flask import Flask
from .routes import bp # Alterado para importação relativa, comum em pacotes
import os # Para gerar uma secret key ou ler de variáveis de ambiente

def create_app():
    app = Flask(__name__)

    # Configuração da SECRET_KEY
    # Em um ambiente de produção, você NUNCA deve codificar a chave diretamente.
    # Use variáveis de ambiente ou um arquivo de configuração seguro.
    # Para desenvolvimento, podemos usar uma string aleatória ou os.urandom.
    # Exemplo para desenvolvimento:
    app.config['SECRET_KEY'] = os.urandom(24)
    # OU uma string fixa (menos seguro para produção, mas ok para testes locais):
    # app.config['SECRET_KEY'] = 'uma-chave-secreta-muito-forte-e-aleatoria'

    # Registrar o Blueprint
    app.register_blueprint(bp)
    
    return app
