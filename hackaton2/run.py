# run.py (na raiz do seu projeto)
from app import create_app # Importa a factory function do seu pacote 'app'
import os

# Cria uma instância da aplicação Flask usando a factory
# Isso garante que o blueprint com suas rotas (incluindo a que aceita POST) seja registrado.
app = create_app()

if __name__ == "__main__":
    # Define a porta. Importante se for fazer deploy em algumas plataformas.
    port = int(os.environ.get("PORT", 5000))
    # Executa a aplicação
    # debug=True é útil para desenvolvimento, mas desative em produção.
    app.run(host='0.0.0.0', port=port, debug=True)

