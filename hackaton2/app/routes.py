from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.models import treinar_modelo, prever_area, carregar_modelo_e_transformer
import os
import pandas as pd

bp = Blueprint('main', __name__)

CAMINHO_CSV_ALUNOS = os.path.join(os.path.dirname(__file__), '..', 'data', 'alunos.csv')

@bp.route('/', methods=['GET', 'POST'])
def index():
    # Importar listas de opções do models.py para serem usadas em GET e POST (em caso de erro)
    from app.models import TECNOLOGIAS_POSSIVEIS, CATEGORIAS_APRENDIZADO_POSSIVEIS, CATEGORIAS_PROJETO_POSSIVEIS

    if request.method == 'POST':
        try:
            tecnologias = request.form.getlist('tecnologias_conhecidas')
            aprendizado = request.form.get('aprendizado_recente')
            projeto = request.form.get('tipo_projeto')

            if not tecnologias or not aprendizado or not projeto:
                flash('Por favor, preencha todos os campos do formulário.', 'error')
                # Re-renderiza o formulário com os dados inseridos e mensagem de erro
                return render_template('index.html',
                                       tecnologias_opts=TECNOLOGIAS_POSSIVEIS,
                                       aprendizado_opts=CATEGORIAS_APRENDIZADO_POSSIVEIS,
                                       projeto_opts=CATEGORIAS_PROJETO_POSSIVEIS,
                                       form_data=request.form, # Mantém os dados do formulário
                                       hasattr=hasattr) # Necessário para o template

            dados_candidato = {
                'tecnologias_conhecidas': tecnologias,
                'aprendizado_recente': aprendizado,
                'tipo_projeto': projeto
            }

            area_prevista, justificativa = prever_area(dados_candidato)

            return render_template('results.html',
                                   area=area_prevista,
                                   justificativa=justificativa,
                                   dados_input=dados_candidato)

        except FileNotFoundError as e:
            flash(f"Erro: Modelo de predição não encontrado. Por favor, treine o modelo primeiro. Detalhes: {e}", "error")
            # Re-renderiza o formulário com mensagem de erro
            return render_template('index.html',
                                   tecnologias_opts=TECNOLOGIAS_POSSIVEIS,
                                   aprendizado_opts=CATEGORIAS_APRENDIZADO_POSSIVEIS,
                                   projeto_opts=CATEGORIAS_PROJETO_POSSIVEIS,
                                   form_data=request.form, # Pode estar vazio se o erro foi antes do preenchimento
                                   hasattr=hasattr)
        except ValueError as e:
            flash(f"Erro ao processar os dados: {e}. Verifique o formato do CSV de treinamento e os dados inseridos.", "error")
            return render_template('index.html',
                                   tecnologias_opts=TECNOLOGIAS_POSSIVEIS,
                                   aprendizado_opts=CATEGORIAS_APRENDIZADO_POSSIVEIS,
                                   projeto_opts=CATEGORIAS_PROJETO_POSSIVEIS,
                                   form_data=request.form,
                                   hasattr=hasattr)
        except Exception as e:
            flash(f"Ocorreu um erro inesperado: {e}", "error")
            print(f"Erro detalhado no POST: {e}")
            return render_template('index.html',
                                   tecnologias_opts=TECNOLOGIAS_POSSIVEIS,
                                   aprendizado_opts=CATEGORIAS_APRENDIZADO_POSSIVEIS,
                                   projeto_opts=CATEGORIAS_PROJETO_POSSIVEIS,
                                   form_data=request.form, # Ou um dicionário vazio se preferir limpar em erro genérico
                                   hasattr=hasattr)

    # Método GET: Carregamento inicial da página
    return render_template('index.html',
                           tecnologias_opts=TECNOLOGIAS_POSSIVEIS,
                           aprendizado_opts=CATEGORIAS_APRENDIZADO_POSSIVEIS,
                           projeto_opts=CATEGORIAS_PROJETO_POSSIVEIS,
                           form_data={}, # Dicionário vazio para carga inicial
                           hasattr=hasattr) # <--- ADICIONE ESTA LINHA


@bp.route('/treinar')
def treinar():
    try:
        if not os.path.exists(CAMINHO_CSV_ALUNOS):
            flash(f"Erro: Arquivo de dados para treinamento '{CAMINHO_CSV_ALUNOS}' não encontrado.", "error")
            return redirect(url_for('main.index'))
        try:
            df_test = pd.read_csv(CAMINHO_CSV_ALUNOS, nrows=1)
            if df_test.empty:
                 flash(f"Erro: O arquivo CSV '{CAMINHO_CSV_ALUNOS}' está vazio.", "error")
                 return redirect(url_for('main.index'))
        except Exception as e:
            flash(f"Erro ao ler o arquivo CSV '{CAMINHO_CSV_ALUNOS}': {e}", "error")
            return redirect(url_for('main.index'))

        treinar_modelo(CAMINHO_CSV_ALUNOS)
        flash("Modelo treinado com sucesso!", "success")
        return redirect(url_for('main.index'))
    except ValueError as e:
        flash(f"Erro ao treinar modelo: {e}", "error")
        return redirect(url_for('main.index'))
    except Exception as e:
        flash(f"Erro crítico ao treinar modelo: {e}", "error")
        print(f"Erro detalhado no treino: {e}")
        return redirect(url_for('main.index'))
