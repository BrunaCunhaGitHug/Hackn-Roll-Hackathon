import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin # Adicionado
import joblib
import os
import numpy as np

MODELO_CAMINHO = os.path.join(os.path.dirname(__file__), 'modelo_predicao_area.pkl')

# --- LISTAS E MAPEAMENTO SIMPLIFICADOS (COMO ANTES) ---
TECNOLOGIAS_POSSIVEIS = [
    'Python', 'JavaScript', 'HTML/CSS', 'SQL', 'Pandas', 'Java', 'Spring Boot',
    'Swift', 'Git', 'Vue.js', 'Docker', 'Linux', 'React', 'Node.js', 
    'PHP', 'MySQL', 'AWS', 'Terraform', 'Ansible', 'Shell Scripting'
]

CATEGORIAS_APRENDIZADO_POSSIVEIS = [
    'Desenvolvimento Web Fullstack', 'Ciência de Dados', 'Desenvolvimento Web Backend',
    'Desenvolvimento Mobile (iOS)', 'Desenvolvimento Web Frontend',
    'DevOps / Engenharia de Infraestrutura Cloud',
    'Inteligência Artificial / Machine Learning', 
    'Análise de Dados / Business Intelligence'
]

CATEGORIAS_PROJETO_POSSIVEIS = [
    'Aplicativo Web Completo', 'Analise de Dados Basica', 'API Simples',
    'App iOS Simples', 'Interface Web Simples', 'Script de Automacao',
    'Modelo de Machine Learning', 
    'Dashboard de BI'
]

JUSTIFICATIVAS_MAPEAMENTO = {
    "Desenvolvimento Web Fullstack": {
        "tecnologias_chave": ["Python", "JavaScript", "HTML/CSS", "React", "Node.js", "SQL"],
        "aprendizados_chave": ["Desenvolvimento Web Fullstack", "Desenvolvimento Web Frontend", "Desenvolvimento Web Backend"],
        "projetos_chave": ["Aplicativo Web Completo", "Interface Web Simples", "API Simples"],
        "descricao": "Ideal para construir aplicações web completas, do visual à lógica do servidor.",
        "habilidades_desenvolvidas": "Arquitetura de software, integração de sistemas."
    },
    "Ciência de Dados": {
        "tecnologias_chave": ["Python", "SQL", "Pandas", "Scikit-learn", "TensorFlow"],
        "aprendizados_chave": ["Ciência de Dados", "Inteligência Artificial / Machine Learning"],
        "projetos_chave": ["Analise de Dados Basica", "Modelo de Machine Learning"],
        "descricao": "Para extrair insights de dados usando estatística e machine learning.",
        "habilidades_desenvolvidas": "Pensamento analítico, modelagem estatística."
    },
    "Desenvolvimento Web Backend": {
        "tecnologias_chave": ["Java", "Spring Boot", "SQL", "Python", "Node.js"],
        "aprendizados_chave": ["Desenvolvimento Web Backend"],
        "projetos_chave": ["API Simples"],
        "descricao": "Focado na lógica do servidor, APIs e bancos de dados.",
        "habilidades_desenvolvidas": "Design de APIs, segurança de dados."
    },
    "Desenvolvimento Mobile (iOS Nativo)": {
        "tecnologias_chave": ["Swift", "Git"],
        "aprendizados_chave": ["Desenvolvimento Mobile (iOS)"],
        "projetos_chave": ["App iOS Simples"],
        "descricao": "Especializado em criar aplicativos para o ecossistema Apple.",
        "habilidades_desenvolvidas": "Conhecimento do ecossistema Apple, UI/UX mobile."
    },
    "Desenvolvimento Web Frontend": {
        "tecnologias_chave": ["JavaScript", "HTML/CSS", "Vue.js", "React"],
        "aprendizados_chave": ["Desenvolvimento Web Frontend"],
        "projetos_chave": ["Interface Web Simples"],
        "descricao": "Criação da interface visual e interativa de sites e aplicações.",
        "habilidades_desenvolvidas": "Criatividade, design responsivo."
    },
    "DevOps / Engenharia de Infraestrutura Cloud": {
        "tecnologias_chave": ["Python", "Docker", "Linux", "AWS", "Terraform", "Ansible"],
        "aprendizados_chave": ["DevOps / Engenharia de Infraestrutura Cloud"],
        "projetos_chave": ["Script de Automacao"],
        "descricao": "Une desenvolvimento e operações, focando em automação e nuvem.",
        "habilidades_desenvolvidas": "Automação, infraestrutura como código, CI/CD."
    }
}
# --- FIM DAS LISTAS E MAPEAMENTO SIMPLIFICADOS ---

# --- INÍCIO DO MLBWrapper ---
class MLBWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, classes=None):
        self.classes = classes
        self.mlb = MultiLabelBinarizer(classes=self.classes)
        self.classes_ = None # Atributo esperado pelo scikit-learn

    def fit(self, X, y=None):
        # X aqui é esperado ser uma Series de listas de strings (ex: X['tecnologias_conhecidas'])
        self.mlb.fit(X)
        self.classes_ = self.mlb.classes_ # Expor o atributo classes_ após o fit
        return self

    def transform(self, X, y=None):
        # X aqui é esperado ser uma Series de listas de strings
        return self.mlb.transform(X)

    def get_feature_names_out(self, input_features=None):
        # Retorna os nomes das features geradas (os nomes das classes/tecnologias)
        if hasattr(self.mlb, 'classes_'):
            return self.mlb.classes_.astype(str)
        return np.array([]) # Retorna array vazio se não estiver fitado ou não tiver classes
# --- FIM DO MLBWrapper ---

def treinar_modelo(caminho_csv):
    print(f"[DEBUG MODELOS] Iniciando treinar_modelo com o arquivo: {caminho_csv}")
    if not os.path.exists(caminho_csv):
        print(f"[DEBUG MODELOS] ERRO CRÍTICO: Arquivo CSV não encontrado em {caminho_csv}")
        raise FileNotFoundError(f"Arquivo CSV não encontrado em {caminho_csv}")

    # Bloco de DEBUG (pode ser mantido ou removido se o CSV estiver OK)
    print("[DEBUG MODELOS] Lendo as primeiras linhas do CSV para inspeção...")
    LINHA_DO_ARQUIVO_COM_ERRO_PANDAS = 6 
    try:
        with open(caminho_csv, mode='r', encoding='utf-8') as f_debug:
            linhas_debug = f_debug.readlines()
            if len(linhas_debug) >= LINHA_DO_ARQUIVO_COM_ERRO_PANDAS:
                linha_indice_debug = LINHA_DO_ARQUIVO_COM_ERRO_PANDAS - 1 
                linha_problematica_bruta = linhas_debug[linha_indice_debug].strip()
                print(f"[DEBUG MODELOS] Conteúdo bruto da linha {LINHA_DO_ARQUIVO_COM_ERRO_PANDAS} do arquivo (índice {linha_indice_debug}): '{linha_problematica_bruta}'")
                num_virgulas_brutas = linha_problematica_bruta.count(',')
                print(f"[DEBUG MODELOS] Número de vírgulas (`,`) encontradas na linha bruta: {num_virgulas_brutas}")
                if num_virgulas_brutas == 4:
                    print("[DEBUG MODELOS] ALERTA: A linha bruta parece ter 4 vírgulas, o que causaria 5 campos.")
                elif num_virgulas_brutas == 3:
                    print("[DEBUG MODELOS] INFO: A linha bruta parece ter 3 vírgulas (correto para 4 campos).")
                else:
                    print(f"[DEBUG MODELOS] AVISO: Número de vírgulas na linha bruta ({num_virgulas_brutas}) é inesperado.")
            elif len(linhas_debug) > 0:
                 print(f"[DEBUG MODELOS] AVISO: O arquivo CSV tem apenas {len(linhas_debug)} linhas. A linha {LINHA_DO_ARQUIVO_COM_ERRO_PANDAS} não existe.")
                 print(f"[DEBUG MODELOS] Última linha do arquivo (índice {len(linhas_debug)-1}): '{linhas_debug[-1].strip()}'")
            else:
                print(f"[DEBUG MODELOS] AVISO: O arquivo CSV está vazio ou não pôde ser lido corretamente.")
        print("[DEBUG MODELOS] Fim da inspeção manual das linhas.")
    except Exception as e_debug:
        print(f"[DEBUG MODELOS] Erro ao tentar ler o CSV para debug: {e_debug}")
    # Fim do bloco de DEBUG

    try:
        df = pd.read_csv(caminho_csv)
    except pd.errors.ParserError as e_parser:
        print(f"[DEBUG MODELOS] ERRO DO PANDAS AO LER CSV: {e_parser}")
        raise e_parser 
    except Exception as e_geral_pandas:
        print(f"[DEBUG MODELOS] ERRO GERAL DO PANDAS: {e_geral_pandas}")
        raise e_geral_pandas

    colunas_necessarias = ['tecnologias_conhecidas', 'aprendizado_recente', 'tipo_projeto', 'area_desejada']
    if not all(col in df.columns for col in colunas_necessarias):
        print(f"[DEBUG MODELOS] ERRO: Colunas esperadas não encontradas no CSV. Colunas encontradas: {df.columns.tolist()}")
        raise ValueError(f"O CSV precisa conter as colunas: {', '.join(colunas_necessarias)}")

    df['tecnologias_conhecidas'] = df['tecnologias_conhecidas'].fillna('')
    df['aprendizado_recente'] = df['aprendizado_recente'].fillna(CATEGORIAS_APRENDIZADO_POSSIVEIS[0])
    df['tipo_projeto'] = df['tipo_projeto'].fillna(CATEGORIAS_PROJETO_POSSIVEIS[0])

    X = df[['tecnologias_conhecidas', 'aprendizado_recente', 'tipo_projeto']]
    y = df['area_desejada']

    # Aplica a transformação para que 'tecnologias_conhecidas' seja uma Series de listas
    # Usar .loc para evitar SettingWithCopyWarning
    X.loc[:, 'tecnologias_conhecidas'] = X['tecnologias_conhecidas'].apply(lambda x: x.split(';') if isinstance(x, str) and x else [])

    preprocessor = ColumnTransformer(
        transformers=[
            # Usar o MLBWrapper aqui, passando a coluna como string para obter uma Series
            ('tech', MLBWrapper(classes=TECNOLOGIAS_POSSIVEIS), 'tecnologias_conhecidas'),
            ('learn', OneHotEncoder(categories=[CATEGORIAS_APRENDIZADO_POSSIVEIS], handle_unknown='ignore', sparse_output=False), ['aprendizado_recente']),
            ('project', OneHotEncoder(categories=[CATEGORIAS_PROJETO_POSSIVEIS], handle_unknown='ignore', sparse_output=False), ['tipo_projeto'])
        ],
        remainder='passthrough'
    )

    modelo_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    modelo_pipeline.fit(X, y) # O erro original acontecia aqui

    joblib.dump(modelo_pipeline, MODELO_CAMINHO)
    print(f"Pipeline (pré-processador + modelo) salvo em {MODELO_CAMINHO}")
    return modelo_pipeline

# ... (Restante das funções: carregar_modelo_e_transformer, gerar_justificativa, prever_area COMO ESTAVAM) ...
def carregar_modelo_e_transformer():
    if not os.path.exists(MODELO_CAMINHO):
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODELO_CAMINHO}. "
            "Execute o treinamento primeiro."
        )
    pipeline = joblib.load(MODELO_CAMINHO)
    return pipeline

def gerar_justificativa(dados_candidato, area_prevista):
    if area_prevista not in JUSTIFICATIVAS_MAPEAMENTO:
        return f"A área de **{area_prevista}** é uma sugestão. Recomendamos explorar mais sobre ela!"
    info_area = JUSTIFICATIVAS_MAPEAMENTO[area_prevista]
    justificativa_partes = [f"Sugerimos a área de **{area_prevista}** porque:"]
    if "descricao" in info_area:
        justificativa_partes.append(f"- **Visão Geral:** {info_area['descricao']}")
    
    tecnologias_usuario = dados_candidato.get('tecnologias_conhecidas', [])
    matches_tech = [t for t in tecnologias_usuario if t in info_area.get("tecnologias_chave", [])]
    if matches_tech:
        justificativa_partes.append(f"- **Habilidades:** Seu conhecimento em **{', '.join(matches_tech)}** é relevante.")
    
    aprendizado_usuario = dados_candidato.get('aprendizado_recente')
    if aprendizado_usuario and aprendizado_usuario in info_area.get("aprendizados_chave", []):
        justificativa_partes.append(f"- **Aprendizado:** Seu foco em **'{aprendizado_usuario}'** se alinha bem.")
        
    projeto_usuario = dados_candidato.get('tipo_projeto')
    if projeto_usuario and projeto_usuario in info_area.get("projetos_chave", []):
        justificativa_partes.append(f"- **Projetos:** Sua experiência com **'{projeto_usuario}'** é um bom indicador.")

    if "habilidades_desenvolvidas" in info_area:
        justificativa_partes.append(f"- **Potencial:** Você poderá desenvolver: {info_area['habilidades_desenvolvidas']}")
    
    if len(justificativa_partes) <= 1:
         return f"A área de **{area_prevista}** foi sugerida. {info_area.get('descricao', '')} Pesquise mais sobre as tecnologias e projetos comuns!"
    return "\n".join(justificativa_partes)

def prever_area(candidato_dict):
    pipeline = carregar_modelo_e_transformer()
    tech_list = candidato_dict.get('tecnologias_conhecidas', [])
    if not isinstance(tech_list, list): 
        tech_list = []
    
    aprendizado_recente_val = candidato_dict.get('aprendizado_recente', CATEGORIAS_APRENDIZADO_POSSIVEIS[0])
    if aprendizado_recente_val not in CATEGORIAS_APRENDIZADO_POSSIVEIS:
        aprendizado_recente_val = CATEGORIAS_APRENDIZADO_POSSIVEIS[0]
        
    tipo_projeto_val = candidato_dict.get('tipo_projeto', CATEGORIAS_PROJETO_POSSIVEIS[0])
    if tipo_projeto_val not in CATEGORIAS_PROJETO_POSSIVEIS:
        tipo_projeto_val = CATEGORIAS_PROJETO_POSSIVEIS[0]
        
    dados_para_df = {
        'tecnologias_conhecidas': [tech_list],
        'aprendizado_recente': [aprendizado_recente_val],
        'tipo_projeto': [tipo_projeto_val]
    }
    df_candidato = pd.DataFrame(dados_para_df)
    predicao = pipeline.predict(df_candidato)
    area_prevista = predicao[0]
    justificativa = gerar_justificativa(candidato_dict, area_prevista)
    return area_prevista, justificativa
