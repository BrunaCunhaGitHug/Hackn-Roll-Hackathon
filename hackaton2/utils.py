import pandas as pd
import re
import uuid

def preprocessar_dados(input_file, output_file):
    """Carrega e limpa os dados, removendo duplicados."""
    try:
        df = pd.read_csv(input_file)
        print(f"\nDados carregados: {len(df)} registros")
        
        # Remover duplicados na coluna 'Resume'
        df_clean = df.drop_duplicates(subset=['Resume'], keep='first')
        print(f"Registros após remoção de duplicados: {len(df_clean)}")
        
        df_clean.to_csv(output_file, index=False)
        return df_clean
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return None

def analisar_candidatos(df, requisitos):
    """Analisa os candidatos e detalha quais habilidades foram encontradas."""
    try:
        # Gerar IDs curtos e únicos
        # df['ID'] = [f"CAND-{str(uuid.uuid4())[:6]}" for _ in range(len(df))]
        
        # Detalhar cada habilidade encontrada
        for skill in requisitos:
            variations = {
                'HTML': ['HTML'],
                'CSS': ['CSS'],
                'BOOTSTRAP': ['BOOTSTRAP'],
                'PHP': ['PHP'],
                'JQUERY': ['JQUERY'],
                'AJAX': ['AJAX'],
                'JAVA SCRIPT': ['JAVASCRIPT', 'JS', 'JAVA SCRIPT']
            }
            
            df[skill] = df['Resume'].apply(
                lambda x: int(any(re.search(r'\b' + re.escape(v) + r'\b', str(x).upper()) 
                                for v in variations.get(skill, [skill])))
            )
        
        # Calcular scores
        df['Match_Score'] = df[requisitos].sum(axis=1)
        df['Match_Percentage'] = (df['Match_Score'] / len(requisitos)) * 100
        
        # Filtrar candidatos relevantes
        candidatos = df[df['Match_Score'] >= 3].sort_values('Match_Percentage', ascending=False)
        
        # Criar coluna com habilidades encontradas
        candidatos['Habilidades_Encontradas'] = candidatos.apply(
            lambda row: ', '.join([skill for skill in requisitos if row[skill] == 1]), 
            axis=1
        )
        
        # Criar coluna com habilidades faltantes
        candidatos['Habilidades_Faltantes'] = candidatos.apply(
            lambda row: ', '.join([skill for skill in requisitos if row[skill] == 0]), 
            axis=1
        )
        
        return candidatos
    except Exception as e:
        print(f"Erro na análise: {e}")
        return None

def main():
    # Configurações
    input_file = "content/Curriculum Vitae.csv"
    cleaned_file = 'Curriculum_Vitae.csv'
    output_file = "content/candidatos_web_design_detalhado.csv"
    requisitos = ['HTML', 'CSS', 'BOOTSTRAP', 'PHP', 'JQUERY', 'AJAX', 'JAVA SCRIPT']
    
    # Pré-processamento
    print("=== ETAPA DE PRÉ-PROCESSAMENTO ===")
    df_clean = preprocessar_dados(input_file, cleaned_file)
    
    if df_clean is not None:
        # Análise
        print("\n=== ETAPA DE ANÁLISE ===")
        resultados = analisar_candidatos(df_clean, requisitos)
        
        if resultados is not None:
            # Exibir resultados
            print("\n=== MELHORES CANDIDATOS ===")
            cols_to_show = ['ID', 'Match_Percentage', 'Match_Score', 
                           'Habilidades_Encontradas', 'Habilidades_Faltantes']
            print(resultados[cols_to_show].head(10).to_string(index=False))
            
            # Salvar resultados completos
            resultados.to_csv(output_file, index=False)
            print(f"\nResultados detalhados salvos em: {output_file}")

if __name__ == "__main__":
    main()