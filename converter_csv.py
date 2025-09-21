# converter_csv.py (VERSÃO FINAL - Simula ML)
import pandas as pd
import numpy as np
import json

ARQUIVO_HASH = 'meus_dados_hash.csv'
ARQUIVO_SAIDA_JS = 'data.js'
NUM_LINHAS_AMOSTRA = 15000

def mapear_e_simular_ml(df):
    print("Mapeando hashes para dados legíveis...")
    cidades_ficticias = [
        'São Paulo, SP', 'Rio de Janeiro, RJ', 'Belo Horizonte, MG', 'Curitiba, PR', 
        'Salvador, BA', 'Porto Alegre, RS', 'Brasília, DF', 'Fortaleza, CE',
        'Recife, PE', 'Manaus, AM', 'Campinas, SP', 'Niterói, RJ', 
        'Juiz de Fora, MG', 'Florianópolis, SC', 'Feira de Santana, BA', 'Gramado, RS'
    ]
    for col in ['place_origin_departure', 'place_destination_departure']:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            hashes_unicos = df[col].unique()
            mapa = {h: c for h, c in zip(hashes_unicos, cidades_ficticias * (len(hashes_unicos) // len(cidades_ficticias) + 1))}
            df[col] = df[col].map(mapa)

    print("Simulando resultados do Modelo de Clusterização (Perfis)...")
    perfis_clientes = {
        0: "Viajante Econômico", # Compra passagens mais baratas, pouca frequência
        1: "Cliente Fiel",       # Compra com alta frequência, valor médio
        2: "Viajante a Negócios",# Compra passagens mais caras, dias de semana
        3: "Planejador de Férias" # Compra com antecedência, alto valor
    }
    # Simula a atribuição de um cluster para cada cliente
    df['cluster'] = np.random.choice(list(perfis_clientes.keys()), size=len(df), p=[0.4, 0.3, 0.15, 0.15])
    df['perfil_cliente'] = df['cluster'].map(perfis_clientes)

    print("Simulando resultados do Modelo Preditivo (Propensão)...")
    # Simula a probabilidade de próxima compra
    propensao = np.random.beta(a=2, b=5, size=len(df)) # Gera valores mais concentrados em probabilidades baixas
    df['propensao_compra'] = np.round(propensao, 2)
    
    df.rename(columns={
        'date_purchase': 'data', 'gmv_success': 'receita', 'place_origin_departure': 'origem',
        'place_destination_departure': 'destino', 'total_tickets_quantity_success': 'passagens'
    }, inplace=True)
    
    return df[['data', 'receita', 'origem', 'destino', 'passagens', 'perfil_cliente', 'propensao_compra']]

print(f"Iniciando processo para '{ARQUIVO_HASH}'...")
try:
    df_hash = pd.read_csv(ARQUIVO_HASH, nrows=NUM_LINHAS_AMOSTRA, encoding='utf-8')
    df_final = mapear_e_simular_ml(df_hash)
    
    js_content = f"const rawData = {df_final.to_json(orient='records', date_format='iso')};"
    
    with open(ARQUIVO_SAIDA_JS, 'w', encoding='utf-8') as f:
        f.write(js_content)
        
    print(f"\nSUCESSO! ✨ O arquivo '{ARQUIVO_SAIDA_JS}' foi criado com dados enriquecidos por ML simulado.")
    print("Agora, atualize o 'dashboard.html' e inicie o servidor local.")

except FileNotFoundError:
    print(f"\nERRO: Arquivo '{ARQUIVO_HASH}' não encontrado!")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")