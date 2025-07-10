import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import clickhouse_connect
import collections 
import io 
from datetime import datetime
from typing import List
import numpy as np

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
# Assuming ai_data_science_team is a custom library you have for Tab 2
from ai_data_science_team import PandasDataAnalyst, DataWranglingAgent, DataVisualizationAgent

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Painel Inteligente Unimed",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🩺"
)

# --- STYLING ---
st.markdown("""
    <style>
        /* Base colors */
        :root {
            --primary-color: #007f3e; /* Unimed Green */
            --secondary-color: #006633; /* Darker Green */
            --background-color: #f0f2f6; /* Light grey background */
            --card-background-color: #ffffff;
            --text-color: #333333;
            --sidebar-background-color: #e8f5e9; /* Lighter green for sidebar */
        }

        body, .stApp {
            background-color: var(--background-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stApp > header { background-color: var(--primary-color); color: white; }
        .stApp > header .st-emotion-cache-1avcm0n { color: white; } /* Specific to Streamlit's header text */
        h1, h2, h3 { color: var(--primary-color); }
        .st-emotion-cache-16txtl3 { background-color: var(--sidebar-background-color); } /* Sidebar background */
        .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 .st-emotion-cache-10oheav { color: var(--secondary-color); } /* Sidebar titles */
        .stMetric { background-color: var(--card-background-color); border-radius: 8px; padding: 15px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .stMetric label { color: var(--secondary-color) !important; font-weight: bold; }
        .stMetric .st-emotion-cache-1wivap2 { font-size: 2em !important; } /* Metric value size */
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: var(--card-background-color); border-radius: 4px 4px 0px 0px; padding: 10px 20px; color: var(--secondary-color); }
        .stTabs [aria-selected="true"] { background-color: var(--primary-color); color: white; font-weight: bold; }
        .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; background-color: var(--card-background-color); }
        .stExpander header { background-color: #f9f9f9; color: var(--primary-color); font-weight: bold; }
        .stDataFrame { width: 100%; }
        .stTextInput input, .stSelectbox select, .stDateInput input { border-radius: 5px; }
        .custom-card {
            background-color: var(--card-background-color);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 15px; /* Provides space if cards stack in a column or for rows */
            height: 100%; /* For consistent card height in a row, if needed by content */
        }
        .custom-card h5 { /* Specific styling for titles within custom cards */
            color: var(--primary-color);
            margin-top: 0px; /* Remove default top margin */
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        .custom-card p.value { /* Styling for the main value in custom cards */
            font-size: 1.8em;
            font-weight: bold;
            color: var(--secondary-color);
            margin-bottom: 5px;
        }
        .custom-card p.help-text { /* Styling for the help text in custom cards */
            font-size: 0.9em;
            color: #555555;
            margin-bottom: 0px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #007f3e;'>🩺 Painel de IA para Resultados Laboratoriais Unimed</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.image("unimed-removebg-preview.png", width=350)
    st.markdown("<h2 style='color: #006633;'>🔐 Configuração da IA</h2>", unsafe_allow_html=True)
    api_key = st.text_input("Chave da API da OpenAI", type="password", help="Insira sua chave da API OpenAI para ativar os recursos de IA.")
    model_option = st.selectbox("Modelo OpenAI", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"], index=0)

    #st.markdown("<h2 style='color: #006633;'>🗄️ Configuração do Banco de Dados ClickHouse</h2>", unsafe_allow_html=True)
    
if not api_key:
    st.warning(
        "🔑 Por favor, insira sua chave da API da OpenAI na barra lateral para habilitar as funcionalidades de IA e carregar os dados."
    )
    st.stop()


llm = None
pandas_data_analyst = None

if api_key:
    try:
        llm = ChatOpenAI(model=model_option, api_key=api_key, temperature=0.3)
        data_wrangling_agent = DataWranglingAgent(
            model=llm, bypass_recommended_steps=True, log=False
        )
        data_visualization_agent = DataVisualizationAgent(model=llm, log=False)
        pandas_data_analyst = PandasDataAnalyst(
            model=llm,
            data_wrangling_agent=data_wrangling_agent,
            data_visualization_agent=data_visualization_agent,
        )
    except Exception as e:
        st.error(
            f"Erro ao inicializar os modelos de IA: {e}. Verifique sua chave da API."
        )
        st.stop()

from dotenv import load_dotenv
import os

load_dotenv()  # Carrega o .env da raiz do projeto

ch_host = os.getenv("CH_HOST")
ch_port = int(os.getenv("CH_PORT", 8123))  # forneça default se quiser
ch_username = os.getenv("CH_USERNAME")
ch_password = os.getenv("CH_PASSWORD")
ch_database = os.getenv("CH_DATABASE")
ch_table = os.getenv("CH_TABLE")



# --- GLOBAL DEFINITIONS ---
ALTERATION_MARKERS = ["Acima do VR", "Abaixo do VR", "Alto", "Baixo", "Aumentado", "Diminuído", "Positivo"]

import unicodedata
import re

def normalize_column_name(name):
    """
    Normalize column names by removing accents and special characters
    """
    # Remove accents
    normalized = unicodedata.normalize('NFD', name)
    normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    
    # Convert to lowercase and replace spaces with underscores
    normalized = normalized.lower().replace(' ', '_')
    
    # Remove any remaining special characters except underscores
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    
    return normalized

@st.cache_data
def load_data_from_clickhouse(_client_params, table_name, alteration_markers=None):
    """
    Load data from ClickHouse database with server-side calculations
    """
    try:
        client = clickhouse_connect.get_client(**_client_params)
        
        # Get table schema first to identify status columns
        schema_query = f"""
        SELECT name, type 
        FROM system.columns 
        WHERE table = '{table_name}' 
        AND database = currentDatabase()
        """
        schema_df = client.query_df(schema_query)
        
        # Find status columns
        status_cols = [col for col in schema_df['name'].tolist() if col.lower().endswith('_status')]
        
        if not status_cols:
            st.warning("Nenhuma coluna '_status' encontrada. Funcionalidades de alteração podem não funcionar corretamente.")
            # Simple query without alteration calculations
            query = f"""
            SELECT *,
                   false as paciente_com_alteracao,
                   0 as qtde_exames_alterados
            FROM {table_name}
            """
        else:
            # Build alteration markers condition
            if alteration_markers is None:
                alteration_markers = ALTERATION_MARKERS
            
            # Properly escape column names with backticks for ClickHouse
            escaped_status_cols = [f"`{col}`" for col in status_cols]
            
            # Create SQL condition for alteration markers - properly format the tuple
            markers_list = "', '".join(alteration_markers)
            markers_condition = " OR ".join([f"{col} IN ('{markers_list}')" for col in escaped_status_cols])
            
            # Create SQL for counting altered exams
            count_conditions = []
            for col in escaped_status_cols:
                count_conditions.append(f"CASE WHEN {col} IN ('{markers_list}') THEN 1 ELSE 0 END")
            count_expression = " + ".join(count_conditions)
            
            # Enhanced query with server-side calculations
            query = f"""
            SELECT *,
                   CASE WHEN ({markers_condition}) THEN true ELSE false END as paciente_com_alteracao,
                   ({count_expression}) as qtde_exames_alterados
            FROM {table_name}
            """
        
        # Execute query
        df_loaded = client.query_df(query)
        
        # Process column names (normalize accents and convert to lowercase)
        original_columns = df_loaded.columns.tolist()
        df_loaded.columns = [normalize_column_name(col) for col in df_loaded.columns]
        
        # Update status_cols to match processed column names
        status_cols_processed = [normalize_column_name(col) for col in status_cols]
        
        return df_loaded, status_cols_processed, len(df_loaded)
        
    except Exception as e:
        st.error(f"Erro ao carregar dados do ClickHouse: {e}")
        return pd.DataFrame(), [], 0

# Prepare ClickHouse client parameters
client_params = {
    'host': ch_host,
    'port': ch_port,
    'username': ch_username,
    'password': ch_password if ch_password else None,
    'database': ch_database
}

try:
    df, status_cols, total_records = load_data_from_clickhouse(client_params, ch_table)
    if df.empty:
        st.error("Nenhum dado foi carregado do ClickHouse. Verifique a configuração da conexão e se a tabela contém dados.")
        st.stop()
    else:
        st.success(f"📊 {total_records} registros carregados com sucesso")
except Exception as e:
    st.error(f"Erro ao conectar ou carregar dados: {e}")
    st.stop()

if not status_cols and not df.empty:
    st.error("Nenhuma coluna de status de exame (terminada em '_status') foi encontrada na tabela. Verifique o schema da tabela. Algumas funcionalidades de insights podem não funcionar como esperado.")

# --- HELPER FUNCTION TO CALCULATE TOP ALTERED EXAMS ---
@st.cache_data
def calculate_top_altered_exams(dataframe, status_column_list, markers):
    if dataframe.empty or not status_column_list:
        return pd.DataFrame(columns=["Exame", "Número de Alterações"])
    alteracoes = collections.Counter()
    for col in status_column_list:
        if col in dataframe.columns: 
            exam_name = col.replace("_status", "").replace("_", " ").title()
            alteracoes[exam_name] += dataframe[col].isin(markers).sum()
    if alteracoes:
        df_top_altered = pd.DataFrame.from_dict(alteracoes, orient="index", columns=["Número de Alterações"])
        df_top_altered = df_top_altered.sort_values(by="Número de Alterações", ascending=False).reset_index().rename(columns={"index": "Exame"})
        return df_top_altered[df_top_altered["Número de Alterações"] > 0] # Filter out exams with 0 alterations
    return pd.DataFrame(columns=["Exame", "Número de Alterações"])

# --- NEW ENTERPRISE ANALYSIS FUNCTIONS ---
import pandas as pd
import numpy as np
from typing import List

@st.cache_data
def calculate_enterprise_risk_analysis(
    dataframe: pd.DataFrame,
    status_column_list: List[str],
    markers: List[str]
) -> pd.DataFrame:
    """
    Ultra-optimized version using pure pandas operations.
    Best for datasets with >100k rows.
    """

    if dataframe.empty or not status_column_list or "contratonome" not in dataframe.columns:
        return pd.DataFrame()

    # Remove registros sem contratonome
    df = dataframe[dataframe["contratonome"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    markers_set = set(markers)
    existing_status_cols = [col for col in status_column_list if col in df.columns]

    # Cria a matriz de alterações de forma vetorizada e eficiente
    alteration_matrix = df[existing_status_cols].isin(markers_set)
    alteration_matrix["contratonome"] = df["contratonome"].values

    # Soma de alterações por empresa
    alteration_sums = alteration_matrix.groupby("contratonome")[existing_status_cols].sum()

    # Exame mais problemático por empresa
    exame_problema_series = alteration_sums.idxmax(axis=1).apply(
        lambda x: x.replace("_status", "").replace("_", " ").title() if pd.notna(x) else "N/A"
    )

    # Agrupamento básico
    empresa_stats = df.groupby("contratonome").agg({
        "contratonome": "size"  # Total de funcionários
    }).rename(columns={"contratonome": "Total Funcionários"})

    # Soma de funcionários com alteração
    if "paciente_com_alteracao" in df.columns:
        empresa_stats["Funcionários com Alteração"] = df.groupby("contratonome")["paciente_com_alteracao"].sum()
    else:
        empresa_stats["Funcionários com Alteração"] = 0

    # Média de exames alterados
    if "qtde_exames_alterados" in df.columns:
        empresa_stats["Média Exames Alterados"] = (
            df.groupby("contratonome")["qtde_exames_alterados"]
            .mean().fillna(0).round(2)
        )
    else:
        empresa_stats["Média Exames Alterados"] = 0.0

    # Cálculo da taxa de alteração
    empresa_stats["Taxa de Alteração (%)"] = (
        empresa_stats["Funcionários com Alteração"] / empresa_stats["Total Funcionários"] * 100
    ).round(1)

    # Exame mais problemático
    empresa_stats["Exame Mais Problemático"] = exame_problema_series

    # Classificação de risco
    empresa_stats["Nível de Risco"] = pd.cut(
        empresa_stats["Taxa de Alteração (%)"],
        bins=[-np.inf, 30, 60, np.inf],
        labels=["Baixo", "Médio", "Alto"],
        right=False
    )

    # Resetando index e ordenando
    result = empresa_stats.reset_index().rename(columns={"contratonome": "Empresa"})
    return result.sort_values("Taxa de Alteração (%)", ascending=False)
    
@st.cache_data
def calculate_product_performance(dataframe, status_column_list, markers):
    """Calculate product/plan performance analysis"""
    if dataframe.empty or not status_column_list or "produtonome" not in dataframe.columns:
        return pd.DataFrame()
    
    product_stats = []
    
    for produto in dataframe["produtonome"].dropna().unique():
        produto_df = dataframe[dataframe["produtonome"] == produto]
        total_beneficiarios = len(produto_df)
        
        if total_beneficiarios == 0:
            continue
            
        beneficiarios_com_alteracao = produto_df["paciente_com_alteracao"].sum() if "paciente_com_alteracao" in produto_df.columns else 0
        taxa_alteracao = (beneficiarios_com_alteracao / total_beneficiarios * 100) if total_beneficiarios > 0 else 0
        
        # Calculate average cost impact (proxy using number of altered exams)
        custo_medio_alteracoes = produto_df["qtde_exames_alterados"].mean() if "qtde_exames_alterados" in produto_df.columns else 0
        
        product_stats.append({
            "Produto/Plano": produto,
            "Total Beneficiários": total_beneficiarios,
            "Taxa de Alteração (%)": round(taxa_alteracao, 1),
            "Média Alterações": round(custo_medio_alteracoes, 2)
        })
    
    return pd.DataFrame(product_stats).sort_values("Taxa de Alteração (%)", ascending=False)

@st.cache_data
def calculate_enterprise_insights_data(data_for_insights, status_cols, alteration_markers):
    """Calculate enterprise-specific insights data"""
    enterprise_insights = {}
    
    if "contratonome" in data_for_insights.columns and not data_for_insights.empty:
        # Enterprise risk analysis
        enterprise_risk_df = calculate_enterprise_risk_analysis(data_for_insights, status_cols, alteration_markers)
        
        if not enterprise_risk_df.empty:
            high_risk_enterprises = enterprise_risk_df[enterprise_risk_df["Nível de Risco"] == "Alto"]
            enterprise_insights["high_risk_count"] = len(high_risk_enterprises)
            enterprise_insights["high_risk_enterprises"] = high_risk_enterprises["Empresa"].head(3).tolist()
            enterprise_insights["avg_enterprise_risk"] = enterprise_risk_df["Taxa de Alteração (%)"].mean()
            enterprise_insights["most_problematic_exams"] = enterprise_risk_df["Exame Mais Problemático"].value_counts().head(3).to_dict()
    
    if "produtonome" in data_for_insights.columns and not data_for_insights.empty:
        # Product performance analysis
        product_perf_df = calculate_product_performance(data_for_insights, status_cols, alteration_markers)
        
        if not product_perf_df.empty:
            enterprise_insights["worst_performing_products"] = product_perf_df.head(3)["Produto/Plano"].tolist()
            enterprise_insights["avg_product_risk"] = product_perf_df["Taxa de Alteração (%)"].mean()
    
    return enterprise_insights

top_alterados_df = calculate_top_altered_exams(df, status_cols, ALTERATION_MARKERS)


def get_summary_metrics_from_clickhouse(client, table_name: str) -> list[dict]:
    """
    Consulta resumo dos pacientes diretamente no ClickHouse, com suporte a nomes de colunas com case e acentos.
    """

    # 1. Obter colunas com _status da tabela, respeitando case
    desc = client.query_df(f"DESCRIBE TABLE {table_name}")
    status_cols = [row["name"] for _, row in desc.iterrows() if row["name"].endswith("_status")]

    # 2. Markers usados para identificar alterações
    markers = ["Acima do VR", "Abaixo do VR", "Alto", "Baixo", "Aumentado", "Diminuído", "Positivo"]

    # 3. Montar array de colunas status com aspas duplas (case-sensitive)
    status_array_expr = f"array({', '.join(f'toString(\"{col}\")' for col in status_cols)})"
    marker_array_expr = "[" + ", ".join(f"'{m}'" for m in markers) + "]"
    status_condition = f"arrayExists(x -> x IN {marker_array_expr}, {status_array_expr})"

    # 4. Query SQL
    query = f"""
    WITH
        total_pacientes AS (
            SELECT COUNT(*) AS total FROM {table_name}
        ),
        pacientes_com_alteracao AS (
            SELECT COUNT(*) AS com_alt
            FROM {table_name}
            WHERE {status_condition}
        ),
        pacientes_multiplas_alteracoes AS (
            SELECT COUNT(*) AS multiplas_alt
            FROM (
                SELECT arrayFilter(x -> x IN {marker_array_expr}, {status_array_expr}) AS alteracoes
                FROM {table_name}
            )
            WHERE length(alteracoes) >= 3
        ),
        idade_stats AS (
            SELECT 
                avgIf(idade, {status_condition}) AS media_idade_com_alt,
                avgIf(idade, NOT {status_condition}) AS media_idade_sem_alt,
                minIf(idade, {status_condition}) AS idade_min_com_alt,
                maxIf(idade, {status_condition}) AS idade_max_com_alt
            FROM {table_name}
        )
    SELECT 
        tp.total,
        pca.com_alt,
        round(pca.com_alt / tp.total * 100, 1) AS perc_com_alt,
        pma.multiplas_alt,
        is.media_idade_com_alt,
        is.media_idade_sem_alt,
        is.idade_min_com_alt,
        is.idade_max_com_alt
    FROM total_pacientes tp
    CROSS JOIN pacientes_com_alteracao pca
    CROSS JOIN pacientes_multiplas_alteracoes pma
    CROSS JOIN idade_stats is
    """

    # 5. Execução e formatação para frontend
    try:
        result = client.query_df(query)
        if result.empty:
            return []

        row = result.iloc[0]
        return [
            {"title": "Taxa de Alteração Geral", "value": f"{row['com_alt']} / {row['total']} ({row['perc_com_alt']}%)", "help": "Pacientes com pelo menos um exame alterado."},
            {"title": "Alerta: ≥3 Alterações", "value": f"{row['multiplas_alt']}", "help": "Pacientes com 3 ou mais exames alterados."},
            {"title": "Idade Média (Com/Sem Alter.)", "value": f"{row['media_idade_com_alt']:.1f}a / {row['media_idade_sem_alt']:.1f}a", "help": "Média de idade entre pacientes com ou sem alteração."},
            {"title": "Alerta Jovem com Alteração", "value": f"{int(row['idade_min_com_alt'])} anos", "help": "Idade do paciente mais jovem com alteração."},
            {"title": "Alerta Idoso com Alteração", "value": f"{int(row['idade_max_com_alt'])} anos", "help": "Idade do paciente mais idoso com alteração."},
        ]

    except Exception as e:
        st.error(f"Erro ao consultar métricas no ClickHouse: {e}")
        return []


# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visão Geral e Filtros Interativos",
    "🤖 Chat Analítico (Dados Filtrados)",
    "📋 Visualização Geral dos Dados com AgGrid",
    "💡 Insights Dinâmicos (IA)"
])


# --- TAB 1: VISÃO GERAL E FILTROS ---
with tab1:
    client = clickhouse_connect.get_client(
    host=ch_host,  # ou IP/host do seu servidor
    port=ch_port,
    username=ch_username,
    password=ch_password,  # ou sua senha se tiver
    database=ch_database)
    st.markdown("## 🩺 Panorama dos Resultados Laboratoriais")
    st.markdown("Explore os dados dos pacientes e filtre por diversos critérios para identificar padrões.")

    with st.container(border=True):
        st.subheader("Resumo Geral dos Pacientes (Dataset Completo)")
        
        # Use existing client - no new connection
        dynamic_insights_global = get_summary_metrics_from_clickhouse(client, ch_table)

        if not top_alterados_df.empty:
            exame_top1 = top_alterados_df.iloc[0]
            top_card = {
        "title": f"Mais Alterado: {exame_top1['Exame']}",
        "value": f"{int(exame_top1['Número de Alterações'])} pacientes",
        "help": "Exame com o maior número de alterações."
    }
            dynamic_insights_global.append(top_card)

        if dynamic_insights_global:
            row1_cols = st.columns(3)
            row2_cols = st.columns(3)

            for i, insight in enumerate(dynamic_insights_global):
                target_col = row1_cols[i] if i < 3 else row2_cols[i-3]
                with target_col:
                    st.markdown(f"""
                    <div class="custom-card">
                        <h5>{insight['title']}</h5>
                        <p class="value">{insight['value']}</p>
                        <p class="help-text">{insight['help']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Não foi possível gerar insights dinâmicos para o dataset completo devido à falta de dados ou colunas necessárias.")

    st.markdown("### 🔬 Filtros Detalhados e Visualização de Dados (Interativo)")

    # Store filter conditions instead of copying dataframes
    filter_conditions = []

    with st.expander("🔬 Ajuste os Filtros para Refinar sua Análise:", expanded=True):
        col_filt1, col_filt2 = st.columns([1, 2])

        with col_filt1:
            if "idade" in df.columns and df["idade"].notna().any():
                # Cache age bounds query - FIXED: Use ch_table instead of hardcoded table name
                @st.cache_data
                def get_age_bounds():
                    return client.query_df(f'SELECT min(idade) AS idade_min, max(idade) AS idade_max FROM {ch_table};')
                
                idade_bounds_df = get_age_bounds()
                idade_min_val = int(idade_bounds_df.iloc[0]["idade_min"])
                idade_max_val = int(idade_bounds_df.iloc[0]["idade_max"])

                if idade_min_val < idade_max_val:
                    faixa_idade = st.slider(
                        "Filtrar por Faixa Etária:",
                        min_value=idade_min_val,
                        max_value=idade_max_val,
                        value=(idade_min_val, idade_max_val),
                        key="tab1_idade_slider"
                    )

                    if faixa_idade != (idade_min_val, idade_max_val):
                        filter_conditions.append(
                            (df["idade"].between(faixa_idade[0], faixa_idade[1])) |
                            (df["idade"].isna())
                        )

                elif idade_min_val == idade_max_val:
                    st.caption(f"Todos os pacientes filtrados têm a mesma idade: {idade_min_val} anos.")
                else:
                    st.caption("Dados de idade inconsistentes para criar o filtro.")
            else:
                st.caption("Filtro de idade indisponível (dados ausentes ou não aplicáveis).")

            if "laboratorio" in df.columns and df["laboratorio"].notna().any():
                # Cache unique laboratories
                @st.cache_data
                def get_unique_labs():
                    return sorted(df["laboratorio"].dropna().unique())
                
                laboratorio_opcoes = get_unique_labs()
                if laboratorio_opcoes:
                    laboratorio_sel = st.radio(
                        "Filtrar por laboratorio:",
                        options=["Todos"] + laboratorio_opcoes,
                        horizontal=True,
                        key="tab1_laboratorio_radio"
                    )
                   
                else:
                    st.caption("Filtro de laboratorio indisponível (sem opções válidas).")
                    laboratorio_sel = "Todos"
            else:
                st.caption("Filtro de laboratorio indisponível (coluna 'laboratorio' ausente ou vazia).")
                laboratorio_sel = "Todos"
            
            #aqui
            if "medico" in df.columns and df["medico"].notna().any():
                # Cache unique laboratories
                @st.cache_data
                def get_unique_labs():
                    return sorted(df["medico"].dropna().unique())
                
                medico_opcoes = get_unique_labs()
                if medico_opcoes:
                    medico_selecionados = st.multiselect(
                        "Filtrar por medico:",
                        options=medico_opcoes,
                        help="Escolha um ou mais médicos para filtrar os dados.",
                        key="tab1_medico_multiselect"
                    )
                    
                    # If no medicos are selected, treat as "Todos"
                    if not medico_selecionados:
                        medico_selecionados = medico_opcoes  # Show all options
                else:
                    st.caption("Filtro de medico indisponível (sem opções válidas).")
                    medico_selecionados = []
            else:
                st.caption("Filtro de medico indisponível (coluna 'medico' ausente ou vazia).")
                medico_selecionados = []

            #aqui

            if "qtde_exames_alterados" in df.columns and df["qtde_exames_alterados"].notna().any() and df["qtde_exames_alterados"].max() > 0:
                min_alt = 0
                max_alt = int(df["qtde_exames_alterados"].max())
                if min_alt < max_alt:
                    num_alteracoes_range = st.slider(
                        "Filtrar por Quantidade de Exames Alterados:",
                        min_value=min_alt,
                        max_value=max_alt,
                        value=(min_alt, max_alt),
                        key="tab1_qtde_alt_slider"
                    )
                    if num_alteracoes_range != (min_alt, max_alt):
                        filter_conditions.append(df["qtde_exames_alterados"].between(num_alteracoes_range[0], num_alteracoes_range[1]))
                elif min_alt == max_alt and min_alt == 0:
                    st.caption("Nenhum paciente no filtro atual possui exames alterados.")
                elif min_alt == max_alt:
                    st.caption(f"Todos os pacientes no filtro atual possuem {min_alt} exame(s) alterado(s).")
            else:
                st.caption("Filtro de quantidade de alterações indisponível.")

        # NEW ENTERPRISE AND PRODUCT FILTERS
        st.markdown("##### 🏢 Filtros Empresariais e de Produtos")
        
        enterprise_col1, enterprise_col2 = st.columns(2)
        
        with enterprise_col1:
            if "contratonome" in df.columns and df["contratonome"].notna().any():
                # Cache unique enterprises
                @st.cache_data
                def get_unique_enterprises():
                    return ["Todas as Empresas"] + sorted(df["contratonome"].dropna().unique())
                
                empresa_opcoes = get_unique_enterprises()
                empresa_sel = st.selectbox(
                    "Filtrar por Empresa:",
                    options=empresa_opcoes,
                    key="tab1_empresa_selectbox"
                )
            else:
                st.caption("Filtro de empresa indisponível (coluna 'contratonome' ausente ou vazia).")
                empresa_sel = "Todas as Empresas"
            
            if "contratotpempresa" in df.columns and df["contratotpempresa"].notna().any():
                tipo_empresa_opcoes = ["Todos os Tipos"] + sorted(df["contratotpempresa"].dropna().unique())
                tipo_empresa_sel = st.selectbox(
                    "Tipo de Contrato:",
                    options=tipo_empresa_opcoes,
                    key="tab1_tipo_empresa_selectbox"
                )
            else:
                st.caption("Filtro de tipo de empresa indisponível.")
                tipo_empresa_sel = "Todos os Tipos"

        with enterprise_col2:
            if "produtonome" in df.columns and df["produtonome"].notna().any():
                # Cache unique products
                @st.cache_data
                def get_unique_products():
                    return ["Todos os Produtos"] + sorted(df["produtonome"].dropna().unique())
                
                produto_opcoes = get_unique_products()
                produto_sel = st.selectbox(
                    "Filtrar por Produto/Plano:",
                    options=produto_opcoes,
                    key="tab1_produto_selectbox"
                )
            else:
                st.caption("Filtro de produto indisponível (coluna 'produtonome' ausente ou vazia).")
                produto_sel = "Todos os Produtos"
            
            if "rede_atend" in df.columns and df["rede_atend"].notna().any():
                rede_opcoes = ["Todas as Redes"] + sorted(df["rede_atend"].dropna().unique())
                rede_sel = st.selectbox(
                    "Rede de Atendimento:",
                    options=rede_opcoes,
                    key="tab1_rede_selectbox"
                )
            else:
                st.caption("Filtro de rede indisponível.")
                rede_sel = "Todas as Redes"

        with col_filt2:
            # Cache exam names transformation
            @st.cache_data
            def get_exam_names():
                return [col.replace("_status", "").replace("_", " ").title() for col in status_cols]
            
            exames_nomes_limpos = get_exam_names()
            exames_selecionados_nomes = st.multiselect(
                "Selecionar Exames Específicos para Filtragem e Análise Visual:",
                options=exames_nomes_limpos,
                help="Escolha um ou mais exames para aplicar filtros de status/valor e para gerar gráficos comparativos (correlação, dispersão).",
                key="tab1_exames_multiselect",
                disabled=not status_cols 
            )
            
            exames_status_selecionados = [status_cols[exames_nomes_limpos.index(nome)] for nome in exames_selecionados_nomes]

            # Apply laboratorio filter early to check exam data validity
            temp_filtered_df = df
            if 'laboratorio_sel' in locals() and laboratorio_sel != "Todos":
                temp_filtered_df = df[df["laboratorio"] == laboratorio_sel]
                
                # Check if any of the selected exam columns have non-null values
                if exames_status_selecionados:
                    exames_base = [col.removesuffix("_status") for col in exames_status_selecionados]
                    has_valid_exam_data = any(
                        temp_filtered_df.get(exame).notna().any()
                        for exame in exames_base
                        if exame in temp_filtered_df.columns
                    )
                    if not has_valid_exam_data:
                        st.warning("O laboratório selecionado não possui dados válidos para os exames escolhidos.")
            #aqui
            temp_filtered_df = df
            if 'medico_selecionados' in locals() and medico_selecionados:
                temp_filtered_df = df[df["medico"].isin(medico_selecionados)]

                
                # Check if any of the selected exam columns have non-null values
                if exames_status_selecionados:
                    exames_base = [col.removesuffix("_status") for col in exames_status_selecionados]
                    has_valid_exam_data = any(
                        temp_filtered_df.get(exame).notna().any()
                        for exame in exames_base
                        if exame in temp_filtered_df.columns
                    )
                    if not has_valid_exam_data:
                        st.warning("O médico selecionado não possui dados válidos para os exames escolhidos.")

            #aqui
        if exames_status_selecionados:
            st.markdown("##### Filtros Avançados por Exame Selecionado:")
            filtros_status = {}
            filtros_valores = {}

            num_cols_filter = min(len(exames_status_selecionados), 3) 
            filter_cols = st.columns(num_cols_filter)
            col_idx = 0

            for exame_status_col in exames_status_selecionados:
                exame_base = exame_status_col.removesuffix("_status")
                exame_display_name = exame_base.replace("_", " ").title()

                with filter_cols[col_idx % num_cols_filter]: 
                    # Get status options from the temp filtered dataframe to account for laboratory filter
                    unique_status_vals = temp_filtered_df[exame_status_col].dropna().unique() if exame_status_col in temp_filtered_df.columns else []
                    if len(unique_status_vals) > 0:
                        status_options = ["Todos"] + list(unique_status_vals)
                        
                        priority_status = ALTERATION_MARKERS
                        remaining_options = [opt for opt in status_options if opt not in ["Todos"] + priority_status]
                        sorted_status_options = ["Todos"] + [p for p in priority_status if p in status_options] + sorted(remaining_options)

                        status_sel = st.radio(
                            f"Status de {exame_display_name}:",
                            options=sorted_status_options,
                            key=f"tab1_{exame_base}_status_radio",
                            horizontal=True 
                        )
                        filtros_status[exame_status_col] = status_sel
                    else:
                        st.caption(f"Status para {exame_display_name} não disponível/variado nos dados filtrados.")

                    if exame_base in temp_filtered_df.columns and pd.api.types.is_numeric_dtype(temp_filtered_df[exame_base]):
                        exam_series = temp_filtered_df[exame_base].dropna()
                        if len(exam_series) > 0: 
                            min_val_exam = float(exam_series.min())
                            max_val_exam = float(exam_series.max())
                            if min_val_exam < max_val_exam:
                                valor_range = st.slider(
                                    f"Intervalo de Valores para {exame_display_name}:",
                                    min_value=min_val_exam,
                                    max_value=max_val_exam,
                                    value=(min_val_exam, max_val_exam),
                                    key=f"tab1_{exame_base}_range_slider"
                                )
                                filtros_valores[exame_base] = valor_range
                            elif min_val_exam == max_val_exam:
                                st.caption(f"Valores de {exame_display_name} são constantes ({min_val_exam}).")
                            else:
                                st.caption(f"Dados de valor para {exame_display_name} inconsistentes.")
                        else:
                            st.caption(f"Valores numéricos para {exame_display_name} não disponíveis para filtro.")
                col_idx += 1

            # Apply exam filters to conditions list
            for exame_status_col in exames_status_selecionados:
                exame_base = exame_status_col.removesuffix("_status")
                status_sel = filtros_status.get(exame_status_col)
                if status_sel and status_sel != "Todos":
                    filter_conditions.append(df[exame_status_col] == status_sel)

                if exame_base in filtros_valores:
                    min_v, max_v = filtros_valores[exame_base]
                    # Get the actual min/max from temp_filtered_df for comparison
                    if exame_base in temp_filtered_df.columns:
                        actual_min = temp_filtered_df[exame_base].dropna().min()
                        actual_max = temp_filtered_df[exame_base].dropna().max()
                        if (min_v, max_v) != (actual_min, actual_max):
                            filter_conditions.append(df[exame_base].between(min_v, max_v))

    # Apply laboratorio filter to conditions
    if 'laboratorio_sel' in locals() and laboratorio_sel != "Todos":
        filter_conditions.append(df["laboratorio"] == laboratorio_sel)

    if 'medico_selecionados' in locals() and medico_selecionados:
        filter_conditions.append(df["medico"].isin(medico_selecionados))


    # Apply enterprise filters to conditions
    if 'empresa_sel' in locals() and empresa_sel != "Todas as Empresas":
        filter_conditions.append(df["contratonome"] == empresa_sel)
    
    if 'tipo_empresa_sel' in locals() and tipo_empresa_sel != "Todos os Tipos":
        filter_conditions.append(df["contratotpempresa"] == tipo_empresa_sel)
    
    if 'produto_sel' in locals() and produto_sel != "Todos os Produtos":
        filter_conditions.append(df["produtonome"] == produto_sel)
    
    if 'rede_sel' in locals() and rede_sel != "Todas as Redes":
        filter_conditions.append(df["rede_atend"] == rede_sel)

    # Apply all filters at once using boolean indexing
    if filter_conditions:
        combined_mask = filter_conditions[0]
        for condition in filter_conditions[1:]:
            combined_mask = combined_mask & condition
        df_filtrado_tab1 = df[combined_mask]  # This creates a view, not a copy
    else:
        df_filtrado_tab1 = df  # Use original dataframe
    
    # Additional filter: if exams are selected, only show rows with non-null values for those exams
    if exames_status_selecionados:
        exam_base_cols = [col.removesuffix("_status") for col in exames_status_selecionados if col.removesuffix("_status") in df_filtrado_tab1.columns]
        if exam_base_cols:
            # Create mask for rows where at least one selected exam has non-null values
            exam_mask = df_filtrado_tab1[exam_base_cols].notna().any(axis=1)
            df_filtrado_tab1 = df_filtrado_tab1[exam_mask]

    with st.container(border=True):
        st.subheader(f"Resultados Filtrados ({len(df_filtrado_tab1)} Pacientes)")
        if df_filtrado_tab1.empty:
            st.warning("Nenhum paciente corresponde aos filtros selecionados.")
        else:
            # Cache column calculation - FIXED: Added caching
            @st.cache_data
            def get_display_columns(selected_exams_tuple):
                colunas_base = ["contratonome", "produtonome", "contratotpempresa", "rede_atend", 
                               "laboratorio", "nome", "codigo_os", "sexo"] 
                if "idade" in df.columns:
                    colunas_base.append("idade")
                if "qtde_exames_alterados" in df.columns:
                    colunas_base.append("qtde_exames_alterados")

                colunas_dinamicas_selecionadas = []
                for exame_status_col in selected_exams_tuple: 
                    exame_base = exame_status_col.removesuffix("_status")
                    if exame_base in df.columns: 
                        colunas_dinamicas_selecionadas.append(exame_base)
                    colunas_dinamicas_selecionadas.append(exame_status_col) 

                colunas_finais_temp = colunas_base + colunas_dinamicas_selecionadas
                colunas_finais = [col for col in colunas_finais_temp if col in df.columns]
                return list(dict.fromkeys(colunas_finais))

            colunas_finais = get_display_columns(tuple(exames_status_selecionados))

            # Limit rows for browser performance
            display_data = df_filtrado_tab1[colunas_finais]
            if len(display_data) > 1000:
               # st.warning(f"Mostrando apenas as primeiras 1000 linhas de {len(display_data)} para melhor performance.")
                display_data = display_data.head(1000)
            
            st.dataframe(display_data, height=300, use_container_width=True)

            st.markdown("##### Análise Visual dos Dados Filtrados")
            
            # Only show visualizations for reasonable dataset sizes
            if len(df_filtrado_tab1) <= 100000:
                viz_col1, viz_col2 = st.columns(2)
                with viz_col1:
                    if not df_filtrado_tab1.empty and "qtde_exames_alterados" in df_filtrado_tab1.columns and df_filtrado_tab1["qtde_exames_alterados"].notna().any():
                        if df_filtrado_tab1["qtde_exames_alterados"].nunique() > 0: 
                            # Sample large datasets for performance
                            sample_size = min(50000, len(df_filtrado_tab1))
                            sample_df = df_filtrado_tab1.sample(n=sample_size) if len(df_filtrado_tab1) > sample_size else df_filtrado_tab1
                            
                            fig_dist_alt = px.histogram(
                                sample_df,
                                x="qtde_exames_alterados",
                                title="Distribuição do Nº de Exames Alterados (Filtrado)",
                                labels={"qtde_exames_alterados": "Quantidade de Exames Alterados por Paciente"},
                                   # --- MODIFICATION START ---
                                # Use 'color' to tell Plotly to differentiate bars by this column's values
                                #color="qtde_exames_alterados",
                                # Use 'color_discrete_map' to assign specific colors to each category
                                color_discrete_map={
                                    "Alto": "#d62728",
                                    "Médio": "#ff7f0e", 
                                    "Baixo": "#2ca02c"
                                },
                                # --- MODIFICATION END ---
                                marginal="rug" 
                            )
                            fig_dist_alt.update_layout(bargap=0.1, yaxis_title="Número de Pacientes")
                            if sample_df["qtde_exames_alterados"].notna().any(): 
                                mean_val = sample_df["qtde_exames_alterados"].mean()
                                fig_dist_alt.add_vline(
                                    x=mean_val,
                                    line_dash="dash",
                                    line_color="firebrick",
                                    annotation_text=f"Média: {mean_val:.1f}",
                                    annotation_position="top right"
                                )
                            st.plotly_chart(fig_dist_alt, use_container_width=True)
                        else:
                            st.info("Não há variação na quantidade de exames alterados nos dados filtrados para exibir o histograma.")
                    else:
                        st.info("Não há dados de quantidade de exames alterados para exibir no histograma (coluna ausente, vazia ou sem variação).")

                with viz_col2:
                    if not df_filtrado_tab1.empty and exames_status_selecionados: 
                        top_alt_filtrado_df = calculate_top_altered_exams(df_filtrado_tab1, exames_status_selecionados, ALTERATION_MARKERS)

                        if not top_alt_filtrado_df.empty:
                            fig_top_alt_filt = px.bar(
                                top_alt_filtrado_df,
                                x="Número de Alterações", 
                                y="Exame",              
                                orientation='h',
                                title="Exames Selecionados Mais Alterados (Filtrado)",
                                labels={"Exame": "Exame", "Número de Alterações": "Número de Pacientes com Alteração"},
                                color_discrete_sequence=["#007f3e"],
                                text_auto=True
                            )
                            fig_top_alt_filt.update_layout(
                                yaxis={'categoryorder':'total ascending'}, 
                                xaxis_title="Número de Pacientes com Alteração"
                            )
                            st.plotly_chart(fig_top_alt_filt, use_container_width=True)
                        else:
                            st.info("Nenhuma alteração nos exames selecionados para o conjunto de dados filtrado.")
                    elif not exames_status_selecionados:
                        st.info("Selecione um ou mais exames nos filtros acima para ver o gráfico de alterações.")
                    else: 
                        st.info("Nenhum dado no filtro atual para exibir alterações dos exames selecionados.")
            else:
                st.info(f"Dataset muito grande ({len(df_filtrado_tab1)} registros). Use mais filtros para habilitar visualizações.")

            st.markdown("---")
            st.markdown("##### Mais Análises Visuais dos Dados Filtrados")

            if len(df_filtrado_tab1) <= 50000:
                if "idade" in df_filtrado_tab1.columns and "paciente_com_alteracao" in df_filtrado_tab1.columns and df_filtrado_tab1["idade"].notna().any():
                    if not df_filtrado_tab1.empty: 
                        df_copy_for_plot = df_filtrado_tab1.copy() 
                        df_copy_for_plot['status_alteracao_label'] = df_copy_for_plot['paciente_com_alteracao'].map({True: 'Com Alteração', False: 'Sem Alteração'})

                        fig_age_alt = px.histogram(
                            df_copy_for_plot.dropna(subset=['idade']), 
                            x="idade",
                            color="status_alteracao_label",
                            title="Distribuição de Idade por Status de Alteração Geral (Filtrado)",
                            labels={"idade": "Idade", "status_alteracao_label": "Status de Alteração"},
                            barmode="overlay", 
                            marginal="box",     
                            color_discrete_map={'Com Alteração': '#d62728', 'Sem Alteração': '#007f3e'}
                        )
                        fig_age_alt.update_layout(yaxis_title="Número de Pacientes")
                        st.plotly_chart(fig_age_alt, use_container_width=True)
                    else:
                        st.caption("Dados filtrados vazios, não é possível exibir a distribuição de idade.")
                elif "idade" not in df_filtrado_tab1.columns:
                    st.caption("Coluna 'idade' não disponível nos dados filtrados para exibir a distribuição por status de alteração.")
                else: 
                    st.caption("Dados insuficientes ('idade' ou 'paciente_com_alteracao') para exibir a distribuição de idade por status de alteração.")

                selected_exam_bases_for_viz = [s.replace("_status", "") for s in exames_status_selecionados]
                numeric_exam_cols_for_viz = []
                if not df_filtrado_tab1.empty:
                    for base_name in selected_exam_bases_for_viz:
                        if base_name in df_filtrado_tab1.columns and pd.api.types.is_numeric_dtype(df_filtrado_tab1[base_name]):
                            if df_filtrado_tab1[base_name].nunique(dropna=True) > 1:
                                numeric_exam_cols_for_viz.append(base_name)

                if len(numeric_exam_cols_for_viz) >= 2:
                    corr_matrix = df_filtrado_tab1[numeric_exam_cols_for_viz].corr()
                    if not corr_matrix.empty and corr_matrix.shape[0] > 1 and corr_matrix.shape[1] > 1: 
                        corr_matrix.columns = [col.replace("_", " ").title() for col in corr_matrix.columns]
                        corr_matrix.index = [idx.replace("_", " ").title() for idx in corr_matrix.index]

                        fig_corr_matrix = px.imshow(
                            corr_matrix,
                            text_auto=True, 
                            aspect="auto",
                            color_continuous_scale='RdBu_r', 
                            title=f"Matriz de Correlação entre Exames Numéricos Selecionados (Filtrado)",
                            labels=dict(color="Correlação"),
                            zmin=-1, zmax=1 
                        )
                        fig_corr_matrix.update_xaxes(side="bottom") 
                        st.plotly_chart(fig_corr_matrix, use_container_width=True)
                    else:
                        st.caption("Não foi possível calcular uma matriz de correlação significativa para os exames numéricos selecionados (pouca variação ou dados insuficientes).")
                elif exames_status_selecionados and len(numeric_exam_cols_for_viz) < 2:
                    st.caption(f"Para a matriz de correlação, selecione pelo menos dois exames com dados numéricos distintos e variados. Encontrados válidos: {len(numeric_exam_cols_for_viz)}.")

                # Scatter plot if exactly two numeric exams are selected
                if len(numeric_exam_cols_for_viz) == 2:
                    exam1_key, exam2_key = numeric_exam_cols_for_viz[0], numeric_exam_cols_for_viz[1]
                    exam1_name = exam1_key.replace("_", " ").title()
                    exam2_name = exam2_key.replace("_", " ").title()

                    df_copy_for_scatter = df_filtrado_tab1.copy()
                    color_option_scatter = None
                    color_discrete_map_scatter = None
                    title_suffix_scatter = ""

                    if "paciente_com_alteracao" in df_copy_for_scatter.columns:
                        df_copy_for_scatter['status_alteracao_label'] = df_copy_for_scatter['paciente_com_alteracao'].map({True: 'Com Alteração', False: 'Sem Alteração'})
                        color_option_scatter = 'status_alteracao_label'
                        color_discrete_map_scatter = {'Com Alteração': '#d62728', 'Sem Alteração': '#007f3e'}
                        title_suffix_scatter = " (Colorido por Status de Alteração Geral)"

                    hover_data_scatter = ['nome'] if 'nome' in df_copy_for_scatter.columns else []
                    if 'idade' in df_copy_for_scatter.columns: hover_data_scatter.append('idade')

                    if not df_copy_for_scatter.dropna(subset=[exam1_key, exam2_key]).empty:
                        fig_scatter = px.scatter(
                            df_copy_for_scatter.dropna(subset=[exam1_key, exam2_key]),
                            x=exam1_key,
                            y=exam2_key,
                            color=color_option_scatter,
                            color_discrete_map=color_discrete_map_scatter,
                            title=f"Relação entre {exam1_name} e {exam2_name}{title_suffix_scatter}",
                            labels={
                                exam1_key: exam1_name,
                                exam2_key: exam2_name,
                                "status_alteracao_label": "Status Geral do Paciente"
                            },
                            marginal_x="box",
                            marginal_y="box",
                            hover_data=hover_data_scatter if hover_data_scatter else None
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.caption(f"Não há dados suficientes para exibir o gráfico de dispersão entre {exam1_name} e {exam2_name} após remover valores ausentes.")
                elif exames_status_selecionados and len(numeric_exam_cols_for_viz) != 2 and len(numeric_exam_cols_for_viz) >= 1:
                    st.caption("Para um gráfico de dispersão, selecione exatamente dois exames com dados numéricos e variados.")
            else:
                st.info(f"Para análises visuais avançadas, use filtros adicionais para reduzir o dataset a menos de 5.000 registros (atual: {len(df_filtrado_tab1)}).")

            # NEW ENTERPRISE RISK ANALYSIS SECTION - OPTIMIZED: Added size limits
            st.markdown("---")
            st.markdown("##### 🏢 Análise de Risco Empresarial")
            
            # PERFORMANCE FIX: Only show enterprise analysis for reasonable dataset sizes
            if len(df_filtrado_tab1) <= 50000 and "contratonome" in df_filtrado_tab1.columns:
                enterprise_risk_df = calculate_enterprise_risk_analysis(df_filtrado_tab1, status_cols, ALTERATION_MARKERS)
                
                if not enterprise_risk_df.empty:
                    risk_col1, risk_col2 = st.columns(2)
                    
                    with risk_col1:
                        st.markdown("**Top 10 Empresas com Maior Risco:**")
                        enterprise_risk_df = enterprise_risk_df[enterprise_risk_df['Total Funcionários']>1]
                        top_risk_enterprises = enterprise_risk_df.head(10)
                        st.dataframe(top_risk_enterprises, height=300, use_container_width=True)
                        
                        # Risk level distribution
                        if len(enterprise_risk_df) > 1:
                            risk_distribution = enterprise_risk_df["Nível de Risco"].value_counts()
                            
                            # Create a proper DataFrame for the pie chart
                            risk_df = pd.DataFrame({
                                'Nível de Risco': risk_distribution.index,
                                'count': risk_distribution.values
                            })
                            
                            # Define the color mapping
                            color_discrete_map = {
                                "Alto": "#d62728",    # Red for high risk
                                "Médio": "#ff7f0e",   # Orange for medium risk  
                                "Baixo": "#2ca02c"    # Green for low risk
                            }
                            
                            fig_risk_dist = px.pie(
                                risk_df,
                                values='count',
                                names='Nível de Risco',
                                title="Distribuição de Níveis de Risco por Empresa",
                                color='Nível de Risco',
                                color_discrete_map=color_discrete_map
                            )
                            
                            # Update layout for better appearance
                            fig_risk_dist.update_traces(textposition='inside', textinfo='percent+label')
                            fig_risk_dist.update_layout(showlegend=True)
                            
                            st.plotly_chart(fig_risk_dist, use_container_width=True)
                    with risk_col2:
                        # Enterprise alteration rate chart
                        if len(enterprise_risk_df) > 1:
                            fig_enterprise_rates = px.bar(
                                enterprise_risk_df.head(15),
                                x="Taxa de Alteração (%)",
                                y="Empresa",
                                orientation='h',
                                title="Taxa de Alteração por Empresa (Top 15)",
                                labels={"Taxa de Alteração (%)": "Taxa de Alteração (%)", "Empresa": "Empresa"},
                                color="Taxa de Alteração (%)",
                                color_continuous_scale="Reds"
                            )
                            fig_enterprise_rates.update_layout(
                                yaxis={'categoryorder':'total ascending'},
                                height=500
                            )
                            st.plotly_chart(fig_enterprise_rates, use_container_width=True)
                
                # Product performance analysis
                st.markdown("##### 📋 Análise de Performance de Produtos/Planos")
                product_perf_df = calculate_product_performance(df_filtrado_tab1, status_cols, ALTERATION_MARKERS)
                
                if not product_perf_df.empty:
                    st.markdown("**Performance dos Produtos/Planos:**")
                    st.dataframe(product_perf_df, height=250, use_container_width=True)
                    
                    if len(product_perf_df) > 1:
                        fig_product_perf = px.scatter(
                            product_perf_df,
                            x="Total Beneficiários",
                            y="Taxa de Alteração (%)",
                            size="Média Alterações",
                            hover_name="Produto/Plano",
                            title="Performance dos Produtos: Volume vs Taxa de Alteração",
                            labels={
                                "Total Beneficiários": "Número de Beneficiários",
                                "Taxa de Alteração (%)": "Taxa de Alteração (%)"
                            }
                        )
                        st.plotly_chart(fig_product_perf, use_container_width=True)
            
            else:
                st.info("Para análise empresarial, use filtros adicionais para reduzir o dataset ou verifique se as colunas empresariais estão disponíveis.")

# --- TAB 2: IA CHAT (FOCUSED ON FILTERED DATA IF AVAILABLE, OR GENERAL DF) ---
with tab2:
    st.markdown("## 🤖 Chat Analítico com IA (Dados Atuais da Aba 1)")
    st.markdown(
        "Faça perguntas sobre os dados **visíveis na Aba 1 (aplicando os filtros)**. A IA pode ajudar a realizar análises, gerar tabelas e gráficos."
    )

    # Performance optimization: Use filtered data efficiently without unnecessary copying
    data_for_tab2_chat = df_filtrado_tab1  # Use view instead of copy for initial check

    if data_for_tab2_chat.empty and not df.empty:
        st.info(
            "Os filtros atuais na Aba 1 resultaram em nenhum dado. O chat abaixo operará sobre o conjunto completo de exames."
        )
        data_for_tab2_chat = df  # Use view instead of copy
    elif data_for_tab2_chat.empty and df.empty:  # Should not happen if file is uploaded
        st.error("Nenhum dado de exames laboratoriais carregado para análise.")
        st.stop()  # Stop if df itself is empty and therefore data_for_tab2_chat is also empty

    with st.container(border=True):
        msgs_tab2 = StreamlitChatMessageHistory(
            key="langchain_unimed_messages_tab2"
        )  # Unique key for this tab's chat
        
        # Initialize session state efficiently
        if "plots_tab2" not in st.session_state:
            st.session_state.plots_tab2 = []
        if "dataframes_tab2" not in st.session_state:
            st.session_state.dataframes_tab2 = []

        # Medical context function to identify available lab tests
        @st.cache_data
        def get_medical_context(data_columns):
            """Extract available lab tests and medical context from data"""
            lab_tests = []
            patient_info = []
            
            # Common lab test patterns
            lab_patterns = [
                'ALBUMINA', 'CREATININA', 'GLICOSE', 'HEMOGLOBINA', 'COLESTEROL', 
                'TRIGLICERÍDEOS', 'UREIA', 'TGO', 'TGP', 'BILIRRUBINA', 'LEUCÓCITOS',
                'HEMÁCIAS', 'PLAQUETAS', 'HEMATÓCRITO', 'VCM', 'HCM', 'CHCM',
                'TSH', 'T4', 'PSA', 'VITAMINA', 'FERRO', 'FERRITINA', 'HbA1c', 'GLICOSE'
            ]
            
            # Patient demographic patterns
            demo_patterns = ['idade', 'sexo', 'data_nascimento', 'convenio', 'medico']
            
            for col in data_columns:
                col_lower = col.lower()
                # Check for lab tests (exclude reference and status columns)
                if any(pattern.lower() in col_lower for pattern in lab_patterns) and not col.endswith(('_ref', '_status')):
                    lab_tests.append(col)
                # Check for patient info
                elif any(pattern in col_lower for pattern in demo_patterns):
                    patient_info.append(col)
            
            return lab_tests, patient_info

        # Cache initial message generation but make it sensitive to data changes
        @st.cache_data
        def get_initial_message(data_shape, data_hash, filtered_empty, using_full_df, available_tests_count):
            """Generate context-aware initial message for medical data"""
            initial_message_tab2 = "Olá! Sou sua assistente especializada em análise de exames laboratoriais. "
            
            if using_full_df and not filtered_empty:  # Using full df because no effective filter
                initial_message_tab2 += f"Estou analisando o **dataset completo** com {data_shape[0]} exames "
                #initial_message_tab2 += f"e {available_tests_count} tipos de testes laboratoriais disponíveis."
            elif using_full_df and filtered_empty:  # Filtered resulted in empty, so using full df
                initial_message_tab2 += f"Os filtros não retornaram dados, então estou analisando o **dataset completo** "
                initial_message_tab2 += f"com {data_shape[0]} exames."
            else:  # Using filtered data
                initial_message_tab2 += f"Estou analisando {data_shape[0]} exames **filtrados** da Aba 1 "
                #initial_message_tab2 += f"com {available_tests_count} tipos de testes disponíveis."
            
           
            
            return initial_message_tab2

        # Clear chat history when filtered data changes to ensure AI responses match current data
        # Create a signature of current data to detect changes
        current_data_signature = {
            'shape': data_for_tab2_chat.shape,
            'using_full_df': data_for_tab2_chat.equals(df),
            'filtered_empty': df_filtrado_tab1.empty,
            'data_hash': hash(str(data_for_tab2_chat.index.tolist()[:10]) + str(data_for_tab2_chat.index.tolist()[-10:]) if len(data_for_tab2_chat) > 0 else '')
        }
        
        # Store current data signature in session state and compare
        if 'tab2_data_signature' not in st.session_state:
            st.session_state.tab2_data_signature = current_data_signature
        elif st.session_state.tab2_data_signature != current_data_signature:
            # Data changed! Clear chat history so AI responds to new filtered data
            st.session_state.tab2_data_signature = current_data_signature
            msgs_tab2.clear()  # Clear chat history
            st.session_state.plots_tab2 = []  # Clear stored plots
            st.session_state.dataframes_tab2 = []  # Clear stored dataframes
            st.info("🔄 Os dados filtrados mudaram. Chat reiniciado para refletir os novos exames.")

        # Get medical context
        available_lab_tests, patient_demographics = get_medical_context(data_for_tab2_chat.columns.tolist())

        if len(msgs_tab2.messages) == 0:
            initial_message = get_initial_message(
                current_data_signature['shape'],
                current_data_signature['data_hash'],
                current_data_signature['filtered_empty'], 
                current_data_signature['using_full_df'],
                len(available_lab_tests)
            )
            msgs_tab2.add_ai_message(initial_message)

        # Medical data validation function
        def validate_medical_response(response, user_question, available_tests, patient_demos):
            """Validate if response is relevant to medical lab data"""
            if not response or not response.get("answer"):
                return False, "Resposta vazia da IA"
            
            answer = response["answer"].lower()
            question = user_question.lower()
            
            # Check if response mentions available lab tests or patient demographics
            mentioned_tests = any(test.lower() in answer for test in available_tests[:10])  # Check first 10 for performance
            mentioned_demographics = any(demo.lower() in answer for demo in patient_demographics)
            mentioned_medical = any(term in answer for term in ['exame', 'laboratori', 'paciente', 'resultado', 'valor', 'referência'])
            
            # Check for irrelevant content
            irrelevant_phrases = [
                'não posso ajudar', 'não tenho informações suficientes', 
                'preciso de mais detalhes', 'não entendo a pergunta',
                'como modelo de linguagem', 'não sou médico'
            ]
            is_irrelevant = any(phrase in answer for phrase in irrelevant_phrases)
            
            # Check if it's too generic
            is_relevant = mentioned_tests or mentioned_demographics or mentioned_medical
            
            if is_irrelevant:
                return False, "Resposta genérica ou irrelevante"
            elif not is_relevant:
                return False, "Resposta não relacionada aos dados médicos disponíveis"
            
            return True, "Resposta válida"

        # Enhanced medical context builder
        def build_medical_context(data, user_question):
            """Build comprehensive medical context for the AI"""
            context_parts = []
            
            # Basic data info
            context_parts.append("=== CONTEXTO MÉDICO ===")
            context_parts.append(f"Você está analisando um dataset de EXAMES LABORATORIAIS com {len(data)} registros.")
            
            # Available lab tests
            if available_lab_tests:
                context_parts.append(f"\nTestes laboratoriais disponíveis ({len(available_lab_tests)} tipos):")
                # Show first 15 most common tests to avoid overwhelming the AI
                main_tests = available_lab_tests[:15]
                context_parts.append(f"Principais: {', '.join(main_tests)}")
                if len(available_lab_tests) > 15:
                    context_parts.append(f"... e mais {len(available_lab_tests) - 15} outros testes")
            
            # Patient demographics available
            if patient_demographics:
                context_parts.append(f"\nDados demográficos disponíveis: {', '.join(patient_demographics)}")
            
            # Age range if available
            if 'idade' in data.columns:
                age_stats = data['idade'].describe()
                context_parts.append(f"\nFaixa etária: {age_stats['min']:.0f} a {age_stats['max']:.0f} anos (média: {age_stats['mean']:.1f})")
            
            # Dataset context
            if current_data_signature['using_full_df']:
                if current_data_signature['filtered_empty']:
                    context_parts.append("\n⚠️  IMPORTANTE: Filtros resultaram em dados vazios, usando dataset COMPLETO.")
                else:
                    context_parts.append("\n📊 Analisando dataset COMPLETO (sem filtros ativos).")
            else:
                context_parts.append(f"\n🔍 Analisando dados FILTRADOS ({len(data)} de {len(df)} registros totais).")
            
            # Instructions for AI
            context_parts.append("\n=== INSTRUÇÕES ===")
            context_parts.append("• Analise APENAS os dados fornecidos")
            context_parts.append("• Para valores laboratoriais, considere as colunas _ref (referência) e _status quando disponíveis")
            context_parts.append("• Se a pergunta não pode ser respondida com os dados disponíveis, seja específico sobre o que está faltando")
            context_parts.append("• Use terminologia médica apropriada mas mantenha linguagem acessível")
            context_parts.append("• Nunca invente resultados ou dados que não estão presentes")
            context_parts.append("• Sempre que a palavra 'alterada' ou 'alterado' o usuário está buscando por exames acima e abaixo do valor de referência")
            
            context_parts.append(f"\n=== PERGUNTA DO USUÁRIO ===")
            context_parts.append(user_question)
            
            return "\n".join(context_parts)

        # Optimize chat history display
        def display_chat_history_tab2():
            for i, msg in enumerate(msgs_tab2.messages):
                with st.chat_message(msg.type):
                    # Performance optimization: Early type checking and string operations
                    if not isinstance(msg.content, str):
                        st.markdown(msg.content)  # Let Streamlit handle non-string content
                        continue
                    
                    # Check for plot content
                    if "PLOT_INDEX_TAB2:" in msg.content:
                        try:
                            parts = msg.content.split("PLOT_INDEX_TAB2:", 1)  # Limit split for performance
                            text_content = parts[0]
                            if text_content.strip():
                                st.markdown(text_content)  # Display text before plot
                            
                            idx = int(parts[1])
                            if 0 <= idx < len(st.session_state.plots_tab2):  # Bounds check
                                st.plotly_chart(
                                    st.session_state.plots_tab2[idx],
                                    use_container_width=True,
                                )
                            else:
                                st.error(f"Índice de gráfico inválido: {idx}")
                        except (IndexError, ValueError, TypeError) as e:
                            st.error(f"Erro ao exibir gráfico: {e}")
                            # Fallback to display text content
                            st.markdown(parts[0] if 'parts' in locals() and parts else msg.content)
                    
                    # Check for dataframe content
                    elif "DATAFRAME_INDEX_TAB2:" in msg.content:
                        try:
                            parts = msg.content.split("DATAFRAME_INDEX_TAB2:", 1)  # Limit split
                            text_content = parts[0]
                            if text_content.strip():
                                st.markdown(text_content)  # Display text before dataframe
                            
                            idx = int(parts[1])
                            if 0 <= idx < len(st.session_state.dataframes_tab2):  # Bounds check
                                st.dataframe(
                                    st.session_state.dataframes_tab2[idx],
                                    use_container_width=True,
                                )
                            else:
                                st.error(f"Índice de DataFrame inválido: {idx}")
                        except (IndexError, ValueError, TypeError) as e:
                            st.error(f"Erro ao exibir DataFrame: {e}")
                            # Fallback to display text content
                            st.markdown(parts[0] if 'parts' in locals() and parts else msg.content)
                    else:
                        st.markdown(msg.content)  # Regular text content

        display_chat_history_tab2()

      

        if not pandas_data_analyst:
            st.error(
                "Agente de IA (Pandas Analyst) não inicializado. Verifique a chave da API."
            )
        elif question_tab2 := st.chat_input(
            "Pergunte sobre os exames... (Ex: 'Quantos pacientes têm glicose alterada?', 'Média de hemoglobina por idade')",
            key="chat_input_tab2",
        ):
            msgs_tab2.add_user_message(question_tab2)
            # Display user message immediately
            with st.chat_message("human"):
                st.markdown(question_tab2)

            with st.spinner("🩺 Analisando os dados laboratoriais..."):
                try:
                    if data_for_tab2_chat.empty:
                        error_msg = "Não há dados laboratoriais para analisar. Verifique seus filtros ou o arquivo carregado."
                        st.error(error_msg)
                        msgs_tab2.add_ai_message(
                            "Não há exames laboratoriais para analisar. Por favor, verifique os filtros ou o arquivo carregado."
                        )
                        st.rerun()
                    else:
                        # Build comprehensive medical context
                        medical_context = build_medical_context(data_for_tab2_chat, question_tab2)
                        
                        # Performance optimization: Only copy data when sending to agent
                        # This delays the expensive copy operation until absolutely necessary
                        pandas_data_analyst.invoke_agent(
                            user_instructions=medical_context,
                            data_raw=data_for_tab2_chat.copy(),  # Copy only here when needed
                        )
                        result_tab2 = pandas_data_analyst.get_response()
                        
                        # Validate response relevance
                        is_valid, validation_msg = validate_medical_response(
                            result_tab2, question_tab2, available_lab_tests, patient_demographics
                        )
                        
                           
                except Exception as e:
                    error_msg = f"Erro ao processar com IA: {e}"
                    st.error(error_msg)
                    msgs_tab2.add_ai_message(
                        f"Desculpe, ocorreu um erro durante a análise dos exames: {str(e)[:500]}"  # Truncate long errors
                    )
                    st.rerun()
                    st.stop()

                # Initialize response message
                ai_response_message_tab2 = ""
                if result_tab2 and result_tab2.get("answer"):
                    ai_response_message_tab2 += result_tab2.get("answer")

                # Handle Plotly graph in response
                if (
                    result_tab2
                    and result_tab2.get("routing_preprocessor_decision") == "chart"
                    and result_tab2.get("plotly_graph")
                ):
                    try:
                        # Performance optimization: Handle different plot types efficiently
                        plotly_data = result_tab2.get("plotly_graph")
                        if isinstance(plotly_data, dict):  # If it's a JSON dict for Plotly
                            plot_tab2 = go.Figure(plotly_data)
                        else:  # Assuming it's already a Plotly Figure object
                            plot_tab2 = plotly_data

                        # Store plot and create reference
                        idx_tab2 = len(st.session_state.plots_tab2)
                        st.session_state.plots_tab2.append(plot_tab2)
                        ai_response_message_tab2 += f"\nPLOT_INDEX_TAB2:{idx_tab2}"
                        msgs_tab2.add_ai_message(ai_response_message_tab2)
                        st.rerun()
                    except Exception as e:
                        error_msg_tab2 = f"Erro ao gerar gráfico: {e}. A IA tentou criar uma visualização, mas falhou."
                        if not ai_response_message_tab2.strip():
                            ai_response_message_tab2 = "Análise realizada, mas sem resposta textual da IA.\n"
                        ai_response_message_tab2 += f"\n\n{error_msg_tab2}"
                        msgs_tab2.add_ai_message(ai_response_message_tab2)
                        st.rerun()

                # Handle DataFrame in response
                elif result_tab2 and result_tab2.get("data_wrangled") is not None:
                    data_wrangled_tab2 = result_tab2.get("data_wrangled")
                    
                    # Performance optimization: Efficient DataFrame handling
                    if not isinstance(data_wrangled_tab2, pd.DataFrame):
                        try:
                            # Attempt to convert if it's list of dicts or similar
                            data_wrangled_tab2 = pd.DataFrame(data_wrangled_tab2)
                        except Exception as e:
                            error_msg_tab2 = f"Erro ao converter para DataFrame: {e}. A IA tentou retornar uma tabela, mas falhou."
                            if not ai_response_message_tab2.strip():
                                ai_response_message_tab2 = "Análise realizada, mas sem resposta textual da IA.\n"
                            ai_response_message_tab2 += f"\n\n{error_msg_tab2}"
                            msgs_tab2.add_ai_message(ai_response_message_tab2)
                            st.rerun()
                            st.stop()

                    # Store DataFrame and create reference
                    if isinstance(data_wrangled_tab2, pd.DataFrame):
                        idx_tab2 = len(st.session_state.dataframes_tab2)
                        st.session_state.dataframes_tab2.append(data_wrangled_tab2)
                        ai_response_message_tab2 += f"\nDATAFRAME_INDEX_TAB2:{idx_tab2}"
                        msgs_tab2.add_ai_message(ai_response_message_tab2)
                        st.rerun()
                else:
                    # Handle text-only responses
                    if not (result_tab2 and ai_response_message_tab2.strip()):
                        ai_response_message_tab2 = "A IA processou os dados laboratoriais, mas não retornou uma resposta específica. "
                        ai_response_message_tab2 += (
                            f"Resposta bruta da IA: {str(result_tab2)[:300]}..."
                            if result_tab2
                            else "Nenhuma resposta da IA."
                        )
                    msgs_tab2.add_ai_message(ai_response_message_tab2)
                    st.rerun()

with tab3:
    st.markdown("## 📋 Visualização Geral dos Dados com AgGrid")
 
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

    if df.empty:
        st.warning(
            "Os dados não foram carregados corretamente. Por favor, envie um arquivo válido na barra lateral."
        )
    else:
        # Backend pagination to avoid loading large datasets into memory
        # This preserves Tab 1 performance while making Tab 3 efficient
        
        # Check if we should use pagination (for large datasets)
        use_pagination = len(df) > 2000
        
        if use_pagination:
            # Backend pagination setup
            records_per_page = 1000
            
            # Get total count from ClickHouse
            @st.cache_data
            def get_total_records():
                count_result = client.query_df(f"SELECT COUNT(*) as total FROM {ch_table}")
                return int(count_result.iloc[0]['total'])  # Convert to int to avoid numpy.float64 issues
            
            total_records = get_total_records()
            total_pages = int((total_records + records_per_page - 1) // records_per_page)  # Ensure integer
            
            # Simple page navigation (keep minimal UI change)
            page_number = st.selectbox(
                f"Página (Total: {total_records:,} registros em {total_pages} páginas):",
                options=list(range(1, total_pages + 1)),
                index=0
            )
            
            # Calculate offset for SQL query
            offset = (page_number - 1) * records_per_page
            
            # Load data for current page using raw SQL
            @st.cache_data
            def load_page_data(page_offset, page_limit):
                query = f"""
                SELECT * FROM {ch_table}
                ORDER BY codigo_os
                LIMIT {page_limit} OFFSET {page_offset}
                """
                page_df = client.query_df(query)
                
                # Process column names (convert to lowercase and replace spaces with underscores)
                page_df.columns = page_df.columns.str.lower().str.replace(' ', '_')
                
                # Add calculated columns if status columns exist
                status_cols_page = [col for col in page_df.columns if col.endswith("_status")]
                if status_cols_page:
                    page_df["paciente_com_alteracao"] = page_df[status_cols_page].apply(
                        lambda row: row.isin(ALTERATION_MARKERS).any(), axis=1
                    )
                    page_df["qtde_exames_alterados"] = page_df[status_cols_page].apply(
                        lambda row: sum(row.isin(ALTERATION_MARKERS)), axis=1
                    )
                else:
                    page_df["paciente_com_alteracao"] = False
                    page_df["qtde_exames_alterados"] = 0
                
                return page_df
            
            # Load current page data
            display_df = load_page_data(offset, records_per_page)
        else:
            # Use original dataframe for smaller datasets
            display_df = df

        # Original grid building code (unchanged)
        gb = GridOptionsBuilder.from_dataframe(display_df)

        gb.configure_default_column(
            filter=True,
            sortable=True,
            resizable=False,
            editable=False,
            tooltipValueGetter=JsCode("function(params) { return params.value; }"),
        )

        gb.configure_grid_options(
            domLayout="normal",
            pagination=True,
            paginationPageSize=20,
            floatingFilter=True,
            rowHoverHighlight=True,
            suppressRowClickSelection=False,
            rowSelection="multiple",
            defaultExcelExportParams={
                "processCellCallback": JsCode(
                    """
                    function(params) {
                        if (params.value && typeof params.value === 'string' && (params.value.includes('↑') || params.value.includes('↓'))) {
                            return { styleId: params.value.includes('↑') ? 'highlightUp' : 'highlightDown' };
                        }
                        // ADDED: If it's the data_nascimento column, format the date for Excel
                        if (params.column.getColDef().field === 'data_nascimento' && params.value) {
                            try {
                                var date = new Date(params.value);
                                if (!isNaN(date.getTime())) {
                                    let day = date.getDate().toString().padStart(2, '0');
                                    let month = (date.getMonth() + 1).toString().padStart(2, '0');
                                    let year = date.getFullYear();
                                    // Excel usually prefers YYYY-MM-DD or system locale for date recognition
                                    // Alternatively, pass as string dd/mm/yyyy and let Excel try to parse
                                    return month + '/' + day + '/' + year; // Or dd/mm/yyyy as string
                                }
                            } catch(e) { /* ignore if formatting fails */ }
                        }
                        return null;
                    }
                """
                ),
            },
            excelStyles=[
                {"id": "header", "font": {"bold": True}},
                {
                    "id": "highlightUp",
                    "font": {"color": "#FF0000"},
                    "interior": {"color": "#FFCCCC", "pattern": "Solid"},
                },
                {
                    "id": "highlightDown",
                    "font": {"color": "#0000FF"},
                    "interior": {"color": "#CCCCFF", "pattern": "Solid"},
                },
            ],
            localeText={  # Keep your extensive localization
                "page": "Página",
                "more": "Mais",
                "to": "até",
                "of": "de",
                "next": "Próxima",
                "last": "Última",
                "first": "Primeira",
                "previous": "Anterior",
                "loadingOoo": "Carregando...",
                "noRowsToShow": "Nenhum dado para mostrar",
                "filterOoo": "Filtrar...",
                "applyFilter": "Aplicar",
                "equals": "Igual",
                "notEqual": "Diferente",
                "blank": "Vazio",
                "notBlank": "Preenchido",
                "greaterThan": "Maior que",
                "greaterThanOrEqual": "Maior ou igual a",
                "lessThan": "Menor que",
                "lessThanOrEqual": "Menor ou igual a",
                "inRange": "Entre",
                "contains": "Contém",
                "notContains": "Não contém",
                "startsWith": "Começa com",
                "endsWith": "Termina com",
                "andCondition": "E",
                "orCondition": "OU",
                "clearFilter": "Limpar Filtro",
                "resetFilter": "Redefinir Filtro",
                "filterConditions": "Condições",
                "filterValue": "Valor",
                "filterFrom": "De",
                "filterTo": "Até",
                "selectAll": "(Selecionar Tudo)",
                "searchOoo": "Buscar...",
                "noMatches": "Nenhum resultado",
                "group": "Grupo",
                "columns": "Colunas",
                "filters": "Filtros",
                "rowGroupColumns": "Colunas para Agrupar por Linha",
                "rowGroupColumnsEmptyMessage": "Arraste colunas aqui para agrupar",
                "valueColumns": "Colunas de Valor",
                "pivotMode": "Modo Pivô",
                "groups": "Grupos",
                "values": "Valores",
                "pivots": "Pivôs",
                "valueColumnsEmptyMessage": "Arraste colunas aqui para agregar",
                "pivotColumnsEmptyMessage": "Arraste aqui para definir colunas pivô",
                "toolPanelButton": "Painel de Ferramentas",
                "noRowsLabel": "Sem dados",
                "pinColumn": "Fixar Coluna",
                "valueAggregation": "Agregação de Valor",
                "autosizeThiscolumn": "Ajustar Esta Coluna",
                "autosizeAllColumns": "Ajustar Todas as Colunas",
                "groupBy": "Agrupar por",
                "ungroupBy": "Desagrupar por",
                "resetColumns": "Redefinir Colunas",
                "expandAll": "Expandir Tudo",
                "collapseAll": "Recolher Tudo",
                "copy": "Copiar",
                "copyWithHeaders": "Copiar com Cabeçalhos",
                "ctrlC": "Ctrl+C",
                "paste": "Colar",
                "ctrlV": "Ctrl+V",
                "export": "Exportar",
                "csvExport": "Exportar para CSV",
                "excelExport": "Exportar para Excel (.xlsx)",
                "sum": "Soma",
                "min": "Mínimo",
                "max": "Máximo",
                "none": "Nenhum",
                "count": "Contagem",
                "avg": "Média",
                "filteredRows": "Linhas Filtradas",
                "selectedRows": "Linhas Selecionadas",
                "totalRows": "Total de Linhas",
                "pinLeft": "Fixar à Esquerda",
                "pinRight": "Fixar à Direita",
                "noPin": "Não Fixar",
                "pivotChartTitle": "Gráfico Pivô",
                "advancedFilterContains": "Contém",
                "advancedFilterNotContains": "Não contém",
                "advancedFilterEquals": "É igual a",
                "advancedFilterNotEqual": "Não é igual a",
                "advancedFilterStartsWith": "Começa com",
                "advancedFilterEndsWith": "Termina com",
            },
        )

        grid_options = gb.build()

        AgGrid(
            display_df,
            gridOptions=grid_options,
            height=700,
            width="100%",
            theme="streamlit",  # or "balham", "alpine", etc.
            update_mode=GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=False,  # Adjust as needed
            # Consider adding enableRangeSelection=True for Excel-like copy/paste
            sideBar={
                "toolPanels": [
                    {
                        "id": "columns",
                        "labelDefault": "Colunas",
                        "labelKey": "columns",
                        "iconKey": "columns",
                        "toolPanel": "agColumnsToolPanel",
                        "toolPanelParams": {
                            "suppressRowGroups": False,
                            "suppressValues": False,
                            "suppressPivots": False,
                            "suppressPivotMode": False,
                        },
                    },
                    {
                        "id": "filters",
                        "labelDefault": "Filtros",
                        "labelKey": "filters",
                        "iconKey": "filter",
                        "toolPanel": "agFiltersToolPanel",
                    },
                ],
                "defaultToolPanel": "columns",
            },
        )

with tab4:
    st.markdown("## 💡 Insights de Negócios Dinâmicos (Gerados por IA)")
    st.markdown(
        "Obtenha uma análise estratégica com base nos **dados filtrados na Aba 1**."
    )
    st.markdown("---")

    # Performance optimization: Use filtered data from Tab 1 efficiently
    # This ensures we work with the already optimized df_filtrado_tab1
    data_for_insights = df_filtrado_tab1  # Use view instead of copy() for better performance
    
    # Cache expensive comparison to avoid recalculating on every interaction
    @st.cache_data
    def check_if_filtered(filtered_len, original_len):
        return filtered_len != original_len
    
    is_filtered = check_if_filtered(len(data_for_insights), len(df))
    num_pac_insights = len(data_for_insights)

    st.markdown(
        f"**Análise Atual Baseada em:** `{num_pac_insights} paciente(s)` (dados conforme Aba 1)."
    )
    if is_filtered and num_pac_insights < len(df):
        st.caption(
            f"Dataset original completo continha `{len(df)}` pacientes. Os insights abaixo são para o subconjunto filtrado."
        )
    elif not is_filtered and num_pac_insights > 0:
        st.caption(
            "Analisando o conjunto de dados completo (nenhum filtro ativo ou filtros não alteraram o conjunto)."
        )
    elif num_pac_insights == 0 and not df.empty:  # Filters resulted in zero patients
        st.warning(
            "Os filtros atuais na Aba 1 resultaram em nenhum paciente. Não é possível gerar insights para este subconjunto."
        )
    elif df.empty:  # Original df is empty
        st.warning(
            "Nenhum dado carregado. Por favor, faça o upload de um arquivo CSV para gerar insights."
        )

    if "dyn_biz_insights" not in st.session_state:
        st.session_state.dyn_biz_insights = ""
    if "gen_dyn_insights" not in st.session_state:
        st.session_state.gen_dyn_insights = False

    if not llm:
        st.error(
            "O modelo de IA não foi inicializado. Verifique sua chave da API na barra lateral."
        )
    elif not df.empty:  # Only show button if there's data to potentially analyze
        if st.button(
            "🔍 Gerar Novos Insights (Dados Atuais da Aba 1)",
            key="gen_dyn_biz_insights_btn",
            disabled=st.session_state.gen_dyn_insights,
            use_container_width=True,
        ):
            if data_for_insights.empty:
                st.error(
                    "Não há dados para os filtros atuais. Ajuste os filtros na Aba 1 para gerar insights."
                )
                st.session_state.dyn_biz_insights = ""  # Clear any previous insights
                st.session_state.gen_dyn_insights = False  # Reset button state
            else:
                st.session_state.gen_dyn_insights = True
                st.session_state.dyn_biz_insights = ""  # Clear previous insights
                with st.spinner(
                    "🧠 A Inteligência Artificial está analisando os dados selecionados e elaborando os insights... Por favor, aguarde."
                ):
                    try:
                        # Performance optimization: Pre-calculate and cache expensive computations
                        # Cache based on actual data content to ensure insights update when filters change
                        @st.cache_data
                        def calculate_insights_data(data_signature, num_patients):
                            # Only recalculate if filtered data actually changed
                            num_cols_ins = len(data_for_insights.columns)

                            # Use status_cols which is globally available and reflects columns ending with _status from the uploaded file
                            ls_stat_cols_str = ", ".join(
                                [
                                    s.replace("_status", "").replace("_", " ").title()
                                    for s in status_cols[:10]
                                ]
                            ) + ("..." if len(status_cols) > 10 else "")

                            pac_alt_ins, pc_pac_alt_ins = 0, 0.0
                            if "paciente_com_alteracao" in data_for_insights.columns:
                                pac_alt_ins = int(data_for_insights["paciente_com_alteracao"].sum())
                                pc_pac_alt_ins = (
                                    (pac_alt_ins / num_patients * 100)
                                    if num_patients > 0
                                    else 0.0
                                )

                            med_ex_alt_geral, med_ex_alt_com_alt = 0.0, 0.0
                            if (
                                "qtde_exames_alterados" in data_for_insights.columns
                                and data_for_insights["qtde_exames_alterados"].notna().any()
                            ):
                                med_ex_alt_geral = float(data_for_insights["qtde_exames_alterados"].mean())
                                # Calculate mean only for those with alterations and non-NaN qtde_exames_alterados
                                df_pac_alt_ins = data_for_insights[
                                    data_for_insights["paciente_com_alteracao"]
                                    & data_for_insights["qtde_exames_alterados"].notna()
                                ]
                                if not df_pac_alt_ins.empty:
                                    med_ex_alt_com_alt = float(df_pac_alt_ins["qtde_exames_alterados"].mean())

                            # Use the optimized calculate_top_altered_exams function
                            top_alt_df_ins = calculate_top_altered_exams(
                                data_for_insights, status_cols, ALTERATION_MARKERS
                            )
                            top_alt_str_ins = ""
                            if not top_alt_df_ins.empty:
                                for _, r in top_alt_df_ins.head(5).iterrows():
                                    top_alt_str_ins += f"- {r['Exame']}: {r['Número de Alterações']} alterações\n"
                            else:
                                top_alt_str_ins = "Nenhum exame alterado proeminente identificado neste subconjunto de dados."

                            return {
                                'num_cols_ins': num_cols_ins,
                                'ls_stat_cols_str': ls_stat_cols_str,
                                'pac_alt_ins': pac_alt_ins,
                                'pc_pac_alt_ins': pc_pac_alt_ins,
                                'med_ex_alt_geral': med_ex_alt_geral,
                                'med_ex_alt_com_alt': med_ex_alt_com_alt,
                                'top_alt_str_ins': top_alt_str_ins
                            }
                        
                        # Create a comprehensive signature of the filtered data to ensure cache invalidation
                        # when filters change in Tab 1
                        data_signature = {
                            'shape': data_for_insights.shape,
                            'columns': tuple(data_for_insights.columns.tolist()),
                            'index_hash': hash(tuple(data_for_insights.index.tolist())),  # Critical: track which rows
                            'first_few_ids': tuple(data_for_insights.head(5).index.tolist()) if len(data_for_insights) > 0 else (),
                            'last_few_ids': tuple(data_for_insights.tail(5).index.tolist()) if len(data_for_insights) > 0 else ()
                        }
                        
                        # Get cached calculations that will update when filters change
                        insights_data = calculate_insights_data(str(data_signature), num_pac_insights)
                        
                        # Calculate enterprise insights
                        enterprise_insights_data = calculate_enterprise_insights_data(data_for_insights, status_cols, ALTERATION_MARKERS)
                        
                        dataset_desc = (
                            "completo"
                            if not is_filtered
                            else f"filtrado ({num_pac_insights} de {len(df)} pacientes)"
                        )
                        
                        # Build enhanced prompt with enterprise information
                        prompt_dyn = f"""
                        Você é um consultor de negócios sênior para uma cooperativa de saúde como a Unimed, especializado em análise de dados laboratoriais para otimização de gestão e cuidado ao paciente.
                        Sua tarefa é analisar o resumo do conjunto de dados laboratoriais ({dataset_desc}) de {num_pac_insights} pacientes e fornecer de 4 a 6 insights de negócios estratégicos e acionáveis em português do Brasil, com foco especial em análise empresarial e planos de contingência.

                        Resumo dos Dados Analisados ({dataset_desc}):
                        - Número total de pacientes neste conjunto: {num_pac_insights}
                        - Número total de colunas de dados (exames, dados demográficos, etc.): {insights_data['num_cols_ins']}
                        - Principais exames monitorados: {insights_data['ls_stat_cols_str'] if insights_data['ls_stat_cols_str'] else "N/A"}
                        - Pacientes com pelo menos um exame alterado: {insights_data['pac_alt_ins']} ({insights_data['pc_pac_alt_ins']:.1f}%)
                        - Média de exames alterados por paciente (geral): {insights_data['med_ex_alt_geral']:.2f}
                        - Média de exames alterados (apenas pacientes com alteração): {insights_data['med_ex_alt_com_alt']:.2f}

                        ANÁLISE EMPRESARIAL E DE PRODUTOS:
                        """ + (f"""
                        - Empresas de alto risco identificadas: {enterprise_insights_data.get('high_risk_count', 0)} empresas
                        - Empresas mais problemáticas: {', '.join(enterprise_insights_data.get('high_risk_enterprises', [])) if enterprise_insights_data.get('high_risk_enterprises') else 'N/A'}
                        - Taxa média de alteração por empresa: {enterprise_insights_data.get('avg_enterprise_risk', 0):.1f}%
                        - Exames mais problemáticos nas empresas: {enterprise_insights_data.get('most_problematic_exams', {})}
                        - Produtos/planos com pior performance: {', '.join(enterprise_insights_data.get('worst_performing_products', [])) if enterprise_insights_data.get('worst_performing_products') else 'N/A'}
                        - Taxa média de alteração por produto: {enterprise_insights_data.get('avg_product_risk', 0):.1f}%
                        """ if enterprise_insights_data else "- Dados empresariais não disponíveis para análise detalhada") + f"""

                        Top 5 exames com maior número de alterações:
                        {insights_data['top_alt_str_ins'] if insights_data['top_alt_str_ins'].strip() else "   - Dados insuficientes para listar exames problemáticos"}

                        FOQUE ESPECIALMENTE EM:
                        1. **Análise de Risco Empresarial:** Identifique empresas que precisam de intervenção urgente e planos de ação específicos
                        2. **Estratégias de Contingência:** Proponha planos de contingência para empresas de alto risco, incluindo cronograma e responsabilidades
                        3. **Otimização de Produtos/Planos:** Sugira melhorias nos produtos baseadas na performance observada
                        4. **Prevenção e Programas de Saúde:** Desenvolva programas preventivos específicos por empresa/setor
                        5. **Impacto Financeiro:** Estime custos de não-ação e ROI de programas preventivos, não utilize R$ para preservar a formatação
                        6. **Comunicação Estratégica:** Como comunicar riscos para empresas parceiras de forma construtiva

                        Formato da Resposta:
                        **Insights Estratégicos e Planos de Contingência (grupo de {num_pac_insights} pacientes):**

                        * **[Insight Empresarial 1]:** [Análise específica + Plano de ação detalhado com timeline]
                        * **[Plano de Contingência 1]:** [Estratégia específica para empresas de alto risco]
                        * **[Insight de Produto 1]:** [Análise de performance + Recomendações de melhoria]
                        * **[Estratégia Preventiva 1]:** [Programa preventivo específico + métricas de acompanhamento]
                        * **[Impacto Financeiro 1]:** [Análise de custos e benefícios + justificativa para investimento]
                        * **[Recomendação de Comunicação 1]:** [Como abordar empresas parceiras sobre os achados]

                        Seja específico sobre cronogramas, responsabilidades, métricas de sucesso e orçamentos quando relevante.
                        Se o dataset for pequeno, ajuste as recomendações para estudos piloto ou análises mais aprofundadas.
                        Priorize ações que a Unimed pode implementar imediatamente vs. ações de médio/longo prazo.
                        """
                        
                        # Optimize LLM call with timeout and error handling
                        response = llm.invoke(prompt_dyn)
                        st.session_state.dyn_biz_insights = response.content
                        
                    except Exception as e:
                        st.error(f"Ocorreu um erro ao gerar os insights: {str(e)}")
                        st.session_state.dyn_biz_insights = "Não foi possível gerar os insights no momento. Tente novamente."
                    finally:
                        st.session_state.gen_dyn_insights = False
                        st.rerun()  # Rerun to update button state and display insights/error

        if st.session_state.gen_dyn_insights:
            st.info(
                "Gerando insights para os dados selecionados... Este processo pode levar alguns instantes."
            )
        elif st.session_state.dyn_biz_insights:  # If insights have been generated
            with st.container(border=True):
                st.markdown(
                    "#### 🧠 Análise da IA (Baseada nos Filtros Atuais da Aba 1):"
                )
                st.markdown(st.session_state.dyn_biz_insights)
        elif (
            not st.session_state.gen_dyn_insights
            and not st.session_state.dyn_biz_insights
            and not data_for_insights.empty
        ):
            # No insights generated yet, not currently generating, and there's data to analyze
            st.info(
                "Clique no botão acima para que a Inteligência Artificial gere insights de negócios com base nos dados atualmente filtrados na Aba 1."
            )
