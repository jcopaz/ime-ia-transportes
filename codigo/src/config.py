"""
Configuração central do estudo de previsão de falhas de via.

Todos os caminhos e parâmetros do estudo ficam aqui, para que qualquer
mudança (janela, tamanho do trecho, data de corte) seja feita em um lugar só
e fique registrada no código que gerou os resultados do paper.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
# Raiz do projeto (um nível acima de src/)
RAIZ = Path(__file__).resolve().parents[1]

# Dados de entrada ficam em dados/ (não versionados; ver README)
PASTA_DADOS = RAIZ / "dados"

ARQ_FALHAS_VP = PASTA_DADOS / "falhas_via.xlsx"      # registro de falhas de via
ARQ_GATILHOS = PASTA_DADOS / "analises_causa_raiz.xlsx"   # análises de causa raiz
ARQ_NOTAS_VP = PASTA_DADOS / "notas_inspecao.xlsx"   # notas de inspeção de via
ARQ_GEO_KM = PASTA_DADOS / "marcos_km.csv"           # km válidos por ramal (marcos)

# As pastas de dados e tabelas podem ser trocadas por variável de ambiente
# (útil para testar uma versão nova sem sobrescrever os resultados atuais):
#   ML_PROCESSADOS=data/processed_v2 ML_TABELAS=outputs/tabelas_v2
# Caminho relativo é lido a partir da raiz do projeto.
def _pasta(variavel: str, padrao: Path) -> Path:
    valor = os.environ.get(variavel)
    if not valor:
        return padrao
    p = Path(valor)
    return p if p.is_absolute() else RAIZ / p

PASTA_PROCESSADOS = _pasta("ML_PROCESSADOS", RAIZ / "data" / "processed")
PASTA_FIGURAS = RAIZ / "outputs" / "figuras"
PASTA_TABELAS = _pasta("ML_TABELAS", RAIZ / "outputs" / "tabelas")

# --------------------------------------------------------------------------
# Parâmetros do estudo
# --------------------------------------------------------------------------
DATA_REFERENCIA = "2026-10-16"   # "hoje" do estudo: datas depois disso são erro de cadastro
DATA_MIN_NOTAS = "2019-01-01"    # notas antes disso são esparsas demais

TRECHO_KM = 1.0                  # tamanho do trecho (km)
HORIZONTE_DIAS = 90              # alvo: falha nos próximos 90 dias
INICIO_PAINEL = "2023-01-01"     # primeiro mês do painel (início do registro de falhas)

# Divisão temporal: treino usa meses cujo horizonte termina antes do corte
DATA_CORTE_TESTE = "2025-01-01"

TOP_FRACAO = 0.10                # métrica operacional: top 10% dos trechos

SEMENTE = 42

# Avaliação (lidas por src/modelos.py e src/avaliacao.py)
VAL_INICIO = "2024-05-01"           # validação: meses de 2024-05 em diante (antes do corte do teste)
TREINO_SENSIB_INICIO = "2024-01-01" # sensibilidade: treino só com meses t >= esta data
N_BOOT = 2000                       # réplicas do bootstrap por cluster de trecho
N_COMPARACOES_PRIMARIAS = 3         # P1, P2, P3 -> IC de 98,3% (correção de Bonferroni)

# Intervalo válido do deslocamento em km da nota (Market_Dist_Start_2).
# Fora dele (negativo, >= 1 ou vazio) o km da nota é considerado suspeito e a nota é descartada.
KM_DESLOC_VALIDO = (0.0, 1.0)       # [0, 1)

# --------------------------------------------------------------------------
# Domínio
# --------------------------------------------------------------------------
# Siglas de ramal que aparecem com grafias diferentes nas bases. As siglas reais
# são confidenciais: preencha localmente em dados/aliases_ramais.json, por exemplo
# {"AAA": "BBB"}. Sem o arquivo, nenhum alias é aplicado.
def _aliases() -> dict:
    import json
    arq = PASTA_DADOS / "aliases_ramais.json"
    return json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else {}

RAMAIS_ALIASES = _aliases()

# Famílias de anomalia de VP pelo prefixo da codificação SAP
FAMILIAS_VP = {
    "TR": "trilho", "TJ": "junta", "SD": "solda", "AM": "amv",
    "DM": "dormente", "DO": "dormente", "GE": "geometria", "LA": "lastro",
    "FX": "fixacao", "CS": "cota_salvaguarda",
}

PESO_PRIORIDADE = {"1-Muito alta": 4, "2-Alta": 3, "3-Média": 2, "4-Baixa": 1}

# Tipos de inspeção (campo Tipo_atividade_conc) agrupados
TIPOS_INSPECAO = {
    "YRDRONDA": "ronda", "YRIRONDA - INSPETOR": "ronda",
    "YPCPONTO CEGO ULTRASOM": "ultrassom", "YUSULTRASSOM": "ultrassom",
    "YAGATGMS": "geometria_carro", "YTKTRACKSTAR": "geometria_carro", "YTRTROLLER": "geometria_carro",
    "YAVINSPEÇÃO AMV": "amv",
    "YDRINSP DRONE": "drone",
}

# Falhas que a inspeção de via não tem como antecipar (causas externas / outra disciplina)
CODIGOS_EXTERNOS = {
    "QUEDA DE ÁRVORE NO GABARITO DA LINHA", "VEGETAÇÃO EM ALTA TENSÃO", "LINHA ALAGADA",
    "ASSOREAMENTO DE PN", "QUEDA DE BARREIRA NO GABARITO DA LINHA",
    "Risco ou Queda Material no Gab. Linha", "CIRCUITO DE VIA COM OCUP. INDEVIDA",
    "MÁQUINA DE CHAVE INOPERANTE", "MÁQUINA DE CHAVE PERDENDO INDICAÇÃO",
    "DETE DESCARRILAM ALARMADO INDEVIDO", "Problema placa sinalização",
}
# Origens de falha fora do escopo da inspeção de VP. O campo tem dezenas de grafias
# (INTEMPÉRIES / INTEMPERIES / itempéries, "RI - ...", "RELAÇÕES INSTITUCIONAIS - ..."),
# então o texto é normalizado (maiúsculas, sem acento) e comparado com expressões regulares.
ORIGENS_EXTERNAS_PADROES = [
    r"INTEMP|ITEMP",                  # intempéries (inclui erros de digitação)
    r"VANDALISMO",
    r"^RI\b|^RI-|RELACOES INSTITU",   # Relações Institucionais ("RI - ...")
    r"TERCEIROS",
    r"INTERFERENCIA EXTERNA",
    r"SEGURANCA PATRIMONIAL",
    r"FAIXA D",                       # faixa de domínio
    r"REJEITAR",
    r"ELETROELETRONICA",
]

# Pesos padrão do critério de priorização atual, reproduzido para o baseline
ATUAL_MULT_FAMILIA = {
    "trilho": 1.5, "geometria": 1.5, "amv": 1.5, "dormente": 1.2, "lastro": 1.2,
    "junta": 1.0, "solda": 1.0, "cota_salvaguarda": 1.0, "outros": 1.0, "fixacao": 1.0,
}
ATUAL_MULT_TIPO = {"CT": 1.5, "PV": 1.0}
ATUAL_ALPHA = 0.10

# --------------------------------------------------------------------------
# Operacional (src/pontuar.py → app): ação sugerida pela família dominante
# das notas abertas do trecho. O coordenador pode ajustar esta tabela.
# --------------------------------------------------------------------------
ACAO_POR_FAMILIA = {
    "trilho": "Ultrassom dirigido + inspeção de trilho",
    "solda": "Ultrassom + inspeção de soldas",
    "geometria": "Carro controle e correção geométrica",
    "amv": "Inspeção detalhada de AMV",
    "lastro": "Ronda + programar socaria/limpeza de lastro",
    "dormente": "Ronda + programar troca de dormentes",
    "junta": "Inspeção de juntas e talas",
    "fixacao": "Ronda + reaperto/reposição de fixação",
    "outras": "Ronda dirigida no trecho",
}

# Marcos (pátio / entre-pátio por KM)
ARQ_GEO_MARCOS = PASTA_DADOS / "marcos_patios.csv"
ARQ_TONELAGEM = PASTA_DADOS / "tonelagem_por_patio.xlsx"   # tonelagem bruta por pátio e mês

# --------------------------------------------------------------------------
# Rótulos das variáveis para tabelas e figuras do paper
# --------------------------------------------------------------------------
ROTULOS_VARIAVEIS = {
    "falhas_3m": "falhas nos últimos 3 meses",
    "falhas_12m": "falhas nos últimos 12 meses",
    "falhas_36m": "falhas desde jan/2023 (janela até 36 m)",
    "fraturas_12m": "fraturas nos últimos 12 meses",
    "dias_desde_falha": "dias desde última falha (1500 = nenhuma desde jan/2023)",
    "viz_falhas_12m": "falhas nos trechos vizinhos (12 m)",
    "viz_notas_12m": "anomalias nos trechos vizinhos (12 m)",
    "notas_12m": "anomalias nos últimos 12 meses",
    "notas_3m": "anomalias nos últimos 3 meses",
    "alta_prio_12m": "anomalias de alta prioridade (12 m)",
    "anom_trilho_12m": "anomalias de trilho (12 m)",
    "anom_junta_12m": "anomalias de junta (12 m)",
    "anom_solda_12m": "anomalias de solda (12 m)",
    "anom_amv_12m": "anomalias de AMV (12 m)",
    "anom_dormente_12m": "anomalias de dormente (12 m)",
    "anom_geometria_12m": "anomalias de geometria (12 m)",
    "anom_lastro_12m": "anomalias de lastro (12 m)",
    "anom_fixacao_12m": "anomalias de fixação (12 m)",
    "insp_ronda_12m": "anomalias vindas de ronda (12 m)",
    "insp_ultrassom_12m": "anomalias vindas de ultrassom (12 m)",
    "insp_geometria_carro_12m": "anomalias vindas de carro de geometria (12 m)",
    "insp_amv_12m": "anomalias vindas de inspeção de AMV (12 m)",
    "insp_drone_12m": "anomalias vindas de drone (12 m)",
    "abertas": "notas abertas",
    "abertas_alta_prio": "notas abertas de alta prioridade",
    "idade_media_abertas_meses": "idade média das notas abertas (meses)",
    "abertas_mais_12m": "notas abertas há mais de 12 meses",
    "tem_amv": "trecho com AMV",
    "score_atual": "critério atual (aproximado)",
    "mes_do_ano": "mês do ano",
}
