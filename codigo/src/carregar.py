"""
Leitura e limpeza das três bases de Via Permanente.

Cada função devolve um DataFrame padronizado e registra, no dicionário
`qualidade`, quantas linhas foram removidas e por quê. Esse registro vira a
tabela de qualidade de dados do paper ("garbage in, garbage out").
"""
import re
import unicodedata
import numpy as np
import pandas as pd

from . import config as C


def _ramal(tplnr: pd.Series) -> pd.Series:
    """Extrai a sigla do ramal do local de instalação (MF-<RAMAL>-...)."""
    r = tplnr.astype(str).str.upper().str.split("-").str[1].str.strip()
    return r.replace(C.RAMAIS_ALIASES)


def _normalizar_texto(serie: pd.Series) -> pd.Series:
    """Maiúsculas, sem acento e sem espaços nas pontas (para comparar grafias diferentes)."""
    def limpa(x):
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii")
        return x.upper().strip()
    return serie.map(limpa)


# Detalhe da falha (como e por que): só para mostrar na tela, nenhuma variável do modelo usa.
COLS_FALHAS_DETALHE = {"Texto Longo Nota": "texto_nota", "Local da fratura": "local_fratura",
                     "Lado Fratura": "lado_fratura", "Descricao Justificativa Transporte": "justificativa_thp"}
COLS_GATILHOS_DETALHE = {
    "(Campo) Causa Raiz": "campo_causa_raiz", "(Eng) Causa Raiz": "eng_causa_raiz", "(Eng) 6M Nível 1": "eng_6m",
    **{f"(Campo) Por quê ({i})?": f"campo_pq{i}" for i in range(1, 6)},
    **{f"(Eng) Por quê ({i})?": f"eng_pq{i}" for i in range(1, 6)},
}


def carregar_notas(qualidade: dict) -> pd.DataFrame:
    """Notas de inspeção/anomalia de VP (todas as gerências)."""
    n = pd.read_excel(C.ARQ_NOTAS_VP)
    q = {"linhas_lidas": len(n)}

    n = n.rename(columns={
        "Número_da_nota": "nota", "Data_da_nota": "data", "Data_de_encerramento_da_nota": "data_enc",
        "Local_de_instalação_TPLNR": "tplnr", "Codificação_1": "codigo",
        "Texto_referente_à_prioridade": "prioridade", "Tipo_de_nota": "tipo_nota",
        "Tipo_atividade_conc": "tipo_atividade", "Gerencia": "gerencia",
        "Status_Final_ok": "status", "Texto_breve": "texto",
    })
    n["nota"] = pd.to_numeric(n["nota"], errors="coerce")
    n["data"] = pd.to_datetime(n["data"], errors="coerce")
    n["data_enc"] = pd.to_datetime(n["data_enc"], errors="coerce")

    # KM = marco inteiro ("KM372") + deslocamento em km (0.778) -> 372.778
    # O deslocamento só vale se estiver em [0, 1). Vazio, negativo ou >= 1 é erro de
    # cadastro: a nota é marcada como km_suspeito e NÃO recebe o deslocamento
    # (fica no marco inteiro). Quem decide descartar é preparar.py, depois de
    # contar quantas dessas notas caem dentro do universo de trechos.
    km_int = pd.to_numeric(n["Km início"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    desloc = pd.to_numeric(n["Market_Dist_Start_2"], errors="coerce")
    d_min, d_max = C.KM_DESLOC_VALIDO
    desloc_ok = desloc.notna() & (desloc >= d_min) & (desloc < d_max)
    n["km_suspeito"] = ~desloc_ok
    n["km"] = km_int + desloc.where(desloc_ok, 0)
    # km da regra antiga (soma qualquer deslocamento): só serve para contar quantas notas
    # suspeitas a versão anterior do painel colocava dentro do universo (ver preparar.py).
    n["km_legado"] = km_int + desloc.fillna(0)
    q["km_desloc_negativo"] = int((desloc < d_min).sum())
    q["km_desloc_maior_igual_1"] = int((desloc >= d_max).sum())
    q["km_desloc_nan"] = int(desloc.isna().sum())
    q["km_suspeito_total"] = int(n["km_suspeito"].sum())
    n["ramal"] = _ramal(n["tplnr"])

    ref = pd.Timestamp(C.DATA_REFERENCIA)
    futuro = n["data"] > ref
    q["data_no_futuro"] = int(futuro.sum())
    q["data_futura_max"] = str(n.loc[futuro, "data"].max().date()) if futuro.any() else None
    antigo = n["data"] < pd.Timestamp(C.DATA_MIN_NOTAS)
    q["antes_de_2019"] = int(antigo.sum())
    sem_data = n["data"].isna()
    q["sem_data"] = int(sem_data.sum())
    sem_km = n["km"].isna()
    q["sem_km"] = int(sem_km.sum())
    # Encerramento antes da abertura = erro de cadastro: trata como sem encerramento conhecido
    enc_invalido = n["data_enc"] < n["data"]
    q["encerramento_antes_abertura"] = int(enc_invalido.sum())
    n.loc[enc_invalido, "data_enc"] = pd.NaT
    # Encerrada sem data de encerramento: não dá para saber até quando ficou aberta.
    # Tratamos como encerrada no próprio dia (não conta como "aberta" em nenhum mês).
    enc_sem_data = n["data_enc"].isna() & n["status"].eq("Encerrado")
    q["encerrada_sem_data"] = int(enc_sem_data.sum())
    n.loc[enc_sem_data, "data_enc"] = n.loc[enc_sem_data, "data"]

    # Datas de encerramento: mínima e as 5 maiores concentrações num mesmo dia
    # (encerramento em massa costuma ser limpeza de base, não correção de fato).
    q["data_enc_min"] = str(n["data_enc"].min().date()) if n["data_enc"].notna().any() else None
    enc_dia = n["data_enc"].dropna().dt.date.value_counts().head(5)
    q["encerramento_em_massa_top5"] = {str(d): int(v) for d, v in enc_dia.items()}

    n = n[~(futuro | antigo | sem_data | sem_km)].copy()
    dup = n.duplicated("nota") & n["nota"].notna()
    q["duplicadas"] = int(dup.sum())
    n = n[~dup]

    n["familia"] = n["codigo"].astype(str).str[:2].map(C.FAMILIAS_VP).fillna("outros")
    n["peso_prio"] = n["prioridade"].map(C.PESO_PRIORIDADE).fillna(1)
    n["alta_prio"] = n["prioridade"].isin(["1-Muito alta", "2-Alta"]).astype(int)
    n["tipo_insp"] = n["tipo_atividade"].map(C.TIPOS_INSPECAO).fillna("outra")
    q["linhas_finais"] = len(n)
    q["notas_por_ano_abertura"] = {str(a): int(v) for a, v in n["data"].dt.year.value_counts().sort_index().items()}
    qualidade["notas_vp"] = q
    return n[["nota", "data", "data_enc", "status", "ramal", "km", "familia", "peso_prio",
              "alta_prio", "tipo_nota", "tipo_insp", "gerencia", "codigo", "km_suspeito", "km_legado", "tplnr", "texto"]]


def carregar_falhas(qualidade: dict) -> pd.DataFrame:
    """Falhas oficiais de VP = registro de falhas ∩ análises de causa raiz (as que contam no indicador)."""
    f = pd.read_excel(C.ARQ_FALHAS_VP)
    g = pd.read_excel(C.ARQ_GATILHOS)
    q = {"falhas_linhas": len(f), "gatilhos_linhas": len(g)}

    f["nota"] = pd.to_numeric(f["Número da nota"], errors="coerce")
    g["nota"] = pd.to_numeric(g["Número da nota"], errors="coerce")
    # informativa (fora do balanço): notas repetidas no arquivo de gatilhos, antes do merge
    q["gatilhos_notas_duplicadas"] = int(g.duplicated("nota").sum())
    g = g.drop_duplicates("nota")
    cols_g = {"Gatilho": "gatilho", "(Campo) 6M Nível 1": "causa_6m",
              "(Campo) 6M Nível 2": "causa_6m_n2", "(Campo) Expurgo": "expurgo", **COLS_GATILHOS_DETALHE}
    s = f.merge(g[["nota", *cols_g]].rename(columns=cols_g), on="nota", how="inner")
    q["falhas_com_analise"] = len(s)
    # notas repetidas no registro de falhas (uma nota em mais de uma linha): entram no balanço de T1
    n_antes = len(s)
    s = s.drop_duplicates("nota")
    q["notas_duplicadas_falhas"] = n_antes - len(s)

    s = s.rename(columns={"Data da Nota": "data", "Local de instalação TPLNR": "tplnr",
                          "Texto breve para o código": "defeito", "Nova Gerência": "gerencia",
                          "Descrição Tipo de Restrição": "restricao",
                          "Origem de Atividade Correta": "origem", **COLS_FALHAS_DETALHE})
    s["data"] = pd.to_datetime(s["data"], errors="coerce")
    # falha sem data válida não entra em nenhuma janela: descarta e conta (balanço de T1)
    sem_data = s["data"].isna()
    q["sem_data"] = int(sem_data.sum())
    s = s[~sem_data].copy()
    s["km"] = pd.to_numeric(s["KM Início"], errors="coerce")
    s["ramal"] = _ramal(s["tplnr"])
    s["thp_h"] = pd.to_numeric(s["Tempo THP 300 (min)"], errors="coerce").fillna(0) / 60

    externo = s["defeito"].astype(str).str.strip().isin({c.strip() for c in C.CODIGOS_EXTERNOS})
    # Origem externa / outra disciplina: texto normalizado + expressões regulares.
    # Conta quantas falhas casam com cada padrão (um registro pode casar com mais de um).
    origem_norm = _normalizar_texto(s["origem"])
    origem_fora = pd.Series(False, index=s.index)
    por_padrao = {}
    for padrao in C.ORIGENS_EXTERNAS_PADROES:
        casa = origem_norm.str.contains(padrao, regex=True)
        por_padrao[padrao] = int(casa.sum())
        origem_fora |= casa
    q["origem_externa_por_padrao"] = por_padrao
    expurgo = s["expurgo"].eq("Sim")
    sem_km = s["km"].isna()
    q["causa_externa_ou_outra_disciplina"] = int((externo | origem_fora).sum())
    q["expurgadas"] = int((expurgo & ~(externo | origem_fora)).sum())
    q["sem_km"] = int((sem_km & ~(externo | origem_fora | expurgo)).sum())
    s = s[~(externo | origem_fora | expurgo | sem_km)].copy()

    # Garantia: nenhuma origem remanescente casa com os padrões externos
    restante = _normalizar_texto(s["origem"])
    for padrao in C.ORIGENS_EXTERNAS_PADROES:
        assert not restante.str.contains(padrao, regex=True).any(), f"origem externa restante: {padrao}"

    # Duplicatas: a mesma falha registrada em notas diferentes (mesma data, ramal,
    # km inteiro e defeito). Mantém a nota de menor número.
    s["_seg"] = np.floor(s["km"] / C.TRECHO_KM).astype(int)
    s = s.sort_values("nota")
    dup = s.duplicated(["data", "ramal", "_seg", "defeito"], keep="first")
    q["duplicadas_removidas"] = int(dup.sum())
    s = s[~dup].drop(columns="_seg").sort_values(["data", "nota"]).reset_index(drop=True)

    s["fratura"] = s["gatilho"].eq("Fratura").astype(int)
    s["causa_6m"] = s["causa_6m"].fillna("Sem análise")
    q["falhas_finais"] = len(s)
    # Balanço de T1: cada exclusão é exclusiva das demais, então a conta tem que fechar
    esperado = (q["falhas_com_analise"] - q["notas_duplicadas_falhas"] - q["sem_data"]
                - q["causa_externa_ou_outra_disciplina"] - q["expurgadas"] - q["sem_km"]
                - q["duplicadas_removidas"])
    assert esperado == q["falhas_finais"], f"balanço de falhas não fecha: {esperado} != {q['falhas_finais']}"
    assert s["data"].notna().all(), "falha sem data após a limpeza"
    q["periodo"] = f"{s['data'].min().date()} a {s['data'].max().date()}"
    q["primeira_falha"] = str(s["data"].min().date())
    q["falhas_por_mes"] = {str(m): int(v) for m, v in s["data"].dt.to_period("M").value_counts().sort_index().items()}
    qualidade["falhas_vp"] = q
    return s[["nota", "data", "ramal", "km", "gerencia", "defeito", "gatilho", "fratura",
              "causa_6m", "causa_6m_n2", "restricao", "thp_h", "tplnr",
              *COLS_FALHAS_DETALHE.values(), *COLS_GATILHOS_DETALHE.values()]]


def carregar_trechos(qualidade: dict) -> pd.DataFrame:
    """Universo de trechos de 1 km: os KM que existem no arquivo oficial de marcos."""
    g = pd.read_csv(C.ARQ_GEO_KM)
    g["ramal"] = g["ramal"].astype(str).str.upper().replace(C.RAMAIS_ALIASES)
    g["seg"] = np.floor(g["km"] / C.TRECHO_KM).astype(int)
    t = g.groupby(["ramal", "seg"], as_index=False).agg(lat=("lat", "mean"), lon=("lon", "mean"))
    qualidade["trechos"] = {"ramais": int(t["ramal"].nunique()), "trechos_1km": len(t)}
    return t


def no_universo(df: pd.DataFrame, trechos: pd.DataFrame, nome: str, qualidade: dict) -> pd.DataFrame:
    """Mantém só eventos cujo (ramal, km) cai num trecho existente do KMZ."""
    df = df.copy()
    df["seg"] = np.floor(df["km"] / C.TRECHO_KM).astype(int)
    chave = trechos[["ramal", "seg"]].assign(_ok=1)
    m = df.merge(chave, on=["ramal", "seg"], how="left")
    fora = m["_ok"].isna()
    qualidade.setdefault(nome, {})["fora_do_kmz"] = int(fora.sum())
    qualidade[nome]["no_universo"] = int((~fora).sum())
    # Quantos eventos ficaram de fora do KMZ, por ramal (ramais ausentes do arquivo de marcos)
    qualidade[nome]["fora_do_kmz_por_ramal"] = {
        str(r): int(v) for r, v in m.loc[fora, "ramal"].value_counts().items()}
    return m[~fora].drop(columns="_ok")
