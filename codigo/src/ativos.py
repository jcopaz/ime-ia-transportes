"""
Ativos dentro de cada km: traduz o local de instalação SAP (tplnr) para um nome
legível e monta a tabela que o unifilar usa para dizer por onde começar a inspeção.

Formato real do tplnr: MF-<RAMAL>-<ORIGEM>_<DESTINO>[-<LINHA>[-<ITEM>]].
O modelo prevê o km; esta tabela só ordena o que já existe dentro dele.
"""
import re

import numpy as np
import pandas as pd

THP_DE_VIA = r"^M(?:04|05|07|10)\b"       # justificativas de Via (docs/REGRA_THP.md)
FILAS = {"FILADIR": "fila direita", "FILAESQ": "fila esquerda"}
ESTRUTURAS = {"PTE": "Ponte", "TUN": "Túnel"}


def _sem_zeros(codigo: str) -> str:
    return codigo.lstrip("0") or "0"


def _linha(parte: str) -> str | None:
    """L000002 → 'Linha 2'; L00016A → 'Linha 16A'. Outros formatos (L00DM28): None."""
    m = re.fullmatch(r"L(\d+[A-Z]?)", parte)
    return f"Linha {_sem_zeros(m.group(1))}" if m else None


def _partes(tplnr) -> list[str]:
    return [] if pd.isna(tplnr) or not str(tplnr).strip() else str(tplnr).strip().upper().split("-")


def generico(tplnr) -> bool:
    """True quando o local não aponta um ativo conhecido: só pátio/trecho, ou código de formato desconhecido."""
    p = _partes(tplnr)
    return len(p) <= 3 or nome_ativo(tplnr) == "-".join(p)


def nome_ativo(tplnr) -> str:
    """Nome legível do local; o próprio código quando o formato não é conhecido."""
    p = _partes(tplnr)
    if not p:
        return "Local não informado"
    codigo = "-".join(p)
    if len(p) == 3 and re.fullmatch(r"\w+_\w+", p[2]):
        a, b = p[2].split("_", 1)
        return f"Pátio {a}" if a == b else f"Trecho {a}–{b}"
    if len(p) not in (4, 5):
        return codigo
    ultimo = p[-1]
    if ultimo.startswith("AMV") and len(ultimo) > 3:
        return f"AMV {_sem_zeros(ultimo[3:])}"
    m = re.fullmatch(r"(PTE|TUN)(\d+)", ultimo)
    if m:
        return f"{ESTRUTURAS[m.group(1)]} {_sem_zeros(m.group(2))}"
    if ultimo in FILAS and len(p) == 5 and _linha(p[3]):
        return f"{_linha(p[3])} · {FILAS[ultimo]}"
    if len(p) == 4 and _linha(ultimo):
        return _linha(ultimo)
    return codigo


def _texto(serie: pd.Series) -> pd.Series:
    """Texto limpo; vazio, '-' e 'nan' viram ''."""
    s = serie.astype("string").str.strip().fillna("")
    return s.where(~s.str.lower().isin(["-", "nan", "n/a", "na"]), "")


def _analise(f: pd.DataFrame, fonte: str) -> pd.DataFrame:
    """Causa raiz e cadeia de por quês de uma análise ('campo' ou 'eng')."""
    pqs = pd.concat([_texto(f[f"{fonte}_pq{i}"]) for i in range(1, 6)], axis=1)
    return pd.DataFrame({"causa_raiz": _texto(f[f"{fonte}_causa_raiz"]),
                         "porques": pqs.apply(lambda r: " → ".join(x for x in r if x), axis=1)}, index=f.index)


def tabela_falhas(falhas: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """
    Falhas dos 12 meses antes de t com o que, como e por que falhou. É o que a tela mostra:
    nada disto entra no modelo. Vale a análise da Engenharia quando existe; senão, a de Campo.
    """
    f = falhas[(falhas["data"] >= t - pd.DateOffset(months=12)) & (falhas["data"] < t)].copy()
    campo, eng = _analise(f, "campo"), _analise(f, "eng")
    seis_m_eng = _texto(f["eng_6m"])
    seis_m_campo = _texto(f["causa_6m"]).replace("Sem análise", "")
    tem_eng = eng["causa_raiz"].ne("") | eng["porques"].ne("") | seis_m_eng.ne("")
    tem_campo = campo["causa_raiz"].ne("") | campo["porques"].ne("") | seis_m_campo.ne("")
    d = f[["ramal", "seg", "km", "data", "nota", "defeito", "gatilho", "restricao"]].copy()
    d["defeito"] = _texto(f["defeito"])
    d["ativo"] = f["tplnr"].map(nome_ativo)
    d["texto_nota"] = _texto(f["texto_nota"])
    d["local_fratura"] = _texto(f["local_fratura"])
    d["lado_fratura"] = _texto(f["lado_fratura"])
    d["thp_via_h"] = f["thp_h"].where(_texto(f["justificativa_thp"]).str.contains(THP_DE_VIA), 0).round(1)
    d["analise"] = np.select([tem_eng, tem_campo], ["Engenharia", "Campo"], "")
    d["causa_6m"] = seis_m_eng.where(tem_eng, seis_m_campo)
    d["causa_raiz"] = eng["causa_raiz"].where(tem_eng, campo["causa_raiz"])
    d["porques"] = eng["porques"].where(tem_eng, campo["porques"])
    return d.sort_values(["ramal", "seg", "data"], ascending=[True, True, False])


def tabela_ativos(notas: pd.DataFrame, falhas: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """
    Uma linha por (ramal, km de 1 km, ativo) com nota aberta em t ou falha nos 12 meses
    antes de t. Janelas iguais às do painel: as contagens somam o que o modelo viu no km.
    """
    chave = ["ramal", "seg", "tplnr"]
    ab = notas[(notas["data"] < t) & (notas["data_enc"].isna() | (notas["data_enc"] >= t))]
    fa = falhas[(falhas["data"] >= t - pd.DateOffset(months=12)) & (falhas["data"] < t)]
    ab = ab.assign(tplnr=ab["tplnr"].fillna("")).sort_values("data")
    fa = fa.assign(tplnr=fa["tplnr"].fillna("")).sort_values("data")

    a = ab.groupby(chave).agg(notas_abertas=("nota", "size"), notas_abertas_alta_prio=("alta_prio", "sum"),
                              ultimo_defeito=("texto", "last"))
    f = fa.groupby(chave).agg(falhas_12m_ativo=("nota", "size"), ultima_falha=("data", "max"),
                              defeito_falha=("defeito", "last"))
    km = pd.concat([ab[chave + ["km"]], fa[chave + ["km"]]]).groupby(chave)["km"].median()

    d = a.join(f, how="outer").join(km).reset_index()
    for c in ("notas_abertas", "notas_abertas_alta_prio", "falhas_12m_ativo"):
        d[c] = d[c].fillna(0).astype(int)
    d["ultimo_defeito"] = d["ultimo_defeito"].fillna(d["defeito_falha"]).fillna("")
    d["ultima_falha"] = d["ultima_falha"].dt.date
    d["nome_ativo"] = d["tplnr"].map(nome_ativo)
    d["generico"] = d["tplnr"].map(generico)
    return d[["ramal", "seg", "tplnr", "nome_ativo", "generico", "km", "notas_abertas", "notas_abertas_alta_prio",
              "ultimo_defeito", "falhas_12m_ativo", "ultima_falha"]].sort_values(["ramal", "seg", "km"])
