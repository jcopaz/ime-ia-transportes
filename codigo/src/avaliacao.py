"""
Ferramentas de avaliação usadas por src/modelos.py e src/figuras.py.

Aqui ficam só funções "puras" (não treinam modelo nem gravam arquivo):
  - grupos de variáveis em lista branca e montagem da matriz X;
  - divisão temporal ajuste / validação / treino final / teste;
  - eventos (falhas) a avaliar e captura no top x% do ranking;
  - bootstrap por cluster de trecho para os intervalos de confiança.

Convenção do estudo: no início de cada mês t, todos os trechos são ordenados
pelo score de risco. Duas métricas de captura:
  - captura em 90 d (PRINCIPAL, mesma janela do alvo y): a falha da data d é
    coberta por todo ranking t com t ≤ d < t + 90 d (em geral 3, às vezes 2;
    1 nas bordas). Captura da falha = média, sobre esses rankings, da captura
    esperada do trecho dela no top x%. Cada falha pesa 1.
    Funções: eventos_90, cobertura, captura_90, matriz_captura_90.
  - captura no mês (métrica anterior, hoje sensibilidade): a falha do mês t
    conta só no ranking do início do mês t.
    Funções: eventos, captura, matriz_captura.
"""
import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------
# Constantes (lidas do config.py quando existirem; senão, valor padrão)
# --------------------------------------------------------------------------
VAL_INICIO = getattr(C, "VAL_INICIO", "2024-05-01")        # 1º mês da validação interna
DATA_CORTE_TESTE = getattr(C, "DATA_CORTE_TESTE", "2025-01-01")
HORIZONTE_DIAS = getattr(C, "HORIZONTE_DIAS", 90)
VAL_MESES = 6                                             # validação = 2024-05 a 2024-10

# --------------------------------------------------------------------------
# Grupos de variáveis (lista branca: só entra o que está listado aqui)
# --------------------------------------------------------------------------
HIST = ["falhas_3m", "falhas_12m", "falhas_36m", "fraturas_12m", "dias_desde_falha", "viz_falhas_12m"]
CONTEXTO = ["mes_do_ano"]          # + um indicador (one-hot) por ramal, criado em matriz()
ANOM = ["notas_12m", "notas_3m", "alta_prio_12m",
        "anom_trilho_12m", "anom_junta_12m", "anom_solda_12m", "anom_amv_12m",
        "anom_dormente_12m", "anom_geometria_12m", "anom_lastro_12m", "anom_fixacao_12m",
        "insp_ronda_12m", "insp_ultrassom_12m", "insp_geometria_carro_12m", "insp_amv_12m", "insp_drone_12m",
        "abertas", "abertas_alta_prio", "idade_media_abertas_meses", "abertas_mais_12m",
        "tem_amv", "viz_notas_12m"]
GRUPOS = {"HIST": HIST, "CONTEXTO": CONTEXTO, "ANOM": ANOM}

# Colunas que NUNCA podem virar variável: o alvo, o futuro e o score atual da plataforma atual
PROIBIDAS = {"y", "n_falhas_futuro", "thp_futuro_h", "score_atual"}

# Variantes do modelo (ablação) = combinação de grupos
VARIANTES = {
    "completo": ["HIST", "CONTEXTO", "ANOM"],
    "so_historico": ["HIST", "CONTEXTO"],
    "so_anomalias": ["ANOM", "CONTEXTO"],
}


def matriz(df: pd.DataFrame, ramais: list, grupos: list) -> pd.DataFrame:
    """Monta a matriz X com os grupos pedidos (só colunas que existem no painel)."""
    colunas = []
    for g in grupos:
        colunas += [c for c in GRUPOS[g] if c in df.columns]
    X = df[colunas].astype(float).copy()
    if "CONTEXTO" in grupos:
        for r in ramais:                                   # efeito fixo de ramal
            X[f"ramal_{r}"] = (df["ramal"] == r).astype(float)
    assert not PROIBIDAS & set(X.columns), "variável proibida entrou na matriz"
    return X


# --------------------------------------------------------------------------
# Divisão temporal
# --------------------------------------------------------------------------
def divisoes(p: pd.DataFrame):
    """
    Devolve (ajuste, validacao, treino, teste):
      ajuste    = meses t cujo alvo (t + 90 d) termina até VAL_INICIO;
      validacao = 6 meses a partir de VAL_INICIO (2024-05 a 2024-10);
      treino    = meses t cujo alvo termina até o corte do teste (treino final);
      teste     = meses t a partir do corte (avaliado uma única vez).
    """
    h = pd.Timedelta(days=HORIZONTE_DIAS)
    ini_val = pd.Timestamp(VAL_INICIO)
    fim_val = ini_val + pd.DateOffset(months=VAL_MESES)
    corte = pd.Timestamp(DATA_CORTE_TESTE)

    ajuste = p[p["t"] + h <= ini_val]
    validacao = p[(p["t"] >= ini_val) & (p["t"] < fim_val)]
    treino = p[p["t"] + h <= corte]
    teste = p[p["t"] >= corte]

    # Conferências: nenhum alvo atravessa a fronteira seguinte
    assert ajuste["t"].max() + h <= ini_val
    assert treino["t"].max() + h <= corte
    assert set(validacao.index) <= set(treino.index), "validação deve estar dentro do treino final"
    assert validacao["t"].nunique() == VAL_MESES
    assert teste["t"].min() >= corte
    return ajuste, validacao, treino, teste


def eventos(falhas: pd.DataFrame, meses) -> pd.DataFrame:
    """Falhas cujo mês está em `meses`; cada uma é avaliada no ranking do início do seu mês."""
    e = falhas.copy()
    e["t"] = e["data"].dt.to_period("M").dt.to_timestamp()
    meses = set(pd.to_datetime(list(meses)))
    return e[e["t"].isin(meses)].reset_index(drop=True)


# --------------------------------------------------------------------------
# Captura no top x%
# --------------------------------------------------------------------------
def _posicoes(df: pd.DataFrame, score) -> pd.DataFrame:
    """
    Para cada trecho-mês calcula, dentro do mês:
      acima   = quantos trechos têm score estritamente maior;
      iguais  = quantos têm exatamente o mesmo score (contando ele);
      ordem   = posição com desempate pela ordem das linhas ("first");
      n       = trechos no mês;
      rank_pct= posição média dos empatados / n ("average").
    """
    d = df[["t", "ramal", "seg"]].copy()
    d["score"] = np.asarray(score, dtype=float)
    g = d.groupby("t")["score"]
    d["acima"] = g.rank(ascending=False, method="min") - 1
    d["iguais"] = d.groupby(["t", "score"])["score"].transform("size")
    d["ordem"] = g.rank(ascending=False, method="first")
    d["n"] = g.transform("size")
    d["rank_pct"] = g.rank(ascending=False, method="average", pct=True)
    return d


def _incluido(acima, iguais, ordem, n, frac, desempate):
    """Probabilidade de o trecho estar entre as K = floor(frac·n) vagas do mês."""
    K = np.floor(frac * n + 1e-9)
    if desempate == "first":
        return (ordem <= K).astype(float)
    # fracionário = captura ESPERADA com desempate aleatório:
    # acima do limiar conta 1; empatados no limiar dividem as vagas que sobram
    return np.clip((K - acima) / iguais, 0.0, 1.0)


def captura(df: pd.DataFrame, score, eventos_: pd.DataFrame, frac=C.TOP_FRACAO,
            desempate="fracionario") -> tuple[pd.Series, pd.Series]:
    """
    Devolve (captura, rank_pct) por evento, na ordem de `eventos_`:
      captura  = captura esperada da falha no top `frac` (valor entre 0 e 1);
      rank_pct = posição relativa do trecho no mês (empates pela média).
    Falha em trecho-mês ausente do painel conta como não capturada (rank 1).
    """
    assert desempate in ("fracionario", "first")
    d = _posicoes(df, score)
    d["cap"] = _incluido(d["acima"], d["iguais"], d["ordem"], d["n"], frac, desempate)
    m = eventos_[["t", "ramal", "seg"]].merge(d[["t", "ramal", "seg", "cap", "rank_pct"]],
                                              on=["t", "ramal", "seg"], how="left")
    assert len(m) == len(eventos_)
    return m["cap"].fillna(0.0), m["rank_pct"].fillna(1.0)


def matriz_captura(df: pd.DataFrame, score, eventos_: pd.DataFrame, grade) -> np.ndarray:
    """Captura esperada (fracionária) de cada evento para cada fração da `grade` (eventos × grade)."""
    d = _posicoes(df, score)
    m = eventos_[["t", "ramal", "seg"]].merge(d, on=["t", "ramal", "seg"], how="left")
    falta = m["n"].isna().to_numpy()
    ac, ig, n = (m[c].fillna(1).to_numpy()[:, None] for c in ("acima", "iguais", "n"))
    K = np.floor(np.asarray(grade)[None, :] * n + 1e-9)
    v = np.clip((K - ac) / ig, 0.0, 1.0)
    v[falta, :] = 0.0
    return v


def curva_captura(df: pd.DataFrame, score, eventos_: pd.DataFrame, grade=np.linspace(0, 1, 101)) -> np.ndarray:
    """% esperado de falhas capturadas inspecionando os x% trechos de maior score."""
    return matriz_captura(df, score, eventos_, grade).mean(axis=0)


# --------------------------------------------------------------------------
# Captura em 90 d (métrica principal)
# --------------------------------------------------------------------------
def _meses_contiguos(meses) -> pd.DatetimeIndex:
    m = pd.DatetimeIndex(sorted(pd.to_datetime(pd.unique(pd.Series(list(meses))))))
    assert len(m) and m.equals(pd.date_range(m.min(), m.max(), freq="MS").as_unit(m.unit)), "meses não contíguos"
    return m


def _n_cobrem(datas: pd.Series, meses: pd.DatetimeIndex) -> np.ndarray:
    """Quantos inícios de mês em `meses` caem em (d − 90 d, d], para cada data d."""
    h = pd.Timedelta(days=HORIZONTE_DIAS)
    m = meses.to_numpy()
    d = datas.to_numpy(dtype=m.dtype)
    return np.searchsorted(m, d, side="right") - np.searchsorted(m, d - h.to_timedelta64(), side="right")


def eventos_90(falhas: pd.DataFrame, meses) -> pd.DataFrame:
    """
    Falhas com data em [min(meses), max(meses) + 90 d), ordenadas por (data, nota).
    Acrescenta: t (mês da falha), id_ev (0..n−1), n_rankings (quantos rankings de
    `meses` cobrem a falha) e n_rankings_teorico (inícios de mês em (d − 90 d, d]).
    """
    m = _meses_contiguos(meses)
    h = pd.Timedelta(days=HORIZONTE_DIAS)
    e = falhas[(falhas["data"] >= m.min()) & (falhas["data"] < m.max() + h)]
    e = e.sort_values(["data", "nota"], kind="mergesort").reset_index(drop=True)
    e["t"] = e["data"].dt.to_period("M").dt.to_timestamp()
    e["id_ev"] = np.arange(len(e))
    e["n_rankings"] = _n_cobrem(e["data"], m)
    # todos os inícios de mês possíveis (sem limitar a `meses`)
    todos = pd.date_range(m.min() - pd.DateOffset(months=4), m.max() + pd.DateOffset(months=4), freq="MS")
    e["n_rankings_teorico"] = _n_cobrem(e["data"], todos.as_unit(m.unit))
    assert (e["n_rankings"] >= 1).all() and e["id_ev"].is_unique
    assert (e["n_rankings"] <= e["n_rankings_teorico"]).all()
    return e


def cobertura(eventos_: pd.DataFrame, meses) -> pd.DataFrame:
    """Pares (falha, ranking) com t ≤ data < t + 90 d: colunas id_ev, t, ramal, seg."""
    h = pd.Timedelta(days=HORIZONTE_DIAS)
    pares = []
    for t in sorted(pd.to_datetime(pd.unique(pd.Series(list(meses))))):
        s = eventos_[(eventos_["data"] >= t) & (eventos_["data"] < t + h)]
        pares.append(pd.DataFrame({"id_ev": s["id_ev"].to_numpy(), "t": t,
                                   "ramal": s["ramal"].to_numpy(), "seg": s["seg"].to_numpy()}))
    p = pd.concat(pares, ignore_index=True)
    p["t"] = p["t"].astype(eventos_["t"].dtype)
    return p.sort_values(["id_ev", "t"], kind="mergesort").reset_index(drop=True)


def _pares(df, score, eventos_):
    """Pares (falha, ranking) com as posições do trecho naquele ranking; confere o nº de pares."""
    d = _posicoes(df, score)
    p = cobertura(eventos_, df["t"].unique())
    p["t"] = p["t"].astype(d["t"].dtype)
    m = p.merge(d, on=["t", "ramal", "seg"], how="left")
    assert len(m) == len(p)
    n = m.groupby("id_ev").size().reindex(eventos_["id_ev"]).fillna(0).to_numpy()
    assert (n == eventos_["n_rankings"].to_numpy()).all(), "nº de pares ≠ n_rankings"
    return m


def captura_90(df: pd.DataFrame, score, eventos_: pd.DataFrame, frac=C.TOP_FRACAO,
               desempate="fracionario") -> tuple[pd.Series, pd.Series]:
    """
    Captura em 90 d: (captura, rank_pct) por falha, na ordem de `eventos_`
    (saída de eventos_90), = média sobre os rankings que cobrem a falha.
    Par (falha, ranking) ausente do painel conta como não capturado (rank 1).
    """
    assert desempate in ("fracionario", "first")
    m = _pares(df, score, eventos_)
    m["cap"] = _incluido(m["acima"], m["iguais"], m["ordem"], m["n"], frac, desempate)
    m["cap"] = m["cap"].fillna(0.0)
    m["rank_pct"] = m["rank_pct"].fillna(1.0)
    g = m.groupby("id_ev")[["cap", "rank_pct"]].mean().reindex(eventos_["id_ev"])
    return (pd.Series(g["cap"].to_numpy(), index=eventos_.index),
            pd.Series(g["rank_pct"].to_numpy(), index=eventos_.index))


def matriz_captura_90(df: pd.DataFrame, score, eventos_: pd.DataFrame, grade) -> np.ndarray:
    """Captura em 90 d (fracionária) de cada falha para cada fração da `grade` (eventos × grade)."""
    m = _pares(df, score, eventos_)
    falta = m["n"].isna().to_numpy()
    ac, ig, n = (m[c].fillna(1).to_numpy()[:, None] for c in ("acima", "iguais", "n"))
    K = np.floor(np.asarray(grade)[None, :] * n + 1e-9)
    v = np.clip((K - ac) / ig, 0.0, 1.0)
    v[falta, :] = 0.0
    pos = pd.Index(eventos_["id_ev"]).get_indexer(m["id_ev"])
    soma = np.zeros((len(eventos_), v.shape[1]))
    np.add.at(soma, pos, v)
    return soma / np.bincount(pos, minlength=len(eventos_))[:, None]


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def pesos_bootstrap(grupos, n=getattr(C, "N_BOOT", 2000), seed=C.SEMENTE, n_eventos=None):
    """
    Pesos de reamostragem: para cada réplica, quantas vezes cada cluster foi sorteado
    (multinomial). Com grupos=None cada evento é o seu próprio cluster (bootstrap iid).
    Devolve (códigos do cluster de cada evento, matriz réplicas × clusters).
    """
    if grupos is None:
        codigos = np.arange(n_eventos)
    else:
        codigos = pd.factorize(pd.Series(list(grupos)))[0]
    G = codigos.max() + 1
    rng = np.random.default_rng(seed)
    W = rng.multinomial(G, np.ones(G) / G, size=n).astype(float)
    return codigos, W


def boot(valores: dict, grupos=None, n=getattr(C, "N_BOOT", 2000), nivel=0.95, seed=C.SEMENTE,
         denominador=None) -> pd.DataFrame:
    """
    Bootstrap por cluster (ex.: trecho). Estatística = soma dos valores / soma do
    denominador nos clusters sorteados (denominador padrão = 1 por evento, ou seja,
    a média por evento). TODAS as séries de `valores` usam os MESMOS pesos, então
    passar diferenças (modelo − baseline) dá o IC pareado.
    `valores` pode trazer vetores (eventos) ou matrizes (eventos × k, ex.: curva).
    Devolve uma tabela com estimativa, ic_inf, ic_sup e nível para cada nome.
    """
    n_ev = len(next(iter(valores.values())))
    codigos, W = pesos_bootstrap(grupos, n=n, seed=seed, n_eventos=n_ev)
    G = W.shape[1]
    den = np.ones(n_ev) if denominador is None else np.asarray(denominador, dtype=float)
    soma_den = np.bincount(codigos, weights=den, minlength=G)
    linhas = []
    a = (1 - nivel) / 2
    for nome, v in valores.items():
        v = np.asarray(v, dtype=float)
        v2 = v.reshape(n_ev, -1)
        soma = np.stack([np.bincount(codigos, weights=v2[:, j], minlength=G) for j in range(v2.shape[1])], axis=1)
        est = soma.sum(axis=0) / soma_den.sum()
        reps = (W @ soma) / (W @ soma_den)[:, None]          # réplicas × k
        lo, hi = np.quantile(reps, [a, 1 - a], axis=0)
        if v.ndim == 1:
            linhas.append({"nome": nome, "estimativa": est[0], "ic_inf": lo[0], "ic_sup": hi[0], "nivel": nivel})
        else:
            for j in range(v2.shape[1]):
                linhas.append({"nome": nome, "j": j, "estimativa": est[j], "ic_inf": lo[j], "ic_sup": hi[j],
                               "nivel": nivel})
    return pd.DataFrame(linhas)
