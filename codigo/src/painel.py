"""
Monta o painel trecho × mês usado pelo modelo.

Cada linha = um trecho de 1 km (ramal + km) em um mês t (dia 1º do mês).
- Variáveis: calculadas SÓ com dados anteriores a t (sem vazamento).
- Alvo: houve falha oficial de VP no trecho em [t, t + 90 dias)?

Também calcula o critério atual (pesos padrão de core/score_engine.py)
para servir de baseline.

Truncamentos dos dados (valem para o paper):
- Histórico de falhas: o registro de falhas começa em jan/2023, então falhas_36m é na prática
  "falhas desde jan/2023" e dias_desde_falha = 1500 significa "nenhuma falha desde jan/2023".
- Notas de inspeção: a base de notas em aberto só traz notas antes de 2023-02 se elas
  tiveram vida longa (ficaram abertas até hoje ou por muito tempo). Notas curtas
  antigas já sumiram da base. Por isso as janelas de notas de 12 meses só ficam
  completas a partir de t = 2024-02; antes disso as contagens são subestimadas.
"""
import numpy as np
import pandas as pd

from . import config as C

FAMILIAS = ["trilho", "junta", "solda", "amv", "dormente", "geometria", "lastro", "fixacao"]
INSPECOES = ["ronda", "ultrassom", "geometria_carro", "amv", "drone"]


def _meses(inicio, fim):
    return pd.date_range(inicio, fim, freq="MS")


def _janela(df, data_col, t, meses):
    """Eventos com data em [t - meses, t)."""
    ini = t - pd.DateOffset(months=meses)
    return df[(df[data_col] >= ini) & (df[data_col] < t)]


def _juntar(base: pd.DataFrame, serie: pd.Series) -> pd.Series:
    """Alinha uma série indexada por (ramal, seg) às linhas de `base`; ausentes = 0."""
    if serie is None or len(serie) == 0:
        return pd.Series(0.0, index=base.index)
    serie = pd.Series(serie.to_numpy(), index=pd.MultiIndex.from_frame(serie.index.to_frame(index=False)))
    chave = pd.MultiIndex.from_frame(base[["ramal", "seg"]])
    return pd.Series(serie.reindex(chave).fillna(0).to_numpy(), index=base.index)


def _vizinhos(base: pd.DataFrame, col: str) -> pd.Series:
    """Soma de `col` nos trechos vizinhos (seg-1 e seg+1) do mesmo ramal e mês."""
    chave = base.set_index(["t", "ramal", "seg"])[col]
    ant = base[["t", "ramal", "seg"]].assign(seg=base["seg"] - 1)
    pos = base[["t", "ramal", "seg"]].assign(seg=base["seg"] + 1)
    v_ant = chave.reindex(pd.MultiIndex.from_frame(ant)).fillna(0).to_numpy()
    v_pos = chave.reindex(pd.MultiIndex.from_frame(pos)).fillna(0).to_numpy()
    return pd.Series(v_ant + v_pos, index=base.index)


def score_atual(abertas: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """
    Reproduz o critério atual com os pesos padrão, somado por trecho:
    peso_prio × mult_família × mult_tipo × (1 + α·anos aberta) × reincidência.
    Reincidência = mesma família repetida no mesmo trecho (aproximação da
    granularidade ramal+pátio+família usada no app).
    """
    a = abertas.copy()
    anos = (t - a["data"]).dt.days.clip(lower=0) / 365.25
    mult_fam = a["familia"].map(C.ATUAL_MULT_FAMILIA).fillna(1.0)
    mult_tipo = a["tipo_nota"].map(C.ATUAL_MULT_TIPO).fillna(1.0)
    n_loc = a.groupby(["ramal", "seg", "familia"])["nota"].transform("count")
    reinc = np.minimum(1 + 0.15 * (n_loc - 1), 3.0)
    a["score"] = a["peso_prio"] * mult_fam * mult_tipo * (1 + C.ATUAL_ALPHA * anos) * reinc
    return a.groupby(["ramal", "seg"])["score"].sum()


def montar_painel(notas: pd.DataFrame, falhas: pd.DataFrame, trechos: pd.DataFrame,
                  meses=None) -> pd.DataFrame:
    """
    `meses=None` → todos os meses com alvo completo (painel do estudo).
    `meses=[t]` → só esses meses (usado em src/pontuar.py para o mês atual;
    nesse caso o alvo fica incompleto e deve ser ignorado).
    """
    if meses is None:
        fim_falhas = falhas["data"].max()
        ultimo_t = (fim_falhas - pd.Timedelta(days=C.HORIZONTE_DIAS)).to_period("M").to_timestamp()
        meses = _meses(C.INICIO_PAINEL, ultimo_t)
    horizonte = pd.Timedelta(days=C.HORIZONTE_DIAS)
    linhas = []

    for t in meses:
        base = trechos[["ramal", "seg"]].copy()
        base["t"] = t
        idx = ["ramal", "seg"]

        # ---------------- Alvo: falha em [t, t+90d) ----------------
        fut = falhas[(falhas["data"] >= t) & (falhas["data"] < t + horizonte)]
        base["n_falhas_futuro"] = _juntar(base, fut.groupby(idx).size())
        base["y"] = (base["n_falhas_futuro"] > 0).astype(int)
        base["thp_futuro_h"] = _juntar(base, fut.groupby(idx)["thp_h"].sum())

        # ---------------- Histórico de falhas (antes de t) ----------------
        for m in (3, 12, 36):
            h = _janela(falhas, "data", t, m).groupby(idx).size()
            base[f"falhas_{m}m"] = _juntar(base, h)
        fr = _janela(falhas, "data", t, 12)
        fr = fr[fr["fratura"] == 1].groupby(idx).size()
        base["fraturas_12m"] = _juntar(base, fr)
        ult = falhas[falhas["data"] < t].groupby(idx)["data"].max()
        dias_ult = (t - ult).dt.days.astype(float)
        dd = _juntar(base, dias_ult + 1)          # +1 para distinguir "0 dias" de "sem falha"
        base["dias_desde_falha"] = (dd - 1).where(dd > 0, 1500).clip(upper=1500)

        # ---------------- Anomalias de inspeção (antes de t) ----------------
        n12 = _janela(notas, "data", t, 12)
        n3 = _janela(notas, "data", t, 3)
        base["notas_12m"] = _juntar(base, n12.groupby(idx).size())
        base["notas_3m"] = _juntar(base, n3.groupby(idx).size())
        base["alta_prio_12m"] = _juntar(base, n12.groupby(idx)["alta_prio"].sum())
        fam = n12.pivot_table(index=idx, columns="familia", values="nota", aggfunc="count")
        for f in FAMILIAS:
            col = fam[f].dropna() if f in fam.columns else None
            base[f"anom_{f}_12m"] = _juntar(base, col)
        insp = n12.pivot_table(index=idx, columns="tipo_insp", values="nota", aggfunc="count")
        for k in INSPECOES:
            col = insp[k].dropna() if k in insp.columns else None
            base[f"insp_{k}_12m"] = _juntar(base, col)

        # Notas abertas em t (abertas antes de t e ainda não encerradas em t)
        ab = notas[(notas["data"] < t) & (notas["data_enc"].isna() | (notas["data_enc"] >= t))]
        base["abertas"] = _juntar(base, ab.groupby(idx).size())
        base["abertas_alta_prio"] = _juntar(base, ab.groupby(idx)["alta_prio"].sum())
        idade = ((t - ab["data"]).dt.days / 30.44).groupby([ab["ramal"], ab["seg"]]).mean()
        idade.index.names = idx
        base["idade_media_abertas_meses"] = _juntar(base, idade)
        base["abertas_mais_12m"] = _juntar(base, ab[(t - ab["data"]).dt.days > 365].groupby(idx).size())

        # Contexto: trecho com AMV (notas de AMV nos últimos 36 meses)
        amv36 = _janela(notas, "data", t, 36)
        amv36 = amv36[amv36["familia"] == "amv"].groupby(idx).size()
        base["tem_amv"] = (_juntar(base, amv36) > 0).astype(int)

        # Baseline: critério atual
        base["score_atual"] = _juntar(base, score_atual(ab, t))

        linhas.append(base)

    p = pd.concat(linhas, ignore_index=True)
    p["viz_falhas_12m"] = _vizinhos(p, "falhas_12m")
    p["viz_notas_12m"] = _vizinhos(p, "notas_12m")
    p["mes_do_ano"] = p["t"].dt.month
    return p


# Todas as variáveis que o modelo usa (mais o alvo e o critério atual) conferidas por força bruta
VARIAVEIS_CHECADAS = (
    ["falhas_3m", "falhas_12m", "falhas_36m", "fraturas_12m", "dias_desde_falha", "viz_falhas_12m",
     "notas_12m", "notas_3m", "alta_prio_12m"]
    + [f"anom_{f}_12m" for f in FAMILIAS]
    + [f"insp_{k}_12m" for k in INSPECOES]
    + ["abertas", "abertas_alta_prio", "idade_media_abertas_meses", "abertas_mais_12m",
       "tem_amv", "viz_notas_12m", "mes_do_ano", "y", "n_falhas_futuro", "thp_futuro_h", "score_atual"]
)


def checar_amostra(p: pd.DataFrame, notas: pd.DataFrame, falhas: pd.DataFrame,
                   n: int = 300, seed: int = 1) -> int:
    """
    Confere, por força bruta, uma amostra de linhas do painel contra as bases brutas.

    Para n linhas sorteadas, recalcula direto das tabelas de notas e falhas TODAS as
    variáveis de VARIAVEIS_CHECADAS (as do modelo, o alvo, o THP futuro e o critério atual).
    Se qualquer valor divergir do painel, levanta erro. Devolve o número de divergências
    (0 se tudo bate). A checagem é amostral: não cobre todas as linhas.
    """
    from .avaliacao import HIST, ANOM   # import local: evita dependência circular no carregamento
    assert set(HIST + ANOM) <= set(VARIAVEIS_CHECADAS), "variável do modelo fora da checagem"
    amostra = p.sample(n, random_state=seed)
    # índices por ramal: cada linha só olha as notas/falhas do próprio ramal
    notas_r = {r: d for r, d in notas.groupby("ramal")}
    falhas_r = {r: d for r, d in falhas.groupby("ramal")}
    vazio_n, vazio_f = notas.iloc[0:0], falhas.iloc[0:0]
    hor = pd.Timedelta(days=C.HORIZONTE_DIAS)
    divergencias = []
    for _, r in amostra.iterrows():
        t, seg = r["t"], r["seg"]
        nr, fr = notas_r.get(r["ramal"], vazio_n), falhas_r.get(r["ramal"], vazio_f)
        # só o trecho e seus vizinhos (seg-1, seg, seg+1), uma vez por linha
        nr = nr[nr["seg"].between(seg - 1, seg + 1)]
        fr = fr[fr["seg"].between(seg - 1, seg + 1)]
        nt, ft = nr[nr["seg"] == seg], fr[fr["seg"] == seg]
        nv, fv = nr[nr["seg"] != seg], fr[fr["seg"] != seg]

        def jan(d, meses):
            return d[(d["data"] >= t - pd.DateOffset(months=meses)) & (d["data"] < t)]

        n12, n3 = jan(nt, 12), jan(nt, 3)
        ab = nt[(nt["data"] < t) & (nt["data_enc"].isna() | (nt["data_enc"] >= t))]
        ante = ft[ft["data"] < t]
        fut = ft[(ft["data"] >= t) & (ft["data"] < t + hor)]
        idade = (t - ab["data"]).dt.days
        esperado = {
            "falhas_3m": len(jan(ft, 3)), "falhas_12m": len(jan(ft, 12)), "falhas_36m": len(jan(ft, 36)),
            "fraturas_12m": int((jan(ft, 12)["fratura"] == 1).sum()),
            "dias_desde_falha": min(1500, (t - ante["data"].max()).days) if len(ante) else 1500,
            "viz_falhas_12m": len(jan(fv, 12)),
            "notas_12m": len(n12), "notas_3m": len(n3), "alta_prio_12m": int(n12["alta_prio"].sum()),
            "abertas": len(ab), "abertas_alta_prio": int(ab["alta_prio"].sum()),
            "idade_media_abertas_meses": float((idade / 30.44).mean()) if len(ab) else 0.0,
            "abertas_mais_12m": int((idade > 365).sum()),
            "tem_amv": int(len(jan(nt, 36)[lambda d: d["familia"] == "amv"]) > 0),
            "viz_notas_12m": len(jan(nv, 12)),
            "mes_do_ano": t.month,
            "y": int(len(fut) > 0), "n_falhas_futuro": len(fut), "thp_futuro_h": float(fut["thp_h"].sum()),
            "score_atual": float(score_atual(ab, t).sum()) if len(ab) else 0.0,
        }
        for f in FAMILIAS:
            esperado[f"anom_{f}_12m"] = int((n12["familia"] == f).sum())
        for k in INSPECOES:
            esperado[f"insp_{k}_12m"] = int((n12["tipo_insp"] == k).sum())
        assert set(esperado) == set(VARIAVEIS_CHECADAS), "checagem incompleta"
        for coluna, valor in esperado.items():
            if abs(r[coluna] - valor) > 1e-6 * max(1.0, abs(valor)):
                divergencias.append((r["ramal"], r["seg"], str(t.date()), coluna, r[coluna], valor))
    if divergencias:
        raise AssertionError(f"{len(divergencias)} divergências no painel, ex.: {divergencias[:5]}")
    return 0
