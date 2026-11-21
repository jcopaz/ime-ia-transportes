"""
Malha sintética para a demonstração pública.

Nada aqui vem da ferrovia estudada: ramais, pátios, quilômetros, falhas, notas e
tonelagem são gerados com semente fixa. A probabilidade de falha é calculada por
uma função logística com pesos escolhidos para imitar o comportamento descrito
no artigo (histórico de falhas pesa mais; anomalias de trilho e solda pesam nas
falhas de material; tonelagem pesa pouco). Serve só para mostrar como a
priorização aparece para as equipes.
"""
import numpy as np
import pandas as pd

RAMAIS = {"Ramal A": 120, "Ramal B": 85, "Ramal C": 64, "Ramal D": 48, "Ramal E": 36}
FAMILIAS = ["trilho", "solda", "geometria", "amv", "dormente", "lastro"]
ACAO = {
    "trilho": "Ultrassom dirigido + inspeção de trilho",
    "solda": "Ultrassom + inspeção de soldas",
    "geometria": "Carro controle e correção geométrica",
    "amv": "Inspeção detalhada de AMV",
    "dormente": "Ronda + programar troca de dormentes",
    "lastro": "Ronda + programar socaria/limpeza de lastro",
}
# pesos do "modelo" ilustrativo (escala logit)
PESOS = {
    "falhas_36m": 0.30, "viz_falhas_12m": 0.12, "anom_trilho_12m": 0.10, "anom_solda_12m": 0.09,
    "anom_geometria_12m": 0.05, "anom_amv_12m": 0.07, "anom_dormente_12m": 0.03, "anom_lastro_12m": 0.03,
    "notas_abertas": 0.02, "log_tonelagem": 0.25, "tem_amv": 0.35,
}
ROTULOS = {
    "falhas_36m": "falhas no trecho desde 2023", "viz_falhas_12m": "falhas nos trechos vizinhos (12 m)",
    "anom_trilho_12m": "anomalias de trilho (12 m)", "anom_solda_12m": "anomalias de solda (12 m)",
    "anom_geometria_12m": "anomalias de geometria (12 m)", "anom_amv_12m": "anomalias de AMV (12 m)",
    "anom_dormente_12m": "anomalias de dormente (12 m)", "anom_lastro_12m": "anomalias de lastro (12 m)",
    "notas_abertas": "notas em aberto", "log_tonelagem": "tonelagem bruta (12 m)", "tem_amv": "trecho com AMV",
}
INTERCEPTO = -4.1


def _trechos(rng: np.random.Generator) -> pd.DataFrame:
    linhas = []
    for r, n in RAMAIS.items():
        base_ton = rng.uniform(1.0, 3.0)
        pat = 0
        for km in range(n):
            if km % 12 == 0:
                pat += 1
            linhas.append({"ramal": r, "km": km, "patio": f"{r[-1]}{pat:02d}",
                           "tem_amv": int(km % 12 in (0, 1) or rng.random() < .04),
                           "ton_base": base_ton * rng.uniform(.85, 1.15)})
    t = pd.DataFrame(linhas)
    # pontos crônicos: faixas de alguns km com propensão maior
    t["propensao"] = rng.gamma(1.2, .35, len(t))
    for _ in range(14):
        i = rng.integers(len(t))
        for d in range(-2, 3):
            j = i + d
            if 0 <= j < len(t) and t.at[j, "ramal"] == t.at[i, "ramal"]:
                t.at[j, "propensao"] += (3 - abs(d)) * rng.uniform(.4, .9)
    return t


def gerar(meses: int = 12, semente: int = 7) -> pd.DataFrame:
    """Painel sintético trecho × mês com variáveis, probabilidade, motivos e ação."""
    rng = np.random.default_rng(semente)
    t = _trechos(rng)
    blocos = []
    for m in range(meses):
        d = t.copy()
        d["mes"] = pd.Timestamp("2026-01-01") + pd.DateOffset(months=m)
        lam = d["propensao"] * (1 + .15 * np.sin(m / 2))
        d["falhas_36m"] = rng.poisson(lam * .9)
        d["anom_trilho_12m"] = rng.poisson(lam * 1.4 + .3)
        d["anom_solda_12m"] = rng.poisson(lam * 1.1 + .2)
        d["anom_geometria_12m"] = rng.poisson(.8 + lam * .6)
        d["anom_amv_12m"] = rng.poisson(d["tem_amv"] * (1 + lam))
        d["anom_dormente_12m"] = rng.poisson(1.2 + lam * .4)
        d["anom_lastro_12m"] = rng.poisson(1.0 + lam * .3)
        d["notas_abertas"] = rng.poisson(2 + lam * 2)
        d["log_tonelagem"] = np.log1p(d["ton_base"] * rng.uniform(.95, 1.05, len(d)))
        viz = d.groupby("ramal")["falhas_36m"].transform(lambda s: s.rolling(3, center=True, min_periods=1).sum()) - d["falhas_36m"]
        d["viz_falhas_12m"] = viz
        blocos.append(d)
    p = pd.concat(blocos, ignore_index=True)
    logit = INTERCEPTO + sum(PESOS[k] * p[k] for k in PESOS)
    p["prob"] = 1 / (1 + np.exp(-logit))
    # contribuições (pesos × desvio da mediana do mês), como motivos do risco
    med = p.groupby("mes")[list(PESOS)].transform("median")
    for k in PESOS:
        p[f"c_{k}"] = PESOS[k] * (p[k] - med[k])
    fam = p[[f"anom_{f}_12m" for f in FAMILIAS]].to_numpy()
    p["familia"] = [FAMILIAS[i] for i in fam.argmax(axis=1)]
    p["acao"] = p["familia"].map(ACAO)
    p["rank_pct"] = p.groupby("mes")["prob"].rank(ascending=False, pct=True)
    return p.drop(columns=["propensao", "ton_base"])


def motivos(linha: pd.Series, n: int = 4) -> pd.DataFrame:
    c = pd.Series({ROTULOS[k]: linha[f"c_{k}"] for k in PESOS})
    c = c[c > 0].sort_values(ascending=False).head(n)
    return c.rename("contribuição").reset_index().rename(columns={"index": "motivo"})
