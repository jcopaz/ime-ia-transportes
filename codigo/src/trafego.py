"""
Análise adicional: tonelagem bruta (tráfego) no modelo completo.

Atribui a cada trecho de 1 km a tonelagem dos pátios de origem e destino do
entre-pátio mais próximo (média dos dois), somada nos 12 meses anteriores ao
mês t. Depois do último mês disponível, usa o último ano disponível (sem
informação futura). Compara, com a mesma configuração do modelo completo
(Random Forest, folha mínima 50), quatro variantes:
    completo, completo + tráfego, só histórico, só histórico + tráfego.

Uso:  python -m src.trafego   (depois de src.carregar e src.painel)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src import avaliacao as A
from src import config as C


def ler_tonelagem() -> pd.DataFrame:
    tb = pd.read_excel(C.ARQ_TONELAGEM, "Dados")
    tb["TB"] = (tb["TB"].astype(str).str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False).astype(float))
    tb["t"] = pd.to_datetime(tb["ANOMES"].astype(str), format="%Y%m")
    return tb


def mapa_trecho_patio(trechos: pd.DataFrame) -> pd.DataFrame:
    """Pátios de origem e destino do marco mais próximo de cada trecho, no mesmo ramal."""
    mk = pd.read_csv(C.ARQ_GEO_MARCOS)
    linhas = []
    for r, g in trechos.groupby("ramal"):
        m = mk[mk["trecho"] == r].sort_values("km")
        for s in g["seg"]:
            if m.empty:
                linhas.append((r, s, None, None))
                continue
            i = (m["km"] - (s + 0.5)).abs().idxmin()
            linhas.append((r, s, m.at[i, "origem"], m.at[i, "destino"]))
    return pd.DataFrame(linhas, columns=["ramal", "seg", "orig", "dest"])


def variavel_trafego(p: pd.DataFrame, trechos: pd.DataFrame) -> pd.DataFrame:
    tb = ler_tonelagem()
    W = tb.pivot_table(index="t", columns="PATIO", values="TB", aggfunc="sum").sort_index()
    W = W.reindex(pd.date_range(W.index.min(), W.index.max(), freq="MS"))
    soma12 = W.rolling(12, min_periods=12).sum()
    ultimo = W.index.max()
    mp = mapa_trecho_patio(trechos)
    blocos = []
    for t in sorted(p["t"].unique()):
        t = pd.Timestamp(t)
        ref = min(t - pd.DateOffset(months=1), ultimo)     # só o passado
        v = soma12.loc[ref]
        val = np.nanmean(np.vstack([v.reindex(mp["orig"]).to_numpy(), v.reindex(mp["dest"]).to_numpy()]), axis=0)
        blocos.append(pd.DataFrame({"ramal": mp["ramal"], "seg": mp["seg"], "t": t, "tb_12m": val}))
    T = pd.concat(blocos)
    T["t"] = T["t"].astype(p["t"].dtype)
    p = p.merge(T, on=["ramal", "seg", "t"], how="left")
    p["log_tb_12m"] = np.log1p(p["tb_12m"].fillna(0))
    return p


def main():
    p = pd.read_pickle(C.PASTA_PROCESSADOS / "painel.pkl")
    falhas = pd.read_pickle(C.PASTA_PROCESSADOS / "falhas.pkl")
    trechos = pd.read_pickle(C.PASTA_PROCESSADOS / "trechos.pkl")
    p = variavel_trafego(p, trechos)
    A.GRUPOS["TRAF"] = ["log_tb_12m"]
    variantes = {"completo": ["HIST", "CONTEXTO", "ANOM"], "completo_traf": ["HIST", "CONTEXTO", "ANOM", "TRAF"],
                 "hist": ["HIST", "CONTEXTO"], "hist_traf": ["HIST", "CONTEXTO", "TRAF"]}
    ajuste, validacao, treino, teste = A.divisoes(p)
    ramais = sorted(p["ramal"].unique())

    def rf():
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=50, max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=C.SEMENTE)

    def pontuar(base, alvo, grupos):
        m = rf().fit(A.matriz(base, ramais, grupos), base["y"].to_numpy())
        return pd.Series(m.predict_proba(A.matriz(alvo, ramais, grupos))[:, 1], index=alvo.index)

    scores = {k: pontuar(treino, teste, g) for k, g in variantes.items()}
    scores["atual"] = teste["score_atual"]
    scores["H3"] = teste["falhas_36m"] + 0.3 * teste["viz_falhas_12m"]
    scores["so_trafego"] = teste["log_tb_12m"]
    ev = A.eventos_90(falhas, teste["t"].unique())
    cluster = ev["ramal"] + "_" + ev["seg"].astype(str)
    cap = {k: A.captura_90(teste, s, ev, C.TOP_FRACAO)[0].to_numpy() for k, s in scores.items()}
    y = teste["y"].to_numpy()
    print(A.boot(cap, cluster, C.N_BOOT, 0.95).to_string())
    for k, s in scores.items():
        print(f"{k:14s} PR-AUC {average_precision_score(y, s):.3f}  ROC-AUC {roc_auc_score(y, s):.3f}")
    difs = {"completo_traf − completo": cap["completo_traf"] - cap["completo"],
            "completo_traf − hist_traf": cap["completo_traf"] - cap["hist_traf"],
            "hist_traf − hist": cap["hist_traf"] - cap["hist"],
            "completo − hist": cap["completo"] - cap["hist"],
            "completo_traf − atual": cap["completo_traf"] - cap["atual"],
            "completo_traf − H3": cap["completo_traf"] - cap["H3"]}
    print(A.boot(difs, cluster, C.N_BOOT, 1 - 0.05 / 3).to_string())


if __name__ == "__main__":
    main()
