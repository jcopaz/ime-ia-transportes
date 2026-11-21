"""Etapa 1: limpa as bases e monta o painel. Uso: python -m src.preparar"""
import json
import re
import warnings
import pandas as pd
from . import config as C, carregar as L, painel as P

warnings.filterwarnings("ignore")


def main():
    C.PASTA_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    C.PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    q = {}
    notas = L.carregar_notas(q)
    falhas = L.carregar_falhas(q)
    trechos = L.carregar_trechos(q)
    # Quantas notas suspeitas a regra antiga (marco + deslocamento, qualquer valor) colocava
    # dentro do universo de trechos: são as que contaminavam o painel anterior.
    susp = notas[notas["km_suspeito"]].assign(km=lambda d: d["km_legado"])
    n_legado = len(L.no_universo(susp, trechos, "_tmp", {}))
    notas = L.no_universo(notas, trechos, "notas_vp", q)
    falhas = L.no_universo(falhas, trechos, "falhas_vp", q)

    # Notas com km suspeito (deslocamento vazio/negativo/>=1) saem da base.
    # Registramos quantas eram no total e quantas estavam dentro do universo de trechos.
    # km_suspeito_no_universo: pela regra antiga (km com deslocamento); a contagem pelo marco
    # inteiro (regra nova) fica em km_suspeito_no_universo_marco.
    q["notas_vp"]["km_suspeito_no_universo"] = n_legado
    q["notas_vp"]["km_suspeito_no_universo_marco"] = int(notas["km_suspeito"].sum())
    notas = notas[~notas["km_suspeito"]].drop(columns=["km_suspeito", "km_legado"]).reset_index(drop=True)
    q["notas_vp"]["no_universo_final"] = len(notas)

    painel = P.montar_painel(notas, falhas, trechos)
    q["checagem_amostra"] = {"divergencias": P.checar_amostra(painel, notas, falhas, n=300, seed=1),
                             "linhas_checadas": 300,
                             "variaveis_checadas": P.VARIAVEIS_CHECADAS}
    q["painel"] = {"linhas": len(painel), "trechos": int(painel.groupby(["ramal", "seg"]).ngroups),
                   "meses": int(painel["t"].nunique()),
                   "primeiro_mes": str(painel["t"].min().date()), "ultimo_mes": str(painel["t"].max().date()),
                   "taxa_positivos_%": round(painel["y"].mean() * 100, 2)}
    # Truncamentos que afetam a interpretação das variáveis (viram texto no paper)
    treino = painel[painel["t"] + pd.Timedelta(days=C.HORIZONTE_DIAS) <= pd.Timestamp(C.DATA_CORTE_TESTE)]
    q["truncamento"] = {
        "data_enc_min": q["notas_vp"].get("data_enc_min"),
        "janela_notas_completa_desde": "2024-02-01",
        "treino_linhas": len(treino),
        "treino_pct_dias_desde_falha_1500": round(float((treino["dias_desde_falha"] == 1500).mean() * 100), 2),
        "treino_pct_falhas_36m_igual_12m": round(float((treino["falhas_36m"] == treino["falhas_12m"]).mean() * 100), 2),
    }
    notas.to_pickle(C.PASTA_PROCESSADOS / "notas.pkl")
    falhas.to_pickle(C.PASTA_PROCESSADOS / "falhas.pkl")
    trechos.to_pickle(C.PASTA_PROCESSADOS / "trechos.pkl")
    painel.to_pickle(C.PASTA_PROCESSADOS / "painel.pkl")
    with open(C.PASTA_TABELAS / "qualidade_dados.json", "w", encoding="utf-8") as fh:
        json.dump(q, fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(q, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
