"""
Etapa 4 (operacional): ranking de risco do mês atual para o app.
Uso: python -m src.pontuar            (mês atual = mês de C.DATA_REFERENCIA)
     python -m src.pontuar 2026-11    (outro mês)

Não altera nada do paper. Passos:
  1. Lê a configuração escolhida na validação (outputs/modelos/modelo_risco.joblib).
  2. Retreina essa configuração com TODOS os meses rotulados do painel
     (o modelo do paper para em 2024-10 por causa do protocolo de teste).
  3. Calcula as variáveis do mês t com o mesmo código do painel (src/painel.py)
     e pontua cada trecho de 1 km.
  4. Explica cada trecho: 3 motivos (SHAP; regras simples se o SHAP falhar),
     família dominante das notas abertas, ação sugerida e pátio.
  5. Grava outputs/app/risco_ml_atual.csv + risco_ml_meta.json + risco_ativos_atual.csv + risco_falhas_atual.csv,
     lidos pela cópia do app (aplicativo operacional): Unifilar e "Como o modelo funciona".
"""
import json
import sys
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd

from . import ativos as AT
from . import avaliacao as A
from . import config as C
from . import modelos as M
from . import painel as P

warnings.filterwarnings("ignore")

PASTA_APP = C.RAIZ / "outputs" / "app"

# Variáveis que aparecem na tela como "por que este trecho está no topo"
EXPLICATIVAS = ["falhas_12m", "falhas_36m", "dias_desde_falha", "viz_falhas_12m", "notas_12m",
                "alta_prio_12m", "abertas", "abertas_alta_prio", "abertas_mais_12m", "tem_amv"]

# Variáveis que podem virar motivo (o efeito de ramal e o mês do ano não explicam nada ao coordenador)
CANDIDATAS_MOTIVO = A.HIST + A.ANOM

# Frase de cada variável com o valor real do trecho; o que não estiver aqui usa C.ROTULOS_VARIAVEIS
NOMES_FAMILIA = {"trilho": "trilho", "junta": "junta", "solda": "solda", "amv": "AMV", "dormente": "dormente",
                 "geometria": "geometria", "lastro": "lastro", "fixacao": "fixação"}
NOMES_INSP = {"ronda": "ronda", "ultrassom": "ultrassom", "geometria_carro": "carro de geometria",
              "amv": "inspeção de AMV", "drone": "drone"}


def _n(v) -> str:
    return f"{int(round(v))}"


def _qtd(v: float, singular: str, plural: str) -> str:
    """'1 falha' / '7 falhas'."""
    n = int(round(v))
    return f"{n} {singular if n == 1 else plural}"


def frase(var: str, v: float) -> str | None:
    """Motivo em português simples com o valor do trecho; None quando a frase não faria sentido (ex.: valor 0)."""
    if var == "tem_amv":
        return "km com AMV" if v > 0 else None
    if var == "dias_desde_falha":
        return f"última falha há {_n(v)} dias" if v < 1500 else None
    if v <= 0:
        return None
    um = round(v) == 1
    defeitos, falhas = _qtd(v, "defeito", "defeitos"), _qtd(v, "falha", "falhas")
    aberto = "aberto" if um else "abertos"
    fixas = {
        "falhas_3m": f"{falhas} nos últimos 3 meses",
        "falhas_12m": f"{falhas} no último ano",
        "falhas_36m": f"{falhas} desde jan/2023",
        "fraturas_12m": f"{_qtd(v, 'fratura', 'fraturas')} no último ano",
        "viz_falhas_12m": "falha num km vizinho" if um else f"{_n(v)} falhas em kms vizinhos",
        "viz_notas_12m": f"{defeitos} em kms vizinhos no último ano",
        "notas_12m": f"{defeitos} {'registrado' if um else 'registrados'} no último ano",
        "notas_3m": f"{defeitos} nos últimos 3 meses",
        "alta_prio_12m": f"{defeitos} de alta prioridade no último ano",
        "abertas": f"{defeitos} {aberto}",
        "abertas_alta_prio": f"{defeitos} de alta prioridade {aberto}",
        "abertas_mais_12m": f"{defeitos} {aberto} há mais de 1 ano",
        "idade_media_abertas_meses": f"defeitos abertos há {_n(v)} meses em média",
    }
    if var in fixas:
        return fixas[var]
    # anom_* e insp_* contam o que foi REGISTRADO em 12 meses, não o que segue aberto
    if var.startswith("anom_"):
        return f"{defeitos} de {NOMES_FAMILIA.get(var[5:-4], var[5:-4])} no último ano"
    if var.startswith("insp_"):
        achado = "achado" if um else "achados"
        return f"{defeitos} {achado} por {NOMES_INSP.get(var[5:-4], var[5:-4])} no último ano"
    return f"{C.ROTULOS_VARIAVEIS.get(var, var)}: {v:g}"


def contribuicoes(modelo, X: pd.DataFrame, X_malha: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Contribuição de cada variável para o risco de cada trecho.
    1º SHAP (TreeExplainer, classe 1); se falhar, regra simples:
    (valor − média da malha) / desvio × importância da variável no modelo.
    """
    try:
        import shap
        sv = shap.TreeExplainer(modelo).shap_values(X, check_additivity=False)
        if isinstance(sv, list):                           # versões antigas: [classe0, classe1]
            sv = np.asarray(sv[1])
        else:
            sv = np.asarray(sv)
            if sv.ndim == 3:                               # versões novas: (n, variáveis, classes)
                sv = sv[:, :, 1]
        if sv.shape != X.shape:
            raise ValueError(f"formato inesperado do SHAP: {sv.shape} (esperado {X.shape})")
        return pd.DataFrame(sv, index=X.index, columns=X.columns), "SHAP"
    except Exception as erro:                              # sem SHAP: regras simples
        print(f"SHAP indisponível ({erro}); usando regras simples para os motivos.")
        imp = getattr(modelo, "feature_importances_", np.ones(X.shape[1]))
        z = (X - X_malha.mean()) / X_malha.std().replace(0, 1)
        return z * imp, "regras"


def motivos(atual: pd.DataFrame, contrib: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """As n variáveis com maior contribuição POSITIVA de cada trecho, já em frase."""
    cand = [c for c in CANDIDATAS_MOTIVO if c in contrib.columns]
    linhas = []
    for i in atual.index:
        ordem = contrib.loc[i, cand].sort_values(ascending=False)
        lista = []
        for var, c in ordem.items():
            if c <= 0 or len(lista) == n:
                break
            f = frase(var, float(atual.at[i, var]))
            if f:
                lista.append(f)
        linhas.append(lista + [""] * (n - len(lista)))
    return pd.DataFrame(linhas, index=atual.index, columns=[f"motivo_{k + 1}" for k in range(n)])


def familia_dominante(notas: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """Família mais frequente entre as notas abertas em t, por (ramal, seg); 'outras' fora das 8 famílias."""
    ab = notas[(notas["data"] < t) & (notas["data_enc"].isna() | (notas["data_enc"] >= t))]
    fam = ab["familia"].where(ab["familia"].isin(C.ACAO_POR_FAMILIA), "outras")
    cont = fam.groupby([ab["ramal"], ab["seg"]]).value_counts()
    # desempate estável: maior contagem, depois ordem alfabética
    cont = cont.rename("n").reset_index().sort_values(["ramal", "seg", "n", "familia"],
                                                      ascending=[True, True, False, True])
    return cont.drop_duplicates(["ramal", "seg"]).set_index(["ramal", "seg"])["familia"]


def patios(trechos: pd.DataFrame) -> pd.Series:
    """
    Pátio / entre-pátio de cada trecho pelo último marco com km <= início do trecho
    (geo_marcos.csv da plataforma atual). origem == destino → pátio; senão "ORIGEM–DESTINO".
    """
    try:
        g = pd.read_csv(C.ARQ_GEO_MARCOS)
    except FileNotFoundError:
        return pd.Series("", index=pd.MultiIndex.from_frame(trechos[["ramal", "seg"]]))
    g = g.dropna(subset=["km", "origem", "destino"])        # marcos sem km não localizam pátio
    g["ramal"] = g["trecho"].astype(str).str.upper().replace(C.RAMAIS_ALIASES)
    g["patio"] = np.where(g["origem"] == g["destino"], g["origem"], g["origem"] + "–" + g["destino"])
    base = trechos[["ramal", "seg"]].assign(km=lambda d: d["seg"] * C.TRECHO_KM).sort_values("km")
    m = pd.merge_asof(base, g[["ramal", "km", "patio"]].sort_values("km"), on="km", by="ramal",
                      direction="backward")
    # trecho antes do 1º marco do ramal: usa o marco mais próximo à frente
    frente = pd.merge_asof(base, g[["ramal", "km", "patio"]].sort_values("km"), on="km", by="ramal",
                           direction="forward")
    m["patio"] = m["patio"].fillna(frente["patio"]).fillna("")
    return m.set_index(["ramal", "seg"])["patio"]


def exportar_modelo_em_acao(modelo, X, contrib, chave, ramais, grupos, t, painel, falhas, metodo):
    """
    Pacote da aba "🧠 Modelo em Ação" do app (aplicativo operacional):
      modelo_operacional.joblib — o modelo do ranking do mês (simulador "E se…?");
      modelo_em_acao.pkl        — variáveis (X) e contribuições SHAP de cada trecho
                                  (Raio-X), rótulos legíveis, e o REPLAY do teste:
                                  scores do modelo do ESTUDO (treinado só até 2024-10)
                                  e do score atual em cada mês de 2025-01 a 2026-07 +
                                  as falhas desses meses (até t+90 d do último).
    """
    PASTA_APP.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "colunas": list(X.columns), "ramais": ramais, "grupos": grupos,
                 "mes_referencia": str(t.date())}, PASTA_APP / "modelo_operacional.joblib", compress=3)

    rotulos = dict(C.ROTULOS_VARIAVEIS)
    rotulos.update({f"ramal_{r}": "efeito do ramal" for r in ramais})
    prev = pd.read_pickle(C.PASTA_PROCESSADOS / "previsoes_teste.pkl")
    replay = prev[["t", "ramal", "seg", M.COMPLETO, M.ATUAL]].rename(
        columns={M.COMPLETO: "score_modelo", M.ATUAL: "score_atual"})
    ini, fim = replay["t"].min(), replay["t"].max() + pd.Timedelta(days=C.HORIZONTE_DIAS)
    f = falhas[(falhas["data"] >= ini) & (falhas["data"] < fim)].copy()
    f["seg"] = np.floor(f["km"] / C.TRECHO_KM).astype(int)
    pd.to_pickle({
        "X": X.reset_index(drop=True), "shap": contrib.reset_index(drop=True),
        "chave": chave.reset_index(drop=True), "rotulos": rotulos, "metodo_contribuicao": metodo,
        "medianas_malha": A.matriz(painel, ramais, grupos).median().to_dict(),
        "replay": replay, "replay_falhas": f[["data", "ramal", "seg", "km", "fratura", "defeito"]],
        "horizonte_dias": C.HORIZONTE_DIAS, "top_fracao": C.TOP_FRACAO,
    }, PASTA_APP / "modelo_em_acao.pkl")


def main(mes: str | None = None):
    t = pd.Timestamp(mes or C.DATA_REFERENCIA).to_period("M").to_timestamp()
    notas = pd.read_pickle(C.PASTA_PROCESSADOS / "notas.pkl")
    falhas = pd.read_pickle(C.PASTA_PROCESSADOS / "falhas.pkl")
    trechos = pd.read_pickle(C.PASTA_PROCESSADOS / "trechos.pkl")
    painel = pd.read_pickle(C.PASTA_PROCESSADOS / "painel.pkl")
    cfg = joblib.load(C.RAIZ / "outputs" / "modelos" / "modelo_risco.joblib")["configuracao"]

    if t <= painel["t"].max():
        raise SystemExit(f"{t.date()} já está no painel rotulado; use um mês posterior a {painel['t'].max().date()}.")

    # 2. Treino com todos os meses rotulados
    ramais = sorted(painel["ramal"].unique())
    grupos = A.VARIANTES["completo"]
    modelo = M.construir(cfg, painel["y"].to_numpy())
    modelo.fit(A.matriz(painel, ramais, grupos), painel["y"].to_numpy())

    # 3. Variáveis do mês t (o alvo calculado aqui é incompleto e é descartado)
    atual = P.montar_painel(notas, falhas, trechos, meses=[t])
    atual = atual.drop(columns=["y", "n_falhas_futuro", "thp_futuro_h"])
    X = A.matriz(atual, ramais, grupos)
    # Índice de risco 0–1 para ORDENAR trechos. Não é probabilidade calibrada:
    # o treino usa class_weight balanceado, que infla os valores.
    atual["indice_risco"] = modelo.predict_proba(X)[:, 1]

    # 4. Por que está aqui + família dominante + ação + pátio
    contrib, metodo_motivos = contribuicoes(modelo, X, A.matriz(painel, ramais, grupos))
    atual = atual.join(motivos(atual, contrib))
    exportar_modelo_em_acao(modelo, X, contrib, atual[["ramal", "seg"]], ramais, grupos, t,
                            painel, falhas, metodo_motivos)
    chave = pd.MultiIndex.from_frame(atual[["ramal", "seg"]])
    atual["familia_dominante"] = familia_dominante(notas, t).reindex(chave).fillna("").to_numpy()
    atual["acao_sugerida"] = (atual["familia_dominante"].replace("", "outras")
                              .map(C.ACAO_POR_FAMILIA).fillna(C.ACAO_POR_FAMILIA["outras"]))
    atual["patio"] = patios(trechos).reindex(chave).fillna("").to_numpy()

    # Ranking e faixa de risco (top 10% = métrica do estudo)
    atual = atual.sort_values("indice_risco", ascending=False, kind="mergesort").reset_index(drop=True)
    atual["posicao"] = range(1, len(atual) + 1)
    atual["percentil_top"] = atual["posicao"] / len(atual) * 100
    atual["faixa"] = pd.cut(atual["percentil_top"], [0, 5, 10, 20, 100],
                            labels=["Top 5%", "Top 5–10%", "Top 10–20%", "Demais"]).astype(str)
    atual["posicao_atual"] = atual["score_atual"].rank(ascending=False, method="min").astype(int)
    atual["km_ini"] = atual["seg"] * C.TRECHO_KM
    atual["km_fim"] = atual["km_ini"] + C.TRECHO_KM
    atual = atual.merge(trechos[["ramal", "seg", "lat", "lon"]], on=["ramal", "seg"], how="left")

    versao = (f"{cfg['nome']} · treino {painel['t'].min():%Y-%m}..{painel['t'].max():%Y-%m} · "
              f"gerado {date.today():%Y-%m-%d}")
    atual["mes_referencia"] = str(t.date())
    atual["versao_modelo"] = versao

    colunas = (["mes_referencia", "posicao", "ramal", "km_ini", "km_fim", "patio", "indice_risco", "faixa",
                "percentil_top", "score_atual", "posicao_atual"] + EXPLICATIVAS
               + ["motivo_1", "motivo_2", "motivo_3", "familia_dominante", "acao_sugerida",
                  "versao_modelo", "lat", "lon"])
    PASTA_APP.mkdir(parents=True, exist_ok=True)
    atual[colunas].round(6).to_csv(PASTA_APP / "risco_ml_atual.csv", index=False, encoding="utf-8-sig")
    AT.tabela_ativos(notas, falhas, t).to_csv(PASTA_APP / "risco_ativos_atual.csv", index=False,
                                              encoding="utf-8-sig")
    AT.tabela_falhas(falhas, t).to_csv(PASTA_APP / "risco_falhas_atual.csv", index=False, encoding="utf-8-sig")

    resumo = pd.read_csv(C.PASTA_TABELAS / "desempenho_modelos.csv")
    # chave "captura_top10_teste_%" mantida (o app lê); a partir da revisão de 2026-10-16 = captura em 90 d
    linha = lambda nome, col="Captura 90d top 10% teste (%)": resumo.set_index("Método").loc[nome, col]
    mes = lambda nome: linha(nome, "Captura mês top 10% teste (%)")
    with open(C.PASTA_TABELAS / "resumo_avaliacao.json", encoding="utf-8") as fh:
        periodos = json.load(fh)["periodos"]
    meta = {
        "mes_referencia": str(t.date()),
        "horizonte_dias": C.HORIZONTE_DIAS,
        "modelo": cfg["nome"],
        "versao_modelo": versao,
        "gerado_em": str(date.today()),
        "motivos_metodo": metodo_motivos,
        "treinado_com_meses": f"{painel['t'].min().date()} a {painel['t'].max().date()}",
        "linhas_treino": int(len(painel)),
        "trechos_pontuados": int(len(atual)),
        "ultima_falha_na_base": str(falhas["data"].max().date()),
        "ultima_nota_na_base": str(notas["data"].max().date()),
        # Métricas do estudo (modelo treinado só até o treino final, avaliado no teste);
        # o ranking atual vem do retreino com todos os meses e não tem avaliação própria.
        "estudo_treino_final": periodos["treino_final"]["meses"],
        "estudo_teste": periodos["teste"]["meses"],
        "captura_top10_teste_%": {"Modelo completo": float(linha(M.COMPLETO)),
                                  "Critério atual (aproximado)": float(linha(M.ATUAL))},
        "captura_top10_teste_metrica": "captura em 90 dias (falhas em [t, t+90d) cobertas pelos rankings mensais)",
        "captura_mes_top10_teste_%": {"Modelo completo": float(mes(M.COMPLETO)),
                                      "Critério atual (aproximado)": float(mes(M.ATUAL))},
        "taxa_base_%": round(float(painel["y"].mean() * 100), 2),
    }
    with open(PASTA_APP / "risco_ml_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(atual[["posicao", "ramal", "km_ini", "patio", "faixa", "motivo_1", "motivo_2", "motivo_3",
                 "acao_sugerida"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
