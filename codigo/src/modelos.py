"""
Etapa 2: seleção na validação, treino final e avaliação única no teste.
Uso: python -m src.modelos

Pergunta: ranqueando os trechos no início de cada mês, quantas das falhas
dos 90 dias seguintes estavam entre os 10% de trechos de maior risco?
Métrica principal = captura em 90 d (mesma janela do alvo; ver avaliacao.py);
a captura no mês (métrica anterior à revisão de 2026-10-16) fica como T3(h).

Protocolo (sem vazamento; o teste é olhado UMA vez, no fim):
  1. Ajuste interno (meses t ≤ 2024-02) → treina cada configuração da grade
     nas variantes completo e só histórico.
  2. Validação (2024-05 a 2024-10; falhas de 2024-05-01 a 2024-12-29) →
     escolhe a configuração do completo, a do só histórico e o melhor
     baseline de histórico (família H). Grava selecao_validacao.csv.
  3. Treino final (t ≤ 2024-10) → retreina: completo e só anomalias com a
     configuração do completo; só histórico com a sua; e só histórico com a
     configuração do completo (ablação de arquitetura fixa, T3(i)).
  4. Teste (2025-01 a 2026-07; falhas de 2025-01-01 a 2026-09-28) → avalia
     uma vez; ICs por bootstrap por trecho.
"""
import datetime as dt
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from . import avaliacao as A
from . import config as C

warnings.filterwarnings("ignore")

# Constantes novas do config.py (com valor padrão se ainda não existirem)
TREINO_SENSIB_INICIO = getattr(C, "TREINO_SENSIB_INICIO", "2024-01-01")
N_BOOT = getattr(C, "N_BOOT", 2000)
N_COMP = getattr(C, "N_COMPARACOES_PRIMARIAS", 3)
NIVEL_PRIMARIO = 1 - 0.05 / N_COMP          # Bonferroni: 1 − 0,05/3 ≈ 98,3%
TOP = C.TOP_FRACAO
SEMENTES_SENSIB = [42, 1, 2, 3, 4]

# Nomes das linhas da tabela principal (T2)
ATUAL = "Critério atual (aproximado)"
NOTAS = "Contagem de anomalias"
HIST_MOD = "Modelo só histórico"
COMPLETO = "Modelo completo"
SO_ANOM = "Modelo só anomalias (descritivo)"
HIST_FIXA = "Modelo só histórico (config. do completo)"     # ablação com arquitetura fixa
NOME_VARIANTE = {"completo": COMPLETO, "so_historico": HIST_MOD, "so_anomalias": SO_ANOM}


# --------------------------------------------------------------------------
# Regras sem IA
# --------------------------------------------------------------------------
# Família de baselines de histórico de falhas (pré-definida; a melhor é escolhida na validação).
# Os termos 1e-3 só desempatam.
FAMILIA_H = {
    "H1": ("falhas_12m + 0,5·falhas_3m + 0,001·falhas_36m (original)",
           lambda d: d["falhas_12m"] + 0.5 * d["falhas_3m"] + 1e-3 * d["falhas_36m"]),
    "H2": ("falhas_36m + 0,001·falhas_12m",
           lambda d: d["falhas_36m"] + 1e-3 * d["falhas_12m"]),
    "H3": ("falhas_36m + 0,3·viz_falhas_12m",
           lambda d: d["falhas_36m"] + 0.3 * d["viz_falhas_12m"]),
    "H4": ("−dias_desde_falha + 0,001·falhas_36m",
           lambda d: -d["dias_desde_falha"] + 1e-3 * d["falhas_36m"]),
}


def regras_atuais(d: pd.DataFrame) -> dict:
    """Regras usadas hoje (não passam por seleção)."""
    return {ATUAL: d["score_atual"], NOTAS: d["notas_12m"]}


# --------------------------------------------------------------------------
# Modelos: grade pequena pré-definida (10 configurações)
# --------------------------------------------------------------------------
GRADE = (
    [{"nome": f"Regressão logística C={c}", "classe": "LR", "C": c} for c in (0.01, 0.1, 1)]
    + [{"nome": f"Random Forest folha={f}", "classe": "RF", "min_samples_leaf": f} for f in (20, 50, 100)]
    + [{"nome": f"XGBoost prof={p} árvores={n}", "classe": "XGB", "max_depth": p, "n_estimators": n}
       for p in (3, 4) for n in (200, 400)]
)


def construir(cfg: dict, y: np.ndarray, semente: int = C.SEMENTE):
    """Cria o modelo (ainda não treinado) de uma configuração da grade."""
    if cfg["classe"] == "LR":
        return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                             LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=2000))
    if cfg["classe"] == "RF":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=cfg["min_samples_leaf"],
                                      max_features="sqrt", class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=semente)
    pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
    return XGBClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         scale_pos_weight=pos_weight, eval_metric="aucpr",
                         random_state=semente, n_jobs=-1)


def treinar_e_pontuar(cfg, variante, base, alvo, ramais, semente=C.SEMENTE) -> pd.Series:
    """Treina em `base` e devolve o score (probabilidade) de cada linha de `alvo`."""
    grupos = A.VARIANTES[variante]
    m = construir(cfg, base["y"].to_numpy(), semente)
    m.fit(A.matriz(base, ramais, grupos), base["y"].to_numpy())
    return pd.Series(m.predict_proba(A.matriz(alvo, ramais, grupos))[:, 1], index=alvo.index), m


# --------------------------------------------------------------------------
# 1–2. Seleção (só ajuste e validação; o teste não entra aqui)
# --------------------------------------------------------------------------
PAPEL_GRADE = {"completo": "grade (modelo completo)", "so_historico": "grade (modelo só histórico)"}
CRITERIO = "captura 90 d top 10% na validação; desempate PR-AUC"


def agora() -> str:
    return dt.datetime.now().isoformat(timespec="microseconds")


def selecionar(ajuste: pd.DataFrame, validacao: pd.DataFrame, falhas: pd.DataFrame) -> dict:
    """
    Escolhe, pela captura em 90 d no top 10% na validação (desempate: PR-AUC),
      (i) a melhor regra da família H, (ii) a melhor configuração da grade para o
    modelo completo e (iii) para o modelo só histórico (ambos treinados no ajuste).
    Recebe as falhas inteiras e filtra aqui as da janela da validação.
    Grava outputs/tabelas/selecao_validacao.csv.
    """
    h = pd.Timedelta(days=A.HORIZONTE_DIAS)
    assert validacao["t"].max() + h <= pd.Timestamp(A.DATA_CORTE_TESTE)
    ramais = sorted(pd.concat([ajuste["ramal"], validacao["ramal"]]).unique())
    ev90 = A.eventos_90(falhas, validacao["t"].unique())
    ev_mes = A.eventos(falhas, validacao["t"].unique())
    assert ev90["data"].max() < pd.Timestamp(A.DATA_CORTE_TESTE)
    assert ev90["data"].min() >= pd.Timestamp(A.VAL_INICIO)
    assert ev90["data"].max() < validacao["t"].max() + h
    y = validacao["y"].to_numpy()

    def medir(score):
        return {"captura90_top10_val_%": A.captura_90(validacao, score, ev90, TOP)[0].mean() * 100,
                "captura_mes_top10_val_%": A.captura(validacao, score, ev_mes, TOP)[0].mean() * 100,
                "pr_auc_val": average_precision_score(y, score),
                "roc_auc_val": roc_auc_score(y, score)}

    linhas = []
    for nome_h, (descr, f) in FAMILIA_H.items():
        linhas.append({"papel": "família H", "nome": nome_h, "definicao": descr, **medir(f(validacao))})
    for v, papel in PAPEL_GRADE.items():
        for cfg in GRADE:
            s, _ = treinar_e_pontuar(cfg, v, ajuste, validacao, ramais)
            linhas.append({"papel": papel, "nome": cfg["nome"], "definicao": json.dumps(cfg, ensure_ascii=False),
                           **medir(s)})
    tab = pd.DataFrame(linhas)

    def melhor(papel):
        sub = tab[tab["papel"] == papel]
        # maior captura 90 d; empate → maior PR-AUC; empate → primeira da lista
        return sub.sort_values(["captura90_top10_val_%", "pr_auc_val"], ascending=False, kind="mergesort").index[0]

    i_h, i_cfg, i_hist = melhor("família H"), melhor(PAPEL_GRADE["completo"]), melhor(PAPEL_GRADE["so_historico"])
    tab["selecionado"] = False
    tab.loc[[i_h, i_cfg, i_hist], "selecionado"] = True
    cfg = json.loads(tab.loc[i_cfg, "definicao"])
    cfg_h = json.loads(tab.loc[i_hist, "definicao"])

    # Referências (não concorrem): regras atuais e as variantes com a configuração do completo
    ref = [{"papel": "referência", "nome": n, "definicao": "regra atual", **medir(s)}
           for n, s in regras_atuais(validacao).items()]
    for v, nome in (("so_historico", HIST_FIXA), ("so_anomalias", SO_ANOM)):
        s, _ = treinar_e_pontuar(cfg, v, ajuste, validacao, ramais)
        ref.append({"papel": "referência", "nome": nome, "definicao": f"{cfg['nome']} | {v}", **medir(s)})
    tab = pd.concat([tab, pd.DataFrame(ref).assign(selecionado=False)], ignore_index=True)
    tab["n_eventos90_val"] = len(ev90)
    tab["n_eventos_mes_val"] = len(ev_mes)
    tab = tab[["papel", "nome", "definicao", "captura90_top10_val_%", "captura_mes_top10_val_%", "pr_auc_val",
               "roc_auc_val", "selecionado", "n_eventos90_val", "n_eventos_mes_val"]].round(6)
    assert tab["selecionado"].sum() == 3

    C.PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(C.PASTA_TABELAS / "selecao_validacao.csv", index=False, encoding="utf-8-sig")
    return {"config": cfg, "config_hist": cfg_h, "H": tab.loc[i_h, "nome"], "tabela": tab,
            "n_eventos_val": len(ev90), "n_eventos_val_mes": len(ev_mes), "ev90_val": ev90, "ev_mes_val": ev_mes,
            "gravada_em": agora()}


# --------------------------------------------------------------------------
# Auxiliares de avaliação no teste
# --------------------------------------------------------------------------
def pct(df_boot: pd.DataFrame) -> dict:
    """Converte a saída de A.boot (uma linha por nome) em {nome: (estimativa, ic_inf, ic_sup)} em %."""
    return {r["nome"]: (r["estimativa"] * 100, r["ic_inf"] * 100, r["ic_sup"] * 100) for _, r in df_boot.iterrows()}


def janela(meses) -> str:
    """Janela de falhas da métrica de 90 d: [primeiro mês, último mês + 90 d), escrita com fim incluso."""
    m = pd.to_datetime(pd.Series(list(meses)))
    fim = m.max() + pd.Timedelta(days=A.HORIZONTE_DIAS - 1)
    return f"{m.min().date()} a {fim.date()}"


def main():
    p = pd.read_pickle(C.PASTA_PROCESSADOS / "painel.pkl")
    falhas = pd.read_pickle(C.PASTA_PROCESSADOS / "falhas.pkl")
    ajuste, validacao, treino, teste = A.divisoes(p)
    ramais = sorted(p["ramal"].unique())
    h = pd.Timedelta(days=A.HORIZONTE_DIAS)
    corte = pd.Timestamp(A.DATA_CORTE_TESTE)

    # ---------------- 1–2. seleção: o teste ainda não foi tocado ----------------
    sel = selecionar(ajuste, validacao, falhas)
    cfg, cfg_h, H = sel["config"], sel["config_hist"], sel["H"]
    tab_sel = sel["tabela"]
    print("Selecionado na validação:", cfg["nome"], "|", cfg_h["nome"], "(só histórico) |", H)

    def val(papel, nome, coluna):
        r = tab_sel[(tab_sel["papel"] == papel) & (tab_sel["nome"] == nome)]
        assert len(r) == 1, (papel, nome)
        return r[coluna].iloc[0]

    # ---------------- 3. treino final (t ≤ 2024-10) ----------------
    teste_pontuado_em = agora()
    scores, modelos_finais = {}, {}
    scores.update(regras_atuais(teste))
    nome_H = f"Melhor histórico ({H})"
    scores[nome_H] = FAMILIA_H[H][1](teste)
    for v, c in (("so_historico", cfg_h), ("completo", cfg), ("so_anomalias", cfg)):
        s, m = treinar_e_pontuar(c, v, treino, teste, ramais)
        scores[NOME_VARIANTE[v]], modelos_finais[v] = s, m
    if cfg_h == cfg:
        scores[HIST_FIXA] = scores[HIST_MOD]
    else:
        scores[HIST_FIXA], _ = treinar_e_pontuar(cfg, "so_historico", treino, teste, ramais)
    principais = [ATUAL, NOTAS, nome_H, HIST_MOD, COMPLETO, SO_ANOM]

    # ---------------- 4. avaliação única no teste ----------------
    ev = A.eventos_90(falhas, teste["t"].unique())
    assert ev["data"].min() >= corte and ev["data"].max() < teste["t"].max() + h
    assert ev["n_rankings"].between(1, 4).all()      # 4 só quando fevereiro entra na janela
    cluster = ev["ramal"] + "_" + ev["seg"].astype(str)        # cluster = trecho
    mes = ev["t"].astype(str)                                  # mês da falha
    thp = ev["thp_h"].to_numpy(dtype=float)
    yte = teste["y"].to_numpy()

    cap = {n: A.captura_90(teste, s, ev, TOP)[0].to_numpy() for n, s in scores.items()}
    ic_cap = pct(A.boot({n: cap[n] for n in principais}, cluster, N_BOOT, 0.95))
    ic_thp = pct(A.boot({n: cap[n] * thp for n in principais}, cluster, N_BOOT, 0.95, denominador=thp))

    # métrica anterior (captura no mês) — sensibilidade
    ev_mes = A.eventos(falhas, teste["t"].unique())
    cluster_mes = ev_mes["ramal"] + "_" + ev_mes["seg"].astype(str)
    cap_mes = {n: A.captura(teste, s, ev_mes, TOP)[0].to_numpy() for n, s in scores.items()}
    ic_mes = pct(A.boot({n: cap_mes[n] for n in principais}, cluster_mes, N_BOOT, 0.95))

    linha_val = {ATUAL: ("referência", ATUAL), NOTAS: ("referência", NOTAS), nome_H: ("família H", H),
                 HIST_MOD: (PAPEL_GRADE["so_historico"], cfg_h["nome"]),
                 COMPLETO: (PAPEL_GRADE["completo"], cfg["nome"]), SO_ANOM: ("referência", SO_ANOM)}
    config_txt = {ATUAL: "regra atual", NOTAS: "regra atual (notas_12m)", nome_H: f"{H}: {FAMILIA_H[H][0]}",
                  HIST_MOD: cfg_h["nome"], COMPLETO: cfg["nome"], SO_ANOM: cfg["nome"]}
    t2 = []
    for n in principais:
        e, lo, hi = ic_cap[n]
        me, mlo, mhi = ic_mes[n]
        te_, tlo, thi = ic_thp[n]
        t2.append({"Método": n, "Configuração": config_txt[n],
                   "Captura 90d top 10% validação (%)": round(val(*linha_val[n], "captura90_top10_val_%"), 2),
                   "Captura 90d top 10% teste (%)": round(e, 2), "IC95 cluster inf": round(lo, 2),
                   "IC95 cluster sup": round(hi, 2),
                   "Captura mês top 10% validação (%)": round(val(*linha_val[n], "captura_mes_top10_val_%"), 2),
                   "Captura mês top 10% teste (%)": round(me, 2), "Mês IC95 inf": round(mlo, 2),
                   "Mês IC95 sup": round(mhi, 2),
                   "PR-AUC": round(average_precision_score(yte, scores[n]), 4),
                   "ROC-AUC": round(roc_auc_score(yte, scores[n]), 4),
                   "THP top 10% (%)": round(te_, 2), "THP IC95 inf": round(tlo, 2), "THP IC95 sup": round(thi, 2)})
    t2 = pd.DataFrame(t2)
    t2.to_csv(C.PASTA_TABELAS / "desempenho_modelos.csv", index=False, encoding="utf-8-sig")

    # Comparações primárias P1–P3 (captura 90 d; IC 98,3% por cluster de trecho; mesmas réplicas)
    comps = {"P1: Completo − melhor histórico": (COMPLETO, nome_H),
             "P2: Completo − melhor só histórico (ablação)": (COMPLETO, HIST_MOD),
             "P3: Completo − Critério atual": (COMPLETO, ATUAL)}
    difs = {k: cap[a] - cap[b] for k, (a, b) in comps.items()}
    difs_mes = {k: cap_mes[a] - cap_mes[b] for k, (a, b) in comps.items()}

    def tabela_p(d, grupos, nivel, rotulo):
        r = pct(A.boot(d, grupos, N_BOOT, nivel))
        return pd.DataFrame([{"Comparação": k, "Δ (p.p.)": round(r[k][0], 2), "IC inf (p.p.)": round(r[k][1], 2),
                              "IC sup (p.p.)": round(r[k][2], 2), "Nível": round(nivel, 4), "Bootstrap": rotulo}
                             for k in d])

    p123 = tabela_p(difs, cluster, NIVEL_PRIMARIO, "cluster por trecho")
    p123.to_csv(C.PASTA_TABELAS / "comparacao_pareada.csv", index=False, encoding="utf-8-sig")

    # ---------------- sensibilidades (a)–(j); tudo em 90 d, exceto (h) ----------------
    sens = []

    def add(item, descr, metodo, e, lo=None, hi=None, nota=""):
        sens.append({"item": item, "descrição": descr, "método": metodo, "estimativa (%)": round(e, 2),
                     "IC inf": None if lo is None else round(lo, 2),
                     "IC sup": None if hi is None else round(hi, 2), "nota": nota})

    cfg_var = {"completo": cfg, "so_historico": cfg_h}

    # (a) treino só com t ≥ 2024-01 (janelas de notas de 12 m mais completas)
    treino_a = treino[treino["t"] >= pd.Timestamp(TREINO_SENSIB_INICIO)]
    sa = {}
    for v, c in cfg_var.items():
        s, _ = treinar_e_pontuar(c, v, treino_a, teste, ramais)
        sa[NOME_VARIANTE[v]] = A.captura_90(teste, s, ev, TOP)[0].to_numpy()
    for n, r in pct(A.boot(sa, cluster, N_BOOT, 0.95)).items():
        add("a", f"treino só t ≥ {TREINO_SENSIB_INICIO[:7]} ({len(treino_a)} linhas)", n, *r, "IC95 cluster")

    # (b) cinco sementes (cada variante com a sua configuração)
    for v, c in cfg_var.items():
        vals = []
        for sd in SEMENTES_SENSIB:
            s, _ = treinar_e_pontuar(c, v, treino, teste, ramais, semente=sd)
            vals.append(A.captura_90(teste, s, ev, TOP)[0].mean() * 100)
        nota = "mín–máx" + (" (regressão logística não usa semente)" if c["classe"] == "LR" else "")
        add("b", f"5 sementes {SEMENTES_SENSIB}: média", NOME_VARIANTE[v], np.mean(vals), min(vals), max(vals), nota)

    # (c) teste separado pelo ano da DATA DA FALHA
    fim_2026 = (teste["t"].max() + h - pd.Timedelta(days=1)).date()
    for rot, filtro in [("2025", ev["data"].dt.year == 2025), (f"2026 (até {fim_2026})", ev["data"].dt.year == 2026)]:
        idx = filtro.to_numpy()
        r = pct(A.boot({n: cap[n][idx] for n in principais}, cluster[idx], N_BOOT, 0.95))
        for n in principais:
            add("c", f"falhas de {rot} ({idx.sum()} falhas)", n, *r[n], "IC95 cluster")

    # (d) top 5% e top 20%
    for fr in (0.05, 0.20):
        cf = {n: A.captura_90(teste, scores[n], ev, fr)[0].to_numpy() for n in principais}
        for n, r in pct(A.boot(cf, cluster, N_BOOT, 0.95)).items():
            add("d", f"top {int(fr*100)}%", n, *r, "IC95 cluster")

    # (e) as 4 regras H e o baseline original (H1 com desempate "first", como no estudo antigo)
    ch = {f"{k}: {d}": A.captura_90(teste, f(teste), ev, TOP)[0].to_numpy() for k, (d, f) in FAMILIA_H.items()}
    ch["Baseline original (H1, desempate first)"] = A.captura_90(teste, FAMILIA_H["H1"][1](teste), ev, TOP,
                                                                  desempate="first")[0].to_numpy()
    for n, r in pct(A.boot(ch, cluster, N_BOOT, 0.95)).items():
        add("e", "família H no teste", n, *r, "IC95 cluster" + (" | escolhida na validação" if n.startswith(H + ":") else ""))

    # (f) P1–P3 com bootstrap iid e cluster pelo mês da falha
    p_iid = tabela_p(difs, None, NIVEL_PRIMARIO, "iid por evento")
    p_mes = tabela_p(difs, mes, NIVEL_PRIMARIO, "cluster por mês da falha")
    for tb in (p_iid, p_mes):
        for _, r in tb.iterrows():
            add("f", f"Δ em p.p., IC {r['Nível']*100:.1f}% {r['Bootstrap']}", r["Comparação"],
                r["Δ (p.p.)"], r["IC inf (p.p.)"], r["IC sup (p.p.)"], "diferença pareada")

    # (g) desempate "first" (ordem das linhas) vs fracionário (captura esperada)
    for n in principais:
        first = A.captura_90(teste, scores[n], ev, TOP, desempate="first")[0].mean() * 100
        add("g", "desempate first", n, first, nota=f"fracionário = {cap[n].mean()*100:.2f}")

    # (h) captura no mês da falha (métrica principal antes da revisão)
    for n in principais:
        add("h", f"captura no mês ({len(ev_mes)} falhas)", n, *ic_mes[n],
            "IC95 cluster | métrica principal antes da revisão")
    for _, r in tabela_p(difs_mes, cluster_mes, NIVEL_PRIMARIO, "cluster por trecho").iterrows():
        add("h", f"Δ em p.p., IC {r['Nível']*100:.1f}% {r['Bootstrap']}", r["Comparação"],
            r["Δ (p.p.)"], r["IC inf (p.p.)"], r["IC sup (p.p.)"], "métrica principal antes da revisão")

    # (i) ablação com arquitetura fixa: só histórico com a configuração do completo
    igual = " | idêntica a P2" if cfg_h == cfg else ""
    r = pct(A.boot({HIST_FIXA: cap[HIST_FIXA]}, cluster, N_BOOT, 0.95))[HIST_FIXA]
    add("i", f"só histórico com {cfg['nome']}", HIST_FIXA, *r, "IC95 cluster" + igual)
    r = pct(A.boot({"Δ": cap[COMPLETO] - cap[HIST_FIXA]}, cluster, N_BOOT, NIVEL_PRIMARIO))["Δ"]
    add("i", f"Δ em p.p., IC {NIVEL_PRIMARIO*100:.1f}% cluster por trecho", f"Completo − {HIST_FIXA}", *r,
        "diferença pareada" + igual)

    # (j) só falhas com cobertura completa (todos os rankings de 90 d dentro do teste)
    idx = (ev["n_rankings"] == ev["n_rankings_teorico"]).to_numpy()
    r = pct(A.boot({n: cap[n][idx] for n in principais}, cluster[idx], N_BOOT, 0.95))
    for n in principais:
        add("j", f"cobertura completa ({idx.sum()} de {len(ev)} falhas)", n, *r[n], "IC95 cluster")
    pd.DataFrame(sens).to_csv(C.PASTA_TABELAS / "sensibilidade.csv", index=False, encoding="utf-8-sig")

    # ---------------- curvas de captura em 90 d com IC (F2) ----------------
    grade = np.round(np.linspace(0, 1, 101), 2)
    mats = {n: A.matriz_captura_90(teste, scores[n], ev, grade) for n in principais}
    cb = A.boot(mats, cluster, N_BOOT, 0.95)
    cb["fracao_trechos"] = grade[cb["j"].to_numpy()]
    cb = cb.rename(columns={"nome": "método", "estimativa": "captura"}).drop(columns="j")
    cb[["método", "fracao_trechos", "captura", "ic_inf", "ic_sup", "nivel"]].round(6).to_csv(
        C.PASTA_TABELAS / "curvas_captura.csv", index=False, encoding="utf-8-sig")

    # ---------------- captura em 90 d por causa raiz (só CSV; saiu do paper) ----------------
    causa = ev["causa_6m"].replace({"Inconclusivo": "Outros", "Falha Operacional": "Outros",
                                    "Nada Constatado": "Outros", "Meio Ambiente": "Outros"})
    por_causa = []
    for c, idx in causa.groupby(causa).groups.items():
        por_causa.append({"Causa (6M campo)": c, "Falhas no teste": len(idx),
                          **{n: round(cap[n][idx].mean() * 100, 1) for n in principais[:5]}})
    pd.DataFrame(por_causa).sort_values("Falhas no teste", ascending=False).to_csv(
        C.PASTA_TABELAS / "captura_por_causa.csv", index=False, encoding="utf-8-sig")

    # ---------------- resumo ----------------
    def periodo(d):
        return {"meses": f"{d['t'].min().date()} a {d['t'].max().date()}", "n_meses": int(d["t"].nunique()),
                "linhas": int(len(d)), "positivos_%": round(d["y"].mean() * 100, 2)}

    por_trecho = ev.groupby(["ramal", "seg"]).size()
    rpf = ev["n_rankings"].value_counts().sort_index()
    resumo = {
        "selecao_validacao": {
            "configuracao": cfg["nome"], "parametros": cfg,
            "configuracao_so_historico": cfg_h["nome"], "parametros_so_historico": cfg_h,
            "melhor_H": H, "melhor_H_definicao": FAMILIA_H[H][0], "criterio": CRITERIO,
            "n_configuracoes_grade": len(GRADE),
            "protocolo_anterior": {"configuracao": "Random Forest folha=50", "melhor_H": "H3",
                                   "criterio": "captura no mês top 10%; desempate PR-AUC"},
            "revisao": "métrica principal alterada para captura em 90 d após revisão independente de 2026-10-16 "
                       "(coerência com o alvo); captura no mês mantida como sensibilidade"},
        "periodos": {"ajuste": periodo(ajuste), "validacao": periodo(validacao), "treino_final": periodo(treino),
                     "teste": periodo(teste)},
        "eventos": {"metrica_principal": "captura_90d",
                    "validacao": sel["n_eventos_val"], "teste": int(len(ev)),
                    "validacao_mes": sel["n_eventos_val_mes"], "teste_mes": int(len(ev_mes)),
                    "validacao_janela_falhas": janela(validacao["t"].unique()),
                    "teste_janela_falhas": janela(teste["t"].unique()),
                    "teste_rankings_por_falha": {str(k): int(v) for k, v in rpf.items()},
                    "teste_cobertura_completa": int((ev["n_rankings"] == ev["n_rankings_teorico"]).sum()),
                    "teste_trechos_com_evento": int(len(por_trecho)),
                    "teste_participacao_10_trechos_com_mais_eventos_%": round(por_trecho.nlargest(10).sum() / len(ev) * 100, 2),
                    "teste_thp_total_h": round(float(thp.sum()), 2)},
        "definicao_metricas": {
            "captura_90d": "cada falha da data d é coberta pelos rankings mensais t com t ≤ d < t + 90 d "
                           "(mesma janela do alvo); captura da falha = média, nesses rankings, da captura "
                           "esperada do seu trecho no top 10% (empates divididos); cada falha pesa 1",
            "captura_mes": "cada falha conta só no ranking do início do seu mês; captura esperada do seu "
                           "trecho no top 10% (empates divididos)"},
        "registro": {"selecao_gravada_em": sel["gravada_em"], "teste_pontuado_em": teste_pontuado_em},
        "taxa_positivos_por_mes_teste_%": {str(k.date()): round(v * 100, 2)
                                           for k, v in teste.groupby("t")["y"].mean().items()},
        "n_variaveis": {v: int(A.matriz(teste.head(1), ramais, A.VARIANTES[v]).shape[1]) for v in A.VARIANTES},
        "bootstrap": {"replicas": N_BOOT, "semente": C.SEMENTE, "cluster": "trecho (ramal, seg)",
                      "nivel_primario": round(NIVEL_PRIMARIO, 4)},
        "horizonte_dias": A.HORIZONTE_DIAS, "top_fracao": TOP,
    }
    assert resumo["registro"]["selecao_gravada_em"] < resumo["registro"]["teste_pontuado_em"]
    with open(C.PASTA_TABELAS / "resumo_avaliacao.json", "w", encoding="utf-8") as fh:
        json.dump(resumo, fh, ensure_ascii=False, indent=2)

    # ---------------- modelo selecionado na validação + previsões ----------------
    pasta_mod = C.RAIZ / "outputs" / "modelos"
    pasta_mod.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelos_finais["completo"], "configuracao": cfg, "variante": "completo",
                 "colunas": list(A.matriz(teste.head(1), ramais, A.VARIANTES["completo"]).columns),
                 "ramais": ramais, "treinado_ate_t": str(treino["t"].max().date())},
                pasta_mod / "modelo_risco.joblib")
    prev = teste[["t", "ramal", "seg", "y", "thp_futuro_h"]].copy()
    for n, s in scores.items():
        prev[n] = np.asarray(s)
    prev.to_pickle(C.PASTA_PROCESSADOS / "previsoes_teste.pkl")
    ev.assign(**{n: cap[n] for n in scores}).to_pickle(C.PASTA_PROCESSADOS / "eventos_teste.pkl")
    ev_mes.assign(**{n: cap_mes[n] for n in scores}).to_pickle(C.PASTA_PROCESSADOS / "eventos_teste_mes.pkl")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(sel["tabela"].to_string(index=False))
    print(t2.to_string(index=False))
    print(p123.to_string(index=False))


if __name__ == "__main__":
    main()
