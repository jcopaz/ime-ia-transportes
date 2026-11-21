"""
Etapa 3: figuras e tabelas do paper (a partir das saídas de src/modelos.py).
Uso: python -m src.figuras   (rodar depois de python -m src.modelos)

Gera:
  outputs/figuras/F1_protocolo.png       linha do tempo do protocolo
  outputs/figuras/F2_curva_captura.png   curva de captura 0–30% com faixa de IC
  outputs/tabelas/T1_dados.md            dados, qualidade e balanço das falhas
  outputs/tabelas/T2_resultado.md        resultado principal + comparações P1–P3
  outputs/tabelas/T3_sensibilidade.md    análises de sensibilidade (a)–(j)

As figuras não têm título: a legenda (caption) fica no texto do paper.
"""
import json

import matplotlib

matplotlib.use("Agg")                       # gera PNG sem precisar de tela
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from . import avaliacao as A
from . import config as C

PASTA_FIG = getattr(C, "PASTA_FIGURAS", C.RAIZ / "outputs" / "figuras")
ROTULOS = getattr(C, "ROTULOS_VARIAVEIS", {})
TREINO_SENSIB_INICIO = getattr(C, "TREINO_SENSIB_INICIO", "2024-01-01")

# Estilo legível em impressão (tons de cinza + hachuras; fonte 9 pt)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


# --------------------------------------------------------------------------
# Auxiliares
# --------------------------------------------------------------------------
def num(x, casas=1):
    """Número com vírgula decimal (padrão brasileiro)."""
    if x is None or pd.isna(x):
        return "–"
    return f"{x:.{casas}f}".replace(".", ",")


def ic(e, lo, hi, casas=1):
    return f"{num(e, casas)} [{num(lo, casas)}; {num(hi, casas)}]"


def rotular(texto: str) -> str:
    """Troca nomes de colunas pelo rótulo do paper (ROTULOS_VARIAVEIS); sem rótulo, mantém o nome."""
    for col in sorted(ROTULOS, key=len, reverse=True):
        texto = texto.replace(col, ROTULOS[col])
    return texto


def ler_json(nome):
    arq = C.PASTA_TABELAS / nome
    if not arq.exists():
        return {}
    with open(arq, encoding="utf-8") as fh:
        return json.load(fh)


def tabela_md(df: pd.DataFrame) -> str:
    """DataFrame → tabela Markdown simples."""
    linhas = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    linhas += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(linhas)


# --------------------------------------------------------------------------
# F1 — linha do tempo do protocolo
# --------------------------------------------------------------------------
def figura_protocolo(resumo: dict):
    per = resumo.get("periodos", {})
    um_mes = pd.DateOffset(months=1)
    h = pd.Timedelta(days=A.HORIZONTE_DIAS)

    def lim(nome, ini_padrao, fim_padrao):
        m = per.get(nome, {}).get("meses", f"{ini_padrao} a {fim_padrao}").split(" a ")
        return pd.Timestamp(m[0]), pd.Timestamp(m[1])

    faixas = [  # (rótulo, primeiro t, último t)
        ("Ajuste interno\n(treina a grade)", *lim("ajuste", "2023-01-01", "2024-02-01")),
        ("Validação\n(escolhe modelos e regra H)", *lim("validacao", "2024-05-01", "2024-10-01")),
        ("Treino final", *lim("treino_final", "2023-01-01", "2024-10-01")),
        ("Teste\n(avaliado uma vez)", *lim("teste", "2025-01-01", "2026-07-01")),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    y = list(range(len(faixas) + 1))[::-1]       # de cima para baixo
    for (rot, ini, fim), yy in zip(faixas, y[:-1]):
        # meses t usados (cada mês vai do dia 1 ao fim do mês)
        ax.barh(yy, (fim + um_mes) - ini, left=ini, height=0.5, color="0.35", edgecolor="black", lw=0.6)
        # horizonte do alvo do último mês (t + 90 dias)
        ax.barh(yy, (fim + h) - (fim + um_mes), left=fim + um_mes, height=0.5, color="white",
                edgecolor="black", hatch="////", lw=0.6)

    # Exemplo de um mês t: janela das variáveis (12 m antes) e alvo (90 d depois)
    t_ex = pd.Timestamp("2025-07-01")
    yy = y[-1]
    ax.barh(yy, t_ex - (t_ex - pd.DateOffset(months=12)), left=t_ex - pd.DateOffset(months=12), height=0.5,
            color="0.8", edgecolor="black", lw=0.6)
    ax.barh(yy, h, left=t_ex, height=0.5, color="white", edgecolor="black", hatch="xxxx", lw=0.6)
    ax.plot([t_ex, t_ex], [yy - 0.35, yy + 0.35], color="black", lw=1.2)
    ax.annotate("t", (t_ex, yy + 0.3), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom")

    # Fronteiras
    for d, txt in [(pd.Timestamp(A.VAL_INICIO), "início da validação"),
                   (pd.Timestamp(A.DATA_CORTE_TESTE), "corte do teste")]:
        ax.axvline(d, color="black", ls="--", lw=0.8)
        ax.text(d, y[0] + 0.45, txt + "\n" + d.strftime("%m/%Y"), ha="center", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))

    ax.set_yticks(y, [f[0] for f in faixas] + ["Exemplo de um\nmês t do painel"])
    ax.set_ylim(-0.6, y[0] + 1.2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.set_xlabel("Mês")
    ax.legend(handles=[
        Patch(facecolor="0.35", edgecolor="black", label="meses t usados (ranking no início do mês)"),
        Patch(facecolor="white", edgecolor="black", hatch="////", label="alvo dos últimos meses (90 dias)"),
        Patch(facecolor="0.8", edgecolor="black", label="janela das variáveis (12 meses antes de t)"),
        Patch(facecolor="white", edgecolor="black", hatch="xxxx", label="alvo: falha nos 90 dias após t"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8)
    PASTA_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_FIG / "F1_protocolo.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# F2 — curva de captura com faixa de IC (bootstrap por trecho)
# --------------------------------------------------------------------------
def figura_curva(curvas: pd.DataFrame):
    c = curvas[curvas["fracao_trechos"] <= 0.30 + 1e-9]
    nomes = c["método"].unique()
    melhor_h = next(n for n in nomes if n.startswith("Melhor histórico"))
    estilos = [  # (método, rótulo, estilo da linha, cor, com faixa de IC?)
        ("Modelo completo", "Modelo completo", "-", "black", True),
        (melhor_h, melhor_h.replace("Melhor histórico", "Melhor regra de histórico"), "--", "0.45", True),
        ("Critério atual (aproximado)", "Critério atual (aproximado)", ":", "black", False),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for nome, rot, ls, cor, faixa in estilos:
        d = c[c["método"] == nome]
        x = d["fracao_trechos"] * 100
        ax.plot(x, d["captura"] * 100, ls=ls, color=cor, lw=1.4, label=rot)
        if faixa:
            ax.fill_between(x, d["ic_inf"] * 100, d["ic_sup"] * 100, color=cor, alpha=0.18, lw=0)
    ax.plot([0, 30], [0, 30], color="0.6", lw=0.8, label="Ordem aleatória")
    ax.axvline(10, color="0.3", lw=0.6, ls="-.")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Trechos inspecionados, em ordem de risco (% do total)")
    ax.set_ylabel("Falhas do teste capturadas em 90 dias (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(PASTA_FIG / "F2_curva_captura.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# T1 — dados e qualidade
# --------------------------------------------------------------------------
def balanco_falhas(qualidade: dict) -> list:
    """Balanço das falhas (registro com análise → no universo), com as duas identidades conferidas."""
    f = qualidade.get("falhas_vp", {})
    chaves = ["falhas_com_analise", "notas_duplicadas_falhas", "sem_data", "causa_externa_ou_outra_disciplina",
              "expurgadas", "sem_km", "duplicadas_removidas", "falhas_finais", "fora_do_kmz", "no_universo"]
    if not all(k in f for k in chaves):
        return ["## Balanço das falhas\n", "Rode `python -m src.preparar` de novo: faltam contagens no qualidade_dados.json.", ""]
    exclusoes = [("notas_duplicadas_falhas", "− notas repetidas no registro de falhas"), ("sem_data", "− sem data válida"),
                 ("causa_externa_ou_outra_disciplina", "− causa externa ou outra disciplina"),
                 ("expurgadas", "− expurgo (campo Expurgo = Sim)"), ("sem_km", "− sem KM"),
                 ("duplicadas_removidas", "− duplicatas (mesmo dia, ramal, km e defeito)")]
    assert f["falhas_com_analise"] - sum(f[k] for k, _ in exclusoes) == f["falhas_finais"], \
        "balanço T1 não fecha (registro com análise → falhas finais)"
    assert f["falhas_finais"] - f["fora_do_kmz"] == f["no_universo"], \
        "balanço T1 não fecha (falhas finais → no universo)"
    linhas = [("registro de falhas com análise", f["falhas_com_analise"])] + [(r, f"−{f[k]}") for k, r in exclusoes]
    linhas += [("= falhas finais", f["falhas_finais"]), ("− fora do KMZ", f"−{f['fora_do_kmz']}"),
               ("= falhas no universo de trechos", f["no_universo"])]
    txt = ["## Balanço das falhas\n",
           tabela_md(pd.DataFrame({"Etapa": [a for a, _ in linhas], "Falhas": [b for _, b in linhas]})),
           "", "confere: sim (registro com análise − exclusões = falhas finais; falhas finais − fora do KMZ = no universo)"]
    if f.get("gatilhos_notas_duplicadas"):
        txt.append(f"\nInformativo (fora do balanço): {f['gatilhos_notas_duplicadas']} nota(s) repetida(s) "
                   "no arquivo de gatilhos, removidas antes do cruzamento.")
    return txt + [""]


def tabela_t1(qualidade: dict, resumo: dict) -> str:
    partes = ["# T1 — Dados e qualidade\n"]
    titulos = {"notas_vp": "Notas de inspeção de VP", "falhas_vp": "Falhas de VP (registro com análise)",
               "trechos": "Trechos", "painel": "Painel trecho × mês", "truncamento": "Truncamentos",
               "checagem_amostra": "Checagem amostral do painel"}
    for chave, bloco in qualidade.items():
        if not isinstance(bloco, dict):
            continue
        simples = {k: v for k, v in bloco.items() if not isinstance(v, (dict, list))}
        partes.append(f"## {titulos.get(chave, chave)}\n")
        partes.append(tabela_md(pd.DataFrame({"Item": list(simples), "Valor": list(simples.values())})))
        for k, v in bloco.items():                  # sub-blocos (ex.: fora do KMZ por ramal)
            if isinstance(v, dict) and v:
                partes.append(f"\n**{k}**\n")
                partes.append(tabela_md(pd.DataFrame({"Chave": list(v), "Valor": list(v.values())})))
            elif isinstance(v, list) and v:
                partes.append(f"\n**{k}** ({len(v)}): " + "; ".join(map(str, v)))
        partes.append("")
        if chave == "falhas_vp":
            partes += balanco_falhas(qualidade)
    per = resumo.get("periodos", {})
    ev = resumo.get("eventos", {})
    if per:
        nomes = {"ajuste": "Ajuste interno", "validacao": "Validação", "treino_final": "Treino final",
                 "teste": "Teste"}
        linhas = [{"Período": nomes.get(k, k), "Meses t": v.get("meses"), "Nº de meses": v.get("n_meses"),
                   "Linhas trecho×mês": v.get("linhas"), "Positivos (%)": num(v.get("positivos_%"), 2),
                   "Falhas avaliadas (90 d)": ev.get(k, "–") if k in ("validacao", "teste") else "–",
                   "Falhas avaliadas (no mês)": ev.get(f"{k}_mes", "–") if k in ("validacao", "teste") else "–"}
                  for k, v in per.items()]
        partes.append("## Divisão temporal\n")
        partes.append(tabela_md(pd.DataFrame(linhas)))
    if ev:
        rpf = ev.get("teste_rankings_por_falha", {})
        partes.append(f"\nTeste: {ev.get('teste')} falhas (métrica de 90 d) em {ev.get('teste_trechos_com_evento')} "
                      f"trechos; os 10 trechos com mais falhas concentram "
                      f"{num(ev.get('teste_participacao_10_trechos_com_mais_eventos_%'))}% delas.")
        partes.append(f"\nJanela de falhas da validação: {ev.get('validacao_janela_falhas', '–')}; "
                      f"do teste: {ev.get('teste_janela_falhas', '–')}. "
                      f"Falhas do teste por nº de rankings que as cobrem: "
                      # 4 rankings ocorre quando a janela inclui fevereiro (fev-01 + 90 d = 02/05)
                      + ", ".join(f"{k}: {v}" for k, v in sorted(rpf.items(), key=lambda kv: int(kv[0])))
                      + f"; com cobertura completa: {ev.get('teste_cobertura_completa', '–')}.")
        fim_val = str(ev.get("validacao_janela_falhas", "")).split(" a ")[-1] or "–"
        partes.append(f"\nAs falhas da validação vão até {fim_val}, antes do corte do teste ({A.DATA_CORTE_TESTE}): "
                      "nenhuma falha é usada nas duas etapas.")
    return "\n".join(partes) + "\n"


# --------------------------------------------------------------------------
# T2 — resultado principal
# --------------------------------------------------------------------------
def tabela_t2(t2: pd.DataFrame, comp: pd.DataFrame, resumo: dict) -> str:
    sel = resumo.get("selecao_validacao", {})
    ant = sel.get("protocolo_anterior", {})
    dm = resumo.get("definicao_metricas", {})
    linhas = []
    for _, r in t2.iterrows():
        linhas.append({
            "Método": r["Método"],
            "Captura 90 d top 10% validação (%)": num(r["Captura 90d top 10% validação (%)"]),
            "Captura 90 d top 10% teste (%) [IC95]": ic(r["Captura 90d top 10% teste (%)"], r["IC95 cluster inf"],
                                                        r["IC95 cluster sup"]),
            "PR-AUC": num(r["PR-AUC"], 3), "ROC-AUC": num(r["ROC-AUC"], 3),
            "THP top 10% (%) [IC95]*": ic(r["THP top 10% (%)"], r["THP IC95 inf"], r["THP IC95 sup"]),
        })
    nivel = comp["Nível"].iloc[0] * 100 if len(comp) else 98.3
    pc = [{"Comparação": r["Comparação"], "Δ (p.p.)": num(r["Δ (p.p.)"], 2),
           f"IC {num(nivel)}% (p.p.)": f"[{num(r['IC inf (p.p.)'], 2)}; {num(r['IC sup (p.p.)'], 2)}]"}
          for _, r in comp.iterrows()]
    boot = resumo.get("bootstrap", {}).get("replicas", 2000)
    txt = ["# T2 — Resultado principal (teste 2025-01 a 2026-07)\n",
           tabela_md(pd.DataFrame(linhas)), "",
           "## Comparações primárias (diferença pareada na captura em 90 d, top 10%)\n",
           tabela_md(pd.DataFrame(pc)), "",
           "Notas:",
           f"- Configuração escolhida na validação: modelo completo = {sel.get('configuracao', '–')}; "
           f"modelo só histórico = {sel.get('configuracao_so_historico', '–')}; "
           f"melhor regra de histórico: {sel.get('melhor_H', '–')} = {rotular(sel.get('melhor_H_definicao', '–'))}.",
           f"- Captura em 90 d (métrica principal): {dm.get('captura_90d', '–')}",
           f"- Captura no mês (sensibilidade, T3(h)): {dm.get('captura_mes', '–')}",
           "- IC95 e IC das comparações: bootstrap por cluster de trecho (ramal, km), "
           f"{boot} réplicas; comparações com correção de Bonferroni (3).",
           "- P2 compara o completo com o melhor modelo só histórico, escolhido na validação com a mesma grade; "
           "a ablação com arquitetura fixa (mesma configuração do completo) está em T3(i).",
           "- PR-AUC e ROC-AUC: alvo de 90 dias por linha trecho×mês (secundárias).",
           "- *THP: secundário e instável (poucas falhas concentram as horas paradas).",
           "- Modelo só anomalias: descritivo, fora das comparações primárias.",
           f"- Revisão do protocolo: {sel.get('revisao', '–')}. "
           f"Protocolo anterior: {ant.get('configuracao', '–')}, melhor regra {ant.get('melhor_H', '–')}, "
           f"critério {ant.get('criterio', '–')}; os resultados com a métrica anterior estão em T3(h). "
           "A mudança foi feita por coerência com o alvo de 90 dias, não pelos números do teste "
           "(que já tinham sido vistos com a métrica anterior).",
           "- Limitações: o critério atual é uma reprodução aproximada; o desfecho é a falha oficial registrada "
           "(sem auditoria de campo); o registro de falhas de ago–out/2026 pode estar incompleto por atraso "
           "de registro (T3(j) mostra só falhas com cobertura completa)."]
    return "\n".join(txt) + "\n"


# --------------------------------------------------------------------------
# T3 — sensibilidade
# --------------------------------------------------------------------------
TITULOS_SENS = {
    "a": f"(a) Treino só com meses t ≥ {TREINO_SENSIB_INICIO[:7]}",
    "b": "(b) Cinco sementes aleatórias (média e mín–máx)",
    "c": "(c) Teste separado por período",
    "d": "(d) Outras frações de trechos inspecionados",
    "e": "(e) Família de regras de histórico no teste",
    "f": "(f) Comparações P1–P3 com outros bootstraps (Δ em p.p.)",
    "g": "(g) Desempate pela ordem das linhas (first) vs. fracionário",
    "h": "(h) Captura no mês da falha (métrica anterior à revisão)",
    "i": "(i) Ablação com arquitetura fixa (só histórico com a configuração do completo)",
    "j": "(j) Só falhas com cobertura completa (todos os rankings de 90 d disponíveis)",
}


def tabela_t3(s: pd.DataFrame) -> str:
    txt = ["# T3 — Análises de sensibilidade (captura em 90 d no teste, %, exceto (h))\n"]
    for item, d in s.groupby("item", sort=True):
        txt.append(f"## {TITULOS_SENS.get(item, item)}\n")
        linhas = []
        for _, r in d.iterrows():
            faixa = "–" if pd.isna(r["IC inf"]) else f"[{num(r['IC inf'])}; {num(r['IC sup'])}]"
            linhas.append({"Recorte": r["descrição"], "Método": rotular(str(r["método"])),
                           "Estimativa": num(r["estimativa (%)"]), "Intervalo": faixa,
                           "Obs.": str(r["nota"]) if pd.notna(r["nota"]) else ""})
        txt.append(tabela_md(pd.DataFrame(linhas)))
        txt.append("")
    return "\n".join(txt) + "\n"


def main():
    T = C.PASTA_TABELAS
    resumo = ler_json("resumo_avaliacao.json")
    qualidade = ler_json("qualidade_dados.json")
    figura_protocolo(resumo)
    figura_curva(pd.read_csv(T / "curvas_captura.csv", encoding="utf-8-sig"))
    t2 = pd.read_csv(T / "desempenho_modelos.csv", encoding="utf-8-sig")
    comp = pd.read_csv(T / "comparacao_pareada.csv", encoding="utf-8-sig")
    sens = pd.read_csv(T / "sensibilidade.csv", encoding="utf-8-sig")
    (T / "T1_dados.md").write_text(tabela_t1(qualidade, resumo), encoding="utf-8")
    (T / "T2_resultado.md").write_text(tabela_t2(t2, comp, resumo), encoding="utf-8")
    (T / "T3_sensibilidade.md").write_text(tabela_t3(sens), encoding="utf-8")
    print("Figuras em", PASTA_FIG, "| tabelas .md em", T)


if __name__ == "__main__":
    main()
