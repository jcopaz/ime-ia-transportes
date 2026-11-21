"""
Unifilar de risco — demonstração pública (dados sintéticos).

Mostra como a probabilidade de falha de via em 90 dias aparece para as equipes:
um diagrama unifilar com zoom da malha ao trecho de 1 km, os motivos do risco e
uma ação sugerida. A malha, os ramais e todos os números do unifilar são
sintéticos. A aba "Resultados do estudo" traz os resultados agregados do artigo.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dados_sinteticos import RAMAIS, gerar, motivos

st.set_page_config(page_title="Unifilar de risco · demonstração", page_icon="🛤️", layout="wide")

OURO, LARANJA, ACO, AZUL, TEXTO, FUNDO, PAINEL = "#E2B13C", "#F07A45", "#7E8FA9", "#6F97DA", "#EAF0F7", "#0A1322", "#101E34"
REPO = "https://github.com/jcopaz/ime-ia-transportes"
APRESENTACAO = "https://jcopaz.github.io/ime-ia-transportes/"

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; }}
.eyebrow {{ font-family: 'IBM Plex Mono', monospace; font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; color: {OURO}; }}
.aviso {{ border: 1px dashed {OURO}; border-radius: 10px; padding: .6rem .9rem; color: #A3B3CA; font-size: .9rem; }}
.cartao {{ background: {PAINEL}; border: 1px solid #24395C; border-radius: 12px; padding: 1rem 1.1rem; }}
.prob {{ font-size: 2.6rem; font-weight: 700; color: {LARANJA}; line-height: 1; }}
.acao {{ background: rgba(226,177,60,.14); border: 1px solid {OURO}; border-radius: 8px; padding: .6rem .8rem; margin-top: .4rem; }}
.acao b {{ display: block; font-size: .7rem; letter-spacing: .14em; color: {OURO}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def painel() -> pd.DataFrame:
    return gerar()


p = painel()
meses = sorted(p["mes"].unique())

# ---------------- cabeçalho ----------------
st.markdown('<div class="eyebrow">IME · IA Aplicada aos Transportes · 2026.2</div>', unsafe_allow_html=True)
st.title("Unifilar de risco de falha de via")
st.markdown(
    '<div class="aviso">Demonstração com <b>malha e dados sintéticos</b>. Ramais, pátios, quilômetros e números do '
    'unifilar são fictícios; nenhum dado da ferrovia estudada está aqui. '
    f'<a href="{APRESENTACAO}" target="_blank">Apresentação</a> · <a href="{REPO}" target="_blank">artigo e código</a>.</div>',
    unsafe_allow_html=True)
st.write("")

aba1, aba2, aba3 = st.tabs(["Unifilar de risco", "Como o modelo funciona", "Resultados do estudo"])

# ---------------- aba 1: unifilar ----------------
with aba1:
    if "nivel" not in st.session_state:
        ult = p[(p["mes"] == meses[-1]) & (p["ramal"] == "Ramal A")]
        km_max = int(ult.loc[ult["prob"].idxmax(), "km"])
        st.session_state.update(nivel="Malha", ramal="Ramal A", inicio=max(0, km_max - 5), trecho=None, nav=0)
    if "nivel_prox" in st.session_state:          # navegação pedida por um clique no gráfico
        st.session_state.nivel = st.session_state.pop("nivel_prox")

    def navegar(nivel_novo, **estado):
        st.session_state.update(nivel_prox=nivel_novo, nav=st.session_state.nav + 1, **estado)
        st.rerun()

    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    mes = c1.selectbox("Mês de referência", meses, index=len(meses) - 1,
                       format_func=lambda m: pd.Timestamp(m).strftime("%m/%Y"))
    cap = c2.slider("Trechos priorizados por mês (%)", 5, 30, 10, step=1,
                    help="A capacidade de atenção da equipe. No estudo, 10% dos trechos.")
    nivel = c3.radio("Nível do unifilar", ["Malha", "Ramal", "Trecho"], horizontal=True, key="nivel")

    d = p[p["mes"] == mes].copy()
    d["prior"] = d["rank_pct"] <= cap / 100
    n_prior = int(d["prior"].sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Trechos priorizados", f"{n_prior}", help="Trechos de 1 km no topo do ranking do mês")
    m2.metric("Probabilidade média nos priorizados", f"{d.loc[d.prior, 'prob'].mean():.0%}")
    m3.metric("Probabilidade média na malha", f"{d['prob'].mean():.1%}")

    def figura_base(altura):
        f = go.Figure()
        f.update_layout(height=altura, margin=dict(l=10, r=10, t=10, b=30), paper_bgcolor=PAINEL, plot_bgcolor=PAINEL,
                        font=dict(color=TEXTO, family="IBM Plex Sans, sans-serif"), showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False, color="#A3B3CA", title="km"),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=True, color=TEXTO),
                        dragmode=False, clickmode="event+select")
        return f

    def bolhas(f, x, y, prob, n, rotulo, custom, escala):
        f.add_trace(go.Scatter(x=x, y=y, mode="markers", customdata=custom, hovertext=rotulo, hoverinfo="text",
                               marker=dict(size=[escala(v) for v in n], color=prob, cmin=0.05, cmax=0.6,
                                           colorscale=[[0, OURO], [1, LARANJA]], opacity=.9,
                                           line=dict(width=6, color="rgba(240,122,69,.25)"))))

    sel = None
    if nivel == "Malha":
        f = figura_base(430)
        for i, (r, comp) in enumerate(RAMAIS.items()):
            f.add_shape(type="line", x0=0, x1=comp, y0=i, y1=i, line=dict(color=TEXTO, width=3))
        b = d.assign(bloco=(d["km"] // 10) * 10).groupby(["ramal", "bloco"]).agg(
            n=("prior", "sum"), prob=("prob", "max")).reset_index()
        b = b[b["n"] > 0]                                   # sem risco relevante, sem bolha
        y = b["ramal"].map({r: i for i, r in enumerate(RAMAIS)})
        bolhas(f, b["bloco"] + 5, y, b["prob"], b["n"],
               [f"{r} · km {k}–{k + 10}<br>{n} trecho(s) priorizado(s)" for r, k, n in zip(b.ramal, b.bloco, b.n)],
               np.stack([b["ramal"], b["bloco"]], axis=1), lambda v: 14 + 9 * v)
        f.update_yaxes(tickvals=list(range(len(RAMAIS))), ticktext=list(RAMAIS), autorange="reversed")
        ev = st.plotly_chart(f, width="stretch", on_select="rerun", selection_mode="points", key=f"g_malha_{st.session_state.nav}")
        if ev and ev.selection and ev.selection.points and ev.selection.points[0].get("customdata") is not None:
            r, k = ev.selection.points[0]["customdata"]
            navegar("Ramal", ramal=r, inicio=int(k))
        st.caption("Bolhas agregam blocos de 10 km; o tamanho é o número de trechos priorizados. Clique numa bolha para abrir o ramal.")

    elif nivel == "Ramal":
        r = st.selectbox("Ramal", list(RAMAIS), index=list(RAMAIS).index(st.session_state.ramal))
        st.session_state.ramal = r
        dr = d[d["ramal"] == r]
        f = figura_base(300)
        f.add_shape(type="line", x0=0, x1=RAMAIS[r], y0=0, y1=0, line=dict(color=TEXTO, width=3))
        b = dr.assign(bloco=(dr["km"] // 2) * 2).groupby("bloco").agg(n=("prior", "sum"), prob=("prob", "max")).reset_index()
        b = b[b["n"] > 0]
        bolhas(f, b["bloco"] + 1, [0] * len(b), b["prob"], b["n"],
               [f"km {k}–{k + 2} · prob. máx. {pr:.0%}" for k, pr in zip(b.bloco, b.prob)],
               b[["bloco"]].to_numpy(), lambda v: 18 + 10 * v)
        pats = dr.groupby("patio")["km"].min()
        f.add_trace(go.Scatter(x=pats.values, y=[-.35] * len(pats), mode="text", text=[f"Pátio {p_}" for p_ in pats.index],
                               textfont=dict(color="#A3B3CA", size=11), hoverinfo="skip"))
        f.update_yaxes(visible=False, range=[-.6, .5])
        ev = st.plotly_chart(f, width="stretch", on_select="rerun", selection_mode="points", key=f"g_ramal_{st.session_state.nav}")
        if ev and ev.selection and ev.selection.points and ev.selection.points[0].get("customdata") is not None:
            navegar("Trecho", inicio=max(0, int(ev.selection.points[0]["customdata"][0]) - 5), trecho=None)
        st.caption("Blocos de 2 km. Clique numa bolha para ver os trechos de 1 km.")

    else:
        r = st.selectbox("Ramal", list(RAMAIS), index=list(RAMAIS).index(st.session_state.ramal))
        st.session_state.ramal = r
        ini = st.slider("Início da janela (km)", 0, max(0, RAMAIS[r] - 12), min(st.session_state.inicio, max(0, RAMAIS[r] - 12)))
        st.session_state.inicio = ini
        dr = d[(d["ramal"] == r) & d["km"].between(ini, ini + 11)]
        col_g, col_d = st.columns([2, 1])
        with col_g:
            f = figura_base(300)
            f.add_shape(type="line", x0=ini, x1=ini + 12, y0=0, y1=0, line=dict(color=TEXTO, width=3))
            f.add_trace(go.Scatter(x=dr["km"] + .5, y=[0] * len(dr), mode="markers", hoverinfo="skip",
                                   marker=dict(size=6, color=ACO)))
            pr = dr[dr["prior"]]
            bolhas(f, pr["km"] + .5, [0] * len(pr), pr["prob"], (pr["prob"] * 10).round(),
                   [f"km {k}–{k + 1} · {v:.0%}" for k, v in zip(pr.km, pr.prob)], pr[["km"]].to_numpy(),
                   lambda v: 16 + 3 * v)
            f.update_yaxes(visible=False, range=[-.5, .5])
            f.update_xaxes(dtick=1)
            ev = st.plotly_chart(f, width="stretch", on_select="rerun", selection_mode="points", key=f"g_trecho_{st.session_state.nav}")
            if ev and ev.selection and ev.selection.points and ev.selection.points[0].get("customdata") is not None:
                st.session_state.trecho = int(ev.selection.points[0]["customdata"][0])
            st.caption("Um ponto por trecho de 1 km; bolha só nos priorizados. Clique numa bolha para ver os motivos.")
        with col_d:
            km = st.session_state.trecho
            if km is None or km not in dr["km"].values:
                km = int(dr.sort_values("prob", ascending=False)["km"].iloc[0])
            ln = dr[dr["km"] == km].iloc[0]
            st.markdown(f"""<div class="cartao"><div class="eyebrow">{r} · km {km} a {km + 1} · pátio {ln.patio}</div>
<div class="prob">{ln.prob:.0%}</div><div style="color:#A3B3CA">probabilidade de falha nos próximos 90 dias</div></div>""",
                        unsafe_allow_html=True)
            mt = motivos(ln)
            if len(mt):
                fm = go.Figure(go.Bar(x=mt["contribuição"], y=mt["motivo"], orientation="h", marker_color=OURO))
                fm.update_layout(height=210, margin=dict(l=0, r=10, t=30, b=0), paper_bgcolor=FUNDO, plot_bgcolor=FUNDO,
                                 font=dict(color=TEXTO, size=12), title=dict(text="Principais motivos", font=dict(size=13)),
                                 xaxis=dict(visible=False), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fm, width="stretch", config={"displayModeBar": False})
            st.markdown(f'<div class="acao"><b>AÇÃO SUGERIDA</b>{ln.acao}</div>', unsafe_allow_html=True)
            st.caption("A decisão é da engenharia de manutenção.")

    with st.expander("Ranking do mês (trechos priorizados)"):
        tab = d[d["prior"]].sort_values("prob", ascending=False)[["ramal", "km", "patio", "prob", "familia", "acao"]]
        tab = tab.rename(columns={"km": "km inicial", "patio": "pátio", "prob": "probabilidade", "familia": "família dominante", "acao": "ação sugerida"})
        st.dataframe(tab.style.format({"probabilidade": "{:.0%}"}), width="stretch", hide_index=True)

# ---------------- aba 2: como funciona ----------------
with aba2:
    passos = [
        ("1 · Registros", "Notas de inspeção, falhas oficiais, causa raiz, marcos de km e tonelagem são ligados a trechos de 1 km."),
        ("2 · Painel mensal", "Para cada trecho e mês, variáveis calculadas só com o passado: histórico de falhas, anomalias por família, notas em aberto, tráfego."),
        ("3 · Alvo", "Houve falha oficial no trecho nos 90 dias seguintes ao início do mês?"),
        ("4 · Modelo", "Random Forest escolhido na validação, entre dez configurações de regressão logística, Random Forest e XGBoost."),
        ("5 · Ranking", "No início de cada mês, os trechos são ordenados pela probabilidade; a equipe olha o topo da lista."),
        ("6 · Explicação", "Cada trecho mostra os principais motivos do risco e uma ação sugerida pela família de anomalia dominante."),
    ]
    cols = st.columns(3)
    for i, (t, txt) in enumerate(passos):
        cols[i % 3].markdown(f'<div class="cartao" style="margin-bottom:1rem"><b style="color:{OURO}">{t}</b><br>{txt}</div>',
                             unsafe_allow_html=True)
    st.info("O teste (jan/2025 a out/2026) foi avaliado uma única vez, com todas as escolhas congeladas antes, para não deixar o futuro vazar para o passado.")

# ---------------- aba 3: resultados ----------------
with aba3:
    st.markdown("Resultados agregados do artigo, no período de teste (828 falhas). **Estes números são reais e agregados**; o unifilar das outras abas é sintético.")
    curva = {
        "Modelo completo": [0.0, 18.3, 26.1, 32.9, 38.8, 42.8, 45.8, 48.9, 51.7, 53.6, 55.56, 57.7, 59.5, 61.8, 63.3, 64.2, 65.5, 66.4, 67.5, 68.7, 69.5, 70.2, 71.0, 71.5, 72.6, 73.7, 74.7, 75.8, 76.3, 77.2, 77.8],
        "Regra de histórico (H3)": [0.0, 21.1, 28.6, 32.7, 36.4, 41.3, 44.4, 47.2, 49.2, 50.7, 52.49, 54.4, 56.4, 58.4, 59.7, 60.9, 62.0, 62.9, 64.1, 65.4, 66.4, 67.1, 67.9, 69.0, 70.1, 71.1, 72.2, 73.1, 73.9, 74.8, 75.7],
        "Critério atual (aproximado)": [0.0, 12.5, 17.7, 22.7, 26.8, 28.9, 30.6, 32.9, 35.6, 39.0, 41.55, 43.8, 46.1, 48.7, 50.1, 52.1, 53.5, 55.5, 56.4, 57.5, 59.2, 60.9, 62.5, 63.5, 65.0, 66.1, 67.4, 68.2, 69.2, 70.2, 71.2],
    }
    fr = st.slider("Trechos priorizados (%)", 1, 30, 10, key="cap_res")
    a, b = st.columns([1, 1.4])
    with a:
        for nome, cor in zip(curva, [OURO, AZUL, ACO]):
            st.metric(nome, f"{curva[nome][fr]:.1f}%".replace(".", ","))
        st.metric("Ordem aleatória", f"{fr:.1f}%".replace(".", ","))
    with b:
        fc = go.Figure()
        for (nome, v), cor, ds in zip(curva.items(), [OURO, AZUL, ACO], ["solid", "dash", "dot"]):
            fc.add_trace(go.Scatter(x=list(range(31)), y=v, name=nome, line=dict(color=cor, width=3, dash=ds)))
        fc.add_trace(go.Scatter(x=[0, 30], y=[0, 30], name="Ordem aleatória", line=dict(color="#3A4E70", width=2)))
        fc.add_vline(x=fr, line_dash="dot", line_color=TEXTO)
        fc.update_layout(height=380, paper_bgcolor=FUNDO, plot_bgcolor=FUNDO, font=dict(color=TEXTO),
                         margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-.2),
                         xaxis=dict(title="trechos priorizados, em ordem de risco (%)", gridcolor="#1B2C48"),
                         yaxis=dict(title="falhas capturadas em 90 dias (%)", gridcolor="#1B2C48", range=[0, 100]))
        st.plotly_chart(fc, width="stretch")
    st.dataframe(pd.DataFrame({
        "Método": ["Ordem aleatória", "Critério atual (aproximado)", "Contagem de anomalias", "Regra de histórico (H3)",
                   "Modelo só histórico", "Modelo completo (Random Forest)", "Modelo completo + tonelagem (adicional)"],
        "Captura em 10% (%)": ["10,0", "41,5", "47,4", "52,5", "53,8", "55,6", "57,3"],
        "IC 95%": ["—", "33,5–49,4", "39,8–54,8", "44,4–59,7", "46,1–60,7", "48,1–62,4", "49,9–64,1"],
    }), width="stretch", hide_index=True)

st.divider()
st.caption("Declaração de uso de IA: este trabalho foi preparado com o auxílio de IA — Claude (Anthropic) e Codex (OpenAI) — "
           "no código, nas figuras, no protótipo de interface e na revisão do texto. Todos os valores foram conferidos pelo autor; "
           "a responsabilidade pelo conteúdo é do autor. · Júlio César de Oliveira Paz · IME 2026.2")
