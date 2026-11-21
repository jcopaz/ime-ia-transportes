import pandas as pd
import pytest

from src.ativos import generico, nome_ativo, tabela_ativos, tabela_falhas


@pytest.mark.parametrize("tplnr, esperado", [
    ("MF-RMA-PAA_PAA-L000004-AMV0007", "AMV 7"),
    ("MF-RMA-PAA_PAA-L000004-AMV0005A", "AMV 5A"),
    ("MF-RMB-PAB_PAB-L000002-FILADIR", "Linha 2 · fila direita"),
    ("MF-RMB-PAB_PAB-L000002-FILAESQ", "Linha 2 · fila esquerda"),
    ("MF-RMC-PAC_PAD-L000001", "Linha 1"),
    ("MF-RD2-PAE_PAE-L00016A", "Linha 16A"),
    ("MF-RMB-PAB_PAB", "Pátio PAB"),
    ("MF-RME-PAF_PAG", "Trecho PAF–PAG"),
    ("MF-RMA-PAH_PAH-PTE0012", "Ponte 12"),
    ("MF-RMA-PAH_PAH-TUN0003", "Túnel 3"),
    ("MF-RMF-PAI_PAI-L00DM28", "MF-RMF-PAI_PAI-L00DM28"),
    ("MF-RD1-PAJ_PAJ-SINALIZ-144TCDVN", "MF-RD1-PAJ_PAJ-SINALIZ-144TCDVN"),
    (None, "Local não informado"),
    ("", "Local não informado"),
])
def test_nome_ativo(tplnr, esperado):
    assert nome_ativo(tplnr) == esperado


def test_generico():
    assert generico("MF-RME-PAF_PAG") and generico(None)
    assert not generico("MF-RME-PAF_PAG-L000001")
    assert generico("MF-RMF-PAI_PAI-L00DM28") and generico("MF-RD1-PAJ_PAJ-SINALIZ-144TCDVN")


def test_tabela_ativos_janelas_e_filtro():
    t = pd.Timestamp("2026-10-01")
    a, b = "MF-X-A_A-L000001-AMV0001", "MF-X-A_A-L000002"
    notas = pd.DataFrame({
        "nota": [1, 2, 3, 4], "ramal": "X", "seg": 10, "km": [10.2, 10.4, 10.6, 10.8], "tplnr": [a, a, b, b],
        "data": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-01-01", "2026-11-01"]),
        "data_enc": pd.to_datetime([None, "2026-10-15", "2026-09-01", None]),
        "alta_prio": [1, 0, 0, 0], "texto": ["velho", "novo", "fechada", "futura"],
    })
    falhas = pd.DataFrame({
        "nota": [9, 8], "ramal": "X", "seg": 10, "km": [10.3, 10.3], "tplnr": [a, a],
        "data": pd.to_datetime(["2026-05-01", "2025-05-01"]), "defeito": ["quebra", "antiga"],
    })
    d = tabela_ativos(notas, falhas, t)
    assert d["tplnr"].tolist() == [a]                     # b: fechada antes de t; nota futura não conta
    r = d.iloc[0]
    assert (r["notas_abertas"], r["notas_abertas_alta_prio"], r["falhas_12m_ativo"]) == (2, 1, 1)
    assert r["ultimo_defeito"] == "novo" and r["nome_ativo"] == "AMV 1"


def _falha(**k):
    base = {"ramal": "X", "seg": 10, "km": 10.3, "data": pd.Timestamp("2026-05-01"), "nota": 1, "defeito": "Trilho Partido",
            "gatilho": "Fratura", "restricao": "INTERDIÇÃO", "tplnr": "MF-X-A_A-L000001", "thp_h": 2.0, "causa_6m": "Material",
            "texto_nota": "trilho partido no km", "local_fratura": "", "lado_fratura": "", "campo_causa_raiz": "",
            "justificativa_thp": "M04 - TRILHO QUEBRADO", "eng_causa_raiz": "", "eng_6m": ""}
    base.update({f"{a}_pq{i}": "" for a in ("campo", "eng") for i in range(1, 6)})
    return {**base, **k}


def test_tabela_falhas_escolhe_a_analise():
    t = pd.Timestamp("2026-10-01")
    d = tabela_falhas(pd.DataFrame([
        _falha(nota=1, eng_causa_raiz="Fadiga", eng_6m="Material", eng_pq1="a", eng_pq2="b", campo_causa_raiz="Outra"),
        _falha(nota=2, campo_causa_raiz="Solda ruim", campo_pq1="c", eng_pq1="-"),
        _falha(nota=3, causa_6m="Sem análise"),
        _falha(nota=5, eng_pq1="só por quê", campo_causa_raiz="Outra"),
        _falha(nota=6),
        _falha(nota=4, data=pd.Timestamp("2025-01-01")),
    ]), t).set_index("nota")
    assert 4 not in d.index                                # fora dos 12 meses
    assert (d.at[1, "analise"], d.at[1, "causa_raiz"], d.at[1, "porques"]) == ("Engenharia", "Fadiga", "a → b")
    assert (d.at[2, "analise"], d.at[2, "causa_raiz"], d.at[2, "porques"]) == ("Campo", "Solda ruim", "c")
    assert (d.at[3, "analise"], d.at[3, "causa_6m"]) == ("", "")   # sem análise: "pendente" na tela
    assert (d.at[5, "analise"], d.at[5, "porques"]) == ("Engenharia", "só por quê")   # sem causa raiz, ainda é Engenharia
    assert (d.at[6, "analise"], d.at[6, "causa_6m"]) == ("Campo", "Material")         # só a causa 6M de Campo


def test_tabela_falhas_so_conta_thp_de_via():
    t = pd.Timestamp("2026-10-01")
    d = tabela_falhas(pd.DataFrame([_falha(nota=1), _falha(nota=2, justificativa_thp="E02 - FALHA DE SINALIZAÇÃO")]), t)
    assert d.set_index("nota")["thp_via_h"].to_dict() == {1: 2.0, 2: 0.0}
