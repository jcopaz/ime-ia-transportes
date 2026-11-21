# A inspeção de via antecipa as falhas que param trens?

Aprendizado de máquina aplicado às inspeções e às falhas de via de uma ferrovia de carga brasileira para a priorização da manutenção.

**Trabalho final da disciplina ET 261400 – Inteligência Artificial Aplicada aos Transportes**
Programa de Pós-Graduação em Engenharia de Transportes · Instituto Militar de Engenharia (IME) · 2026.2
Autor: Júlio César de Oliveira Paz

| | |
|---|---|
| Apresentação (HTML5) | <https://jcopaz.github.io/ime-ia-transportes/> |
| Artigo (short paper, PDF) | [`artigo.pdf`](artigo.pdf) |
| Código do estudo | [`codigo/`](codigo/) |

## O problema

A equipe de manutenção registra dezenas de milhares de anomalias de via por ano, mas consegue olhar de perto só uma fração da malha a cada mês. O critério de priorização em uso pondera as anomalias registradas e nunca foi medido contra a falha. O estudo pergunta se as anomalias da inspeção, somadas ao histórico de falhas, indicam com até 90 dias de antecedência os trechos de 1 km que vão falhar, melhor que esse critério.

## O que foi feito

- Painel mensal trecho de 1 km × mês, com 69.124 observações (jan/2023 a ago/2026) e 56 variáveis de histórico de falhas, anomalias de inspeção e contexto.
- Divisão temporal com ajuste, validação e treino final; o período de teste (jan/2025 a out/2026, 828 falhas) foi avaliado uma única vez.
- Comparação entre o critério atual, a contagem de anomalias, regras de histórico de falhas, regressão logística, Random Forest e XGBoost.
- Métrica na unidade da decisão: fração das falhas dos 90 dias seguintes que estava nos 10% de trechos de maior risco, com intervalos por bootstrap por trecho.
- Análise adicional com a tonelagem bruta transportada.

## Resultados (teste)

| Método | Captura em 10% dos trechos |
|---|---|
| Ordem aleatória | 10,0% |
| Critério atual (aproximado) | 41,5% |
| Contagem de anomalias | 47,4% |
| Melhor regra de histórico (H3) | 52,5% |
| Modelo completo (Random Forest) | **55,6%** |
| Modelo completo + tonelagem (adicional) | 57,3% |

O modelo captura 14,0 p.p. a mais que o critério atual (IC 98,3%: 8,4 a 19,9). A maior parte do ganho vem do histórico de falhas. Com a tonelagem controlada, a contribuição das anomalias de inspeção sobe de 3,7 para 5,9 p.p., o que indica que o sinal das notas não é apenas esforço de inspeção.

## Dados e confidencialidade

Os dados pertencem à ferrovia estudada, que não é identificada, e **não estão neste repositório**. Nomes de ramais, pátios e quilometragens foram retirados do código e dos materiais. Para reproduzir o estudo com dados próprios, coloque em `codigo/dados/` os arquivos descritos em [`codigo/docs/DICIONARIO_DADOS.md`](codigo/docs/DICIONARIO_DADOS.md):

```
codigo/dados/
├── notas_inspecao.xlsx          # notas de inspeção de via
├── falhas_via.xlsx              # registro oficial de falhas de via
├── analises_causa_raiz.xlsx     # análises de causa raiz das falhas
├── marcos_km.csv                # km válidos por ramal
├── marcos_patios.csv            # pátio / entre-pátio por km
└── tonelagem_por_patio.xlsx     # tonelagem bruta por pátio e mês (análise adicional)
```

```bash
cd codigo
pip install -r requirements.txt
python -m src.carregar     # leitura e limpeza das bases
python -m src.painel       # painel trecho × mês
python -m src.modelos      # seleção, treino final e avaliação única no teste
python -m src.figuras      # tabelas e figuras do artigo
python -m src.trafego      # análise adicional com a tonelagem
python -m pytest tests     # testes
```

## Apresentação

`index.html` é uma apresentação HTML5 de página única, publicada pelo GitHub Pages. Avança com as setas, Page Down / Page Up ou espaço; as animações internas de cada página rodam sozinhas. Os elementos dos slides 01, 02 e 08 usam dados ilustrativos.

## Declaração de uso de IA

Este trabalho foi preparado com o auxílio de IA — Claude (Anthropic) e Codex (OpenAI) —, utilizada no apoio à elaboração do código de tratamento e análise dos dados, com revisão independente do código por uma segunda ferramenta, na construção das figuras e do protótipo de interface e na revisão do texto. Todo o conteúdo foi revisado e corrigido pelo autor: o código foi executado e testado, todos os valores apresentados foram conferidos diretamente nas bases e cada referência foi verificada. A IA é uma ferramenta, não uma autora; a responsabilidade pelo conteúdo é do autor.

## Demonstração interativa (Streamlit)

A pasta [`app/`](app/) traz uma demonstração pública do unifilar de risco, com zoom da malha ao trecho de 1 km, motivos e ação sugerida, além dos resultados agregados do estudo. **A malha e os números do unifilar são sintéticos**, gerados em [`app/dados_sinteticos.py`](app/dados_sinteticos.py); nenhum dado da ferrovia estudada é usado.

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Publicação no Streamlit Community Cloud: repositório `jcopaz/ime-ia-transportes`, branch `main`, arquivo principal `app/streamlit_app.py`.
