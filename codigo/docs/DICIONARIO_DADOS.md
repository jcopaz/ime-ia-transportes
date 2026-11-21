# Dicionário de dados — A inspeção de via antecipa as falhas que param trens?

Documento de apoio à seção *Code and Data Availability* do artigo. Os dados são de uma ferrovia de carga brasileira, cuja identidade é preservada. As bases são internas e confidenciais e não serão publicadas. O código é aberto; este dicionário descreve o que cada variável significa e como foi construída, para que o estudo possa ser reproduzido com dados equivalentes de outra ferrovia.

## 1. Bases de origem

| Base | Origem | Formato | Registros lidos | Chave de ligação |
|---|---|---|---|---|
| Notas de inspeção de via | Sistema de manutenção (SAP), exportação de notas de Via Permanente | .xlsx | 146.428 | ramal + KM |
| Registro de falhas de via | Sistema de manutenção (SAP), registro oficial de falhas de Via Permanente | .xlsx | 5.006 | nº da nota de falha; ramal + KM |
| Análises de causa raiz | Análises das falhas pela Engenharia | .xlsx | 2.356 | nº da nota de falha |
| Marcos quilométricos | Arquivo KMZ de marcos, convertido em tabela de KMs válidos por ramal | .csv | ≈ 1,6 mil trechos | ramal + KM |

## 2. Regras de limpeza

| Regra | Efeito |
|---|---|
| KM da nota = marco + deslocamento; deslocamento negativo, maior ou igual a 1 km ou vazio é considerado inválido | 6.103 notas descartadas |
| Notas e falhas cujo ramal e KM não existem na tabela de marcos | 11.608 notas e 162 falhas fora do universo |
| Falhas consideradas: só as do registro oficial que têm análise de causa raiz | 2.356 falhas |
| Exclusão de causas externas ou de outra disciplina (intempérie, vandalismo, terceiros, faixa de domínio, eletroeletrônica, rejeitadas) | 45 falhas |
| Exclusão de falhas expurgadas | 29 falhas |
| Exclusão de falhas sem KM | 54 falhas |
| Exclusão de falhas duplicadas (mesma nota) | 19 falhas |
| Colunas com nomes de pessoas | não são lidas |
| **Resultado** | **126.771 notas e 2.046 falhas** dentro de ≈ 1,6 mil trechos |

## 3. Bases tratadas

### 3.1 Notas de inspeção (`notas`)

| Variável | Tipo | Descrição |
|---|---|---|
| `nota` | número | Identificador da nota no SAP |
| `data` | data | Data de abertura da nota |
| `data_enc` | data | Data de encerramento (vazia se aberta) |
| `status` | texto | Aberto ou Encerrado |
| `ramal` | texto | Código do ramal |
| `km` | número | KM calculado (marco + deslocamento) |
| `seg` | inteiro | Trecho de 1 km: parte inteira do KM |
| `familia` | texto | Família do defeito: trilho, junta, solda, AMV, dormente, geometria, lastro, fixação ou outra |
| `peso_prio` | número | Peso da prioridade da nota |
| `alta_prio` | 0/1 | 1 se a prioridade é alta |
| `tipo_nota` | texto | Tipo de nota no SAP |
| `tipo_insp` | texto | Origem da inspeção: ronda, ultrassom, carro de geometria, AMV, drone ou outra |
| `gerencia` | texto | Gerência responsável (codificada) |
| `codigo` | texto | Código do defeito |

### 3.2 Falhas de via (`falhas`)

| Variável | Tipo | Descrição |
|---|---|---|
| `nota` | número | Identificador da nota de falha |
| `data` | data | Data da falha |
| `ramal`, `km`, `seg` | — | Localização, como nas notas |
| `gerencia` | texto | Gerência (codificada) |
| `defeito` | texto | Descrição do defeito |
| `gatilho` | texto | Tipo de gatilho da análise |
| `fratura` | 0/1 | 1 se a falha foi fratura |
| `causa_6m`, `causa_6m_n2` | texto | Causa pelos 6M, níveis 1 e 2 |
| `restricao` | texto | Interdição ou restrição de circulação |
| `thp_h` | número | Horas de trem parado atribuídas à falha |

## 4. Painel de modelagem (`painel`): uma linha por trecho de 1 km e mês

Período: jan/2023 a jul/2026 (um registro por trecho e mês). Para cada mês `t`, todas as variáveis explicativas usam **apenas** registros anteriores ao primeiro dia de `t`. O alvo usa os 90 dias a partir de `t`.

### 4.1 Identificação e alvo

| Variável | Descrição |
|---|---|
| `ramal`, `seg` | Trecho de 1 km |
| `t` | Primeiro dia do mês de referência |
| `n_falhas_futuro` | Nº de falhas no trecho em [t, t + 90 dias) |
| `y` | **Alvo**: 1 se houve ao menos uma falha em [t, t + 90 dias) |
| `thp_futuro_h` | Horas de trem parado dessas falhas (métrica secundária) |

### 4.2 Histórico de falhas (grupo HIST)

| Variável | Descrição |
|---|---|
| `falhas_3m`, `falhas_12m`, `falhas_36m` | Nº de falhas no trecho nos 3, 12 e 36 meses anteriores (a série começa em jan/2023) |
| `fraturas_12m` | Nº de fraturas nos 12 meses anteriores |
| `dias_desde_falha` | Dias desde a última falha no trecho (1500 = nenhuma desde jan/2023) |
| `viz_falhas_12m` | Falhas nos 12 meses anteriores nos trechos vizinhos (±1 km) |

### 4.3 Anomalias de inspeção (grupo ANOM)

| Variável | Descrição |
|---|---|
| `notas_3m`, `notas_12m` | Nº de notas abertas no trecho nos 3 e 12 meses anteriores |
| `alta_prio_12m` | Nº de notas de alta prioridade nos 12 meses anteriores |
| `anom_<familia>_12m` | Nº de notas por família de defeito nos 12 meses anteriores (trilho, junta, solda, AMV, dormente, geometria, lastro, fixação) |
| `insp_<tipo>_12m` | Nº de notas por origem da inspeção nos 12 meses anteriores (ronda, ultrassom, carro de geometria, AMV, drone) |
| `abertas` | Nº de notas em aberto no início de `t` |
| `abertas_alta_prio` | Nº de notas de alta prioridade em aberto no início de `t` |
| `idade_media_abertas_meses` | Idade média, em meses, das notas em aberto |
| `abertas_mais_12m` | Nº de notas em aberto há mais de 12 meses |
| `viz_notas_12m` | Notas nos 12 meses anteriores nos trechos vizinhos (±1 km) |

### 4.4 Contexto e referência

| Variável | Descrição |
|---|---|
| `tem_amv` | 1 se houve nota de AMV no trecho nos 36 meses anteriores (indica aparelho de mudança de via) |
| `mes_do_ano` | Mês de `t` (1 a 12), para sazonalidade |
| `score_atual` | Pontuação de prioridade da plataforma atual, calculada com os pesos padrão sobre as notas em aberto. **Usada só como comparação (baseline), nunca como variável do modelo** |

## 5. Limitações conhecidas dos dados

- A base de notas não contém notas encerradas antes de 07/02/2023. As notas abertas antes de 2023 só aparecem se foram encerradas depois dessa data. Por isso, as variáveis de notas em aberto só são completas a partir de fev/2024.
- Houve encerramentos em massa de notas (por exemplo, 10.113 em 17/01/2024), que mudam o significado de "nota em aberto".
- O sistema não registra inspeções sem achado. Por isso, a intensidade de inspeção de cada trecho não é observada.
- Um trecho de pátio reúne várias linhas e AMVs, com exposição maior que um trecho de linha singela.
- As falhas dos meses mais recentes ainda estão em lançamento.
