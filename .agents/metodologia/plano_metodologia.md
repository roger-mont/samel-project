# Plano de Adequação — Metodologia Integrada × Sistema Atual

> Gerado em: 2026-08-13  
> Baseado na Metodologia (Metodologia.docx) + análise do sistema atual + sessão /grill-me

---

## 1. Conflitos Críticos Identificados (o que a Metodologia assume mas o hardware não permite)

### Conflito 1 — Pipeline físico V→R→G→F inaplicável

A Metodologia (context.md §2 e Metodologia §4.2) assume:

```
ADC_bruto (0–1023)  →  V_out = ADC × (VCC / resolução)
                    →  R_FSR = R_fixo × (VCC/V_out - 1)
                    →  C = 1/R_FSR
                    →  F = m·C + b
```

**Realidade:** O WangYing entrega `0–255` já processados pelo firmware. Não temos acesso a VCC, resolução do ADC, R_fixo, nem ao ADC bruto. O pipeline físico completo é **impossível de reproduzir** com este hardware.

**Decisão tomada:** A calibração empírica por bloco (`raw_sum → kg` via polinômio) **substitui** as camadas 1-3 da Metodologia. O `raw_value 0-255` é tratado como "grandeza eletrônica de entrada" (`S_ij`) sem conversão física intermediária.

---

### Conflito 2 — Calibração individual por sensor é impossível

A Metodologia (§4.2, §14) exige curva `F_i = f(S_i)` para cada sensor individualmente. 

**Realidade:** O WangYing agrupa sensores em blocos 16×16 e os entrega por pacotes de bloco. A menor unidade de calibração física praticável é o **bloco inteiro (16×16 = 256 sensores)** usando uma placa rígida.

**Decisão tomada:** Calibração granular = por bloco. Dentro de um bloco, assume-se resposta uniforme (mesma curva para todos os 256 sensores). O modelo refinará isso via camada de correção estatística quando dados suficientes estiverem disponíveis.

---

### Conflito 3 — context.md menciona protótipo 3×3, hardware real é 32×64

O context.md fala em "escalar do protótipo 3×3 para 16×16". O hardware real é 32×64 com 8 blocos de 16×16.

**Decisão tomada:** O software já opera em 32×64. O context.md está desatualizado — ignorar essa parte.

---

## 2. Decisões de Design (resultado da sessão /grill-me)

| # | Questão | Decisão |
|---|---------|---------|
| D1 | Hardware final | WangYing IS o hardware final |
| D2 | Granularidade de calibração | Por bloco 16×16 (mínimo fisicamente acessível) |
| D3 | Acesso ao ADC bruto | Não disponível — usar 0-255 como proxy do sinal elétrico |
| D4 | Prioridade do produto | Peso preciso E mapa de pressão com mesma prioridade |
| D5 | Estratégia de implementação | 3 fases progressivas (ver §3) |
| D6 | Logging de sessão | CSV por sessão de calibração |
| D7 | Critério de estabilidade | Adaptar Metodologia §13 ao weight lock atual |
| D8 | Campanha experimental | Acesso parcial ao hardware — protocolo por sessão |

---

## 3. Pipeline Revisado (adequado ao hardware real)

```
S_ij(t) [0-255, por bloco]
    │
    ├─[Tara]──────────────────── S_net_ij = S_ij - baseline_ij
    │
    ├─[Deadzone]─────────────── S_net_ij < threshold → 0
    │
    ├─[Calibração por bloco]──── F_bloco_k = polyval(coef_k, sum(S_net_k))   ← Camada 2 da Metodologia
    │                            (curva empírica raw_sum → Newton-equivalente)
    │
    ├─[Campo 2D]─────────────── F_ij = F_bloco_k / N_pixels_ativos_k         ← Camada 3
    │
    ├─[Integração 2D]──────────  F_total = Σ_i Σ_j F_ij × ΔA               ← Camada 4 (Fase 2)
    │                            (primeiro: soma direta; depois: trapezoidal)
    │
    ├─[Conversão]──────────────  m_física = F_total / 9.81                   ← Camada 5
    │
    ├─[Correção estatística]───  m̂ = a·m_física + b  →  multivariada        ← Camada 6 (Fase 2/3)
    │
    ├─[EMA temporal]──────────── suavização do display
    │
    └─[Weight Lock]────────────  |m(t) - m(t-Δt)| < ε por T_min → peso travado
```

---

## 4. Plano de Ação por Fase

### FASE 1 — Infraestrutura de dados e pipeline base (implementar agora)

> Objetivo: ter o sistema pronto para coletar dados de calibração de forma estruturada e alimentar as fases seguintes.

#### 4.1.1 Logging de sessão de calibração (CSV)

**Novo módulo:** `Python/services/session_logger.py`

Campos obrigatórios por linha (conforme Metodologia §11):
- `timestamp_iso` — ISO 8601
- `session_id` — UUID por execução
- `block_id` — bloco sendo avaliado
- `reference_kg` — massa real de referência
- `position_tag` — ex: `"center"`, `"upper"`, `"lower"`, etc.
- `repetition` — número da repetição
- `raw_sum_block` — soma bruta do bloco
- `net_sum_block` — após subtração da tara
- `estimated_kg` — pelo modelo atual
- `stability_state` — `"transient"` | `"stable"`
- `mean_raw_block` — média do raw no período de coleta
- `std_raw_block` — desvio padrão (já temos do CLI)
- `cv_pct` — coeficiente de variação (%)

**Integração:** `calibrate_block.py` escreve uma linha CSV por medição registrada.

---

#### 4.1.2 Adaptação do weight lock ao critério da Metodologia §13

**Critério atual:** variância < threshold AND drift < threshold  
**Critério da Metodologia:** `|m(t) - m(t - Δt)| < ε` satisfeito por `T_min` consecutivo

**Decisão:** adicionar o critério `|Δm| < ε` como condição adicional, parametrizável via `settings.py`:
- `STABILITY_EPSILON_KG = 0.3` (variação máxima entre frames consecutivos)
- `STABILITY_TMIN_S = 10.0` (mínimo de 10s estável para travar)

---

#### 4.1.3 Conversão F→m no pipeline (unidades físicas intermediárias)

O pipeline atual vai diretamente de `raw_sum → kg`.  
A Metodologia exige `raw → F (N) → m (kg) = F/g`.

**Mudança:** a curva de calibração passa a ser ajustada em **Newtons** (`raw_sum → N`) e a divisão por 9.81 é feita explicitamente como etapa separada. Isso habilita a Camada 4 (integração 2D em força) na Fase 2.

**Impacto na calibração:** ao re-calibrar, a curva será `raw_sum → N` (peso em Newton = kg × 9.81). Os pontos de calibração existentes podem ser convertidos multiplicando `kg × 9.81`.

---

#### 4.1.4 Protocolo experimental mínimo para a próxima sessão com a maca

Conforme Metodologia §6 e §10, cada sessão deve documentar:

| Item | Ação |
|------|------|
| Firmware/software version | registrar no log |
| Tempo de aquecimento | aguardar 5 min antes de iniciar |
| Tara com manta + colchão | obrigatório a cada sessão |
| N ≥ 3 repetições por ponto de peso | mínimo aceitável no acesso parcial |
| Posições testadas | ao menos: central e 2 off-center |
| Ciclos histerese | um ciclo crescente E decrescente por sessão |

---

### FASE 2 — Integração 2D e modelo de correção (quando tiver ≥ 3 sessões de dados)

> Pré-requisito: ao menos 3 sessões de calibração com logging CSV, cobrindo 3+ posições e faixas de peso.

#### 4.2.1 Modelo A — Soma direta de forças calibradas por bloco (baseline)

```python
F_A = Σ_k  calibration_k.sum_to_newton(raw_sum_k)
m_A = F_A / 9.81
```

Já temos a estrutura. Ajuste: alterar unidade de saída da curva para Newton.

---

#### 4.2.2 Modelo B — Integração numérica 2D (regra trapezoidal)

```python
# Cada pixel tem área ΔA = (largura_manta/64) × (altura_manta/32)
# F_ij = força estimada por pixel (F_bloco / N_ativos_no_bloco)
# Trapezoidal 2D:
from scipy.integrate import trapezoid
F_B = trapezoid(trapezoid(F_matrix, dx=dy, axis=0), dx=dx)
```

Comparar `F_B` vs `F_A` nos dados de validação.

---

#### 4.2.3 Modelo C — Regressão linear simples (correção de ganho/offset)

```python
m_hat = a * m_fisica + b
```

Ajustar `a` e `b` com os dados de calibração (split 70/30 conforme Metodologia §24).

---

#### 4.2.4 Modelo D — Regressão multivariada por features de bloco

```python
# Features: sum de cada um dos 8 blocos → 8 variáveis
m_hat = b0 + b1*S1 + b2*S2 + ... + b8*S8
```

Somente se Modelo C apresentar resíduos sistemáticos.

---

### FASE 3 — Validação formal e seleção do modelo final

> Pré-requisito: dataset com ≥ 5 repetições por condição, cobrindo range 0-100 kg.

- Split treino/validação/teste (70/20/10)
- Calcular EA, EP, MAE, RMSE, R² estratificados por faixa de massa e posição
- Caracterizar histerese (ciclos crescente/decrescente)
- Caracterizar drift (carga fixa por >120s)
- Testar inclinação (0°, 15°, 30°) após consolidar horizontal
- Selecionar modelo final pelo critério §31 da Metodologia
- Documentar o modelo selecionado com coeficientes fixos no `calibration.json`

---

## 5. O que NÃO será implementado (e por quê)

| Item da Metodologia | Decisão | Motivo |
|---|---|---|
| Pipeline V→R→G→F | **Não implementar** | Hardware não expõe ADC bruto |
| Calibração por sensor individual | **Não implementar** | Fisicamente impossível no WangYing |
| Matriz de correção de crosstalk C·F_med | **Fase futura** | Requer ensaios localizados específicos |
| Modelo F (Machine Learning) | **Fase futura** | Requer dataset representativo grande |
| Predição antecipada τ·dy/dt + y = Ku | **Fase futura** | Requer estudo de estabilização temporal sistemático |
| Painel de VCC, R_fixo, Fator M, Offset B | **Remover da UI** | Parâmetros inaplicáveis ao WangYing |

---

## 6. O que está em conflito no document context.md (e precisa de atualização)

| Trecho | Problema | Ação |
|---|---|---|
| "protótipo 3×3 escalando para 16×16" | Hardware real é 32×64 com 8 blocos | Atualizar context.md |
| "ADC 0–1023, VCC, R_fixo 10kΩ" | Não acessível no WangYing | Remover ou marcar como obsoleto |
| "Painel de calibração com VCC, Resolução, R_fixo, M, B" | Parâmetros inexistentes | Substituir por painel de calibração por bloco |
| "50 a 100Hz de transmissão" | Real: 40Hz (25ms/frame) | Corrigir |

---

## 7. Ordem de execução sugerida (próximos sprints)

```
Sprint 1:  Session Logger CSV + integração no calibrate_block.py
Sprint 2:  Conversão da curva de calibração para Newton (F = kg × 9.81)
           Peso final: m = F_total / 9.81 (explícito no pipeline)
Sprint 3:  Adaptar weight lock ao critério da Metodologia §13
Sprint 4:  Sessão experimental com a maca (N≥3 por ponto, 3+ posições)
Sprint 5:  Modelo A validado em dados reais; comparar com polinômio atual
Sprint 6:  Integração trapezoidal 2D (Modelo B)
Sprint 7:  Regressão linear simples (Modelo C)
Sprint 8:  Análise de histerese e drift
Sprint 9:  Seleção do modelo final
```
