# Plano de Banco de Dados Hospitalar (Central UTI & PEP)

> **Projeto:** Maca Inteligente — Sistema de Monitoramento e Prevenção de LPP  
> **Escopo:** Modelagem Relacional, Séries Temporais, Retenção e Segurança (LGPD)  
> **Versão:** 1.0.0 | **Data:** 2026-08-21  

---

## 1. Escolha Tecnológica

* **Banco de Dados Central:** **PostgreSQL 16+** com particionamento nativo por data.
* **Justificativa:** Robustez ACID, suporte nativo a dados geoespaciais/matrizes (JSONB), particionamento de séries temporais de sensores e conformidade com requisitos hospitalares.

---

## 2. Esquema Relacional e DDL

```sql
-- 1. ESTRUTURA FÍSICA E DISPOSITIVOS
CREATE TABLE unidades_hospitalares (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL, -- Ex: "Hospital Samel Matriz"
    codigo_unidade VARCHAR(30) UNIQUE NOT NULL
);

CREATE TABLE alas_setores (
    id SERIAL PRIMARY KEY,
    unidade_id INT REFERENCES unidades_hospitalares(id),
    nome VARCHAR(60) NOT NULL, -- Ex: "UTI A", "Enfermaria 4"
    andar VARCHAR(20)
);

CREATE TABLE macas_dispositivos (
    id SERIAL PRIMARY KEY,
    ala_id INT REFERENCES alas_setores(id),
    codigo_maca VARCHAR(50) UNIQUE NOT NULL, -- Ex: "MACA-UTI-012"
    numero_leito VARCHAR(20) NOT NULL,       -- Ex: "Leito 03"
    mac_address VARCHAR(17),
    versao_software VARCHAR(30),
    data_ultima_calibracao TIMESTAMPTZ,
    status_conexao VARCHAR(20) DEFAULT 'offline', -- 'conectado', 'offline', 'manutencao'
    atualizado_em TIMESTAMPTZ DEFAULT NOW()
);

-- 2. PACIENTES E INTERNAÇÕES (Contexto Clínico)
CREATE TABLE pacientes (
    id BIGSERIAL PRIMARY KEY,
    codigo_paciente_pep VARCHAR(60) UNIQUE NOT NULL, -- ID anonimizado do Prontuário
    nome_completo VARCHAR(150) NOT NULL,
    data_nascimento DATE NOT NULL,
    sexo CHAR(1),
    criado_em TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE internacoes (
    id BIGSERIAL PRIMARY KEY,
    paciente_id BIGINT REFERENCES pacientes(id),
    maca_id INT REFERENCES macas_dispositivos(id),
    data_admissao TIMESTAMPTZ NOT NULL,
    data_alta TIMESTAMPTZ,
    escala_braden_score INT, -- Avaliação de risco LPP (6 a 23)
    tempo_estatico_limite_min INT DEFAULT 60,
    status VARCHAR(20) DEFAULT 'ativo' -- 'ativo', 'encerrado'
);

-- 3. SESSÕES E EVENTOS DE MONITORAMENTO
CREATE TABLE sessoes_monitoramento (
    id BIGSERIAL PRIMARY KEY,
    internacao_id BIGINT REFERENCES internacoes(id),
    maca_id INT REFERENCES macas_dispositivos(id),
    iniciado_em TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finalizado_em TIMESTAMPTZ,
    peso_admissao_kg NUMERIC(5,2),
    peso_atual_kg NUMERIC(5,2)
);

CREATE TABLE eventos_posturais (
    id BIGSERIAL PRIMARY KEY,
    sessao_id BIGINT REFERENCES sessoes_monitoramento(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    postura_anterior VARCHAR(40),  -- Ex: "Decúbito Lateral Direito"
    postura_detectada VARCHAR(40), -- Ex: "Decúbito Dorsal"
    duracao_postura_anterior_seg INT,
    regiao_pico_pressao VARCHAR(50), -- Ex: "Região Sacral (84%)"
    pico_intensidade_pct NUMERIC(5,2),
    area_contato_pct NUMERIC(5,2),
    indice_distribuicao NUMERIC(4,2),
    houve_alerta BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_eventos_sessao_data ON eventos_posturais(sessao_id, timestamp);

-- 4. RESUMO DIÁRIO E MÉTRICAS CONSOLIDADAS
CREATE TABLE metricas_resumo_diario (
    id BIGSERIAL PRIMARY KEY,
    sessao_id BIGINT REFERENCES sessoes_monitoramento(id),
    data_referencia DATE NOT NULL,
    total_rotacoes_realizadas INT DEFAULT 0,
    tempo_medio_por_postura_min INT DEFAULT 0,
    score_alivio_pressao_pct NUMERIC(5,2) DEFAULT 100.0,
    total_alertas_emitidos INT DEFAULT 0,
    total_alertas_atendidos INT DEFAULT 0,
    criado_em TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(sessao_id, data_referencia)
);

-- 5. SÉRIES TEMPORAIS AMOSTRADAS (Evolução 4h / 24h)
-- Armazena métricas decupadas a cada 1 minuto (não grava 15 FPS para poupar I/O e armazenamento)
CREATE TABLE telemetria_minuto (
    timestamp TIMESTAMPTZ NOT NULL,
    maca_id INT NOT NULL REFERENCES macas_dispositivos(id),
    peso_kg NUMERIC(5,2),
    indice_postural NUMERIC(4,2),
    tempo_estatico_seg INT,
    status_alerta BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_telemetria_maca_tempo ON telemetria_minuto(maca_id, timestamp DESC);
```

---

## 3. Política de Retenção e Ciclo de Vida dos Dados

1. **Telemetria Contínua (Minuto a Minuto):**
   * **0 a 30 dias:** Resolução de 1 minuto (para gráficos de evolução detalhados do paciente internado).
   * **31 a 180 dias:** Agregação horária (downsampling via view materializada).
   * **> 180 dias:** Expurgo da telemetria bruta se o paciente já teve alta.
2. **Eventos Posturais e Resumo Clínico:**
   * **Retenção:** Permanente durante a internação + 5 anos após a alta (atendendo exigências legais de auditoria médica e prontuário).

---

## 4. Segurança, LGPD e Conformidade Hospitalar

* **Criptografia em Repouso:** Tabelas criptografadas em disco via `LUKS` ou `TDE PostgreSQL`.
* **Criptografia em Trânsito:** Comunicação obrigatória via `TLS 1.3 / HTTPS` entre macas e servidor.
* **Anonimização (LGPD):** Os logs brutos de pressão e eventos de telemetria utilizam `internacao_id` e `codigo_paciente_pep`, mantendo os dados de identificação pessoal (PII) segregados e restritos à equipe assistencial com controle de acesso baseado em função (RBAC).
* **Backup & RPO/RTO:**
  * **RPO (Recovery Point Objective):** < 15 minutos (via replicação contínua WAL / Barman).
  * **RTO (Recovery Time Objective):** < 1 hora com failover automático.
