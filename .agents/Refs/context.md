# Contexto e Arquitetura do Projeto: Matriz de Sensores FSR

Este documento resume as especificações de hardware, a modelagem matemática e os requisitos de software para a construção do sistema de captação e processamento de dados de uma matriz de sensores de pressão FSR.

---

## 1. Topologia de Hardware e Captação

O projeto consiste em uma matriz de Force Sensing Resistors (FSR). Atualmente operando em um **protótipo 3x3**, mas a arquitetura de software deve ser construída para escalar nativamente para a matriz real de **16x16**.

*   **Varredura Ativa (Polling):** O microcontrolador (código em C) energiza uma linha por vez e varre as colunas usando um multiplexador analógico (CD74HC4067).
*   **Circuito de Leitura:** Utiliza um divisor de tensão coletivo. Os sensores FSR estão no polo positivo e há um resistor fixo de pull-down de **10kΩ** no polo negativo (conectado ao GND).
*   **Transmissão:** Os dados são enviados do microcontrolador para o PC via Serial/USB HID em pacotes ("frames" da matriz completa) a uma taxa de **50 a 100Hz**.
*   **Dados Brutos:** O PC recebe apenas os valores crus do ADC (ex: 0 a 1023 para 10-bit).

---

## 2. Modelagem Matemática (Backend Python)

Como o hardware lê um sensor por vez, mas envia o "frame" completo, o backend em Python deve usar a biblioteca **NumPy** para aplicar cálculos de forma vetorizada na matriz inteira instantaneamente, evitando gargalos de loop.

O pipeline matemático para calcular o peso real a partir do valor bruto do ADC segue estes passos:

### Cálculo de Tensão e Resistência
Conversão da leitura bruta para tensão ($V_{out}$) e descoberta da resistência do sensor pressionado ($R_{FSR}$):
$$V_{out} = Valor_{ADC} \cdot \left( \frac{V_{cc}}{Resolução_{ADC}} \right)$$
$$R_{FSR} = R_{fixo} \cdot \left( \frac{V_{cc}}{V_{out}} - 1 \right)$$

### Conversão para Condutância e Força
A condutância ($C$) possui resposta quase linear em relação à força, permitindo o uso da equação da reta ($y = mx + b$) com coeficientes de calibração empíricos ($m$ e $b$):
$$C = \frac{1}{R_{FSR}}$$
$$F(x,y) = m \cdot C + b$$

### Filtragem e Massa Total
Para mitigar correntes de fuga (crosstalk), aplica-se uma Zona Morta (deadzone), zerando forças abaixo de um limiar mínimo. A massa total é calculada somando todas as coordenadas e dividindo pela gravidade ($g$):
$$Massa_{Total} = \frac{\sum_{x} \sum_{y} F(x,y)}{9.81}$$

---

## 3. Requisitos de Regra de Negócio

*   **Filtro de Estabilização:** A massa final deve passar por um filtro de Média Móvel Exponencial (EMA) para suavizar a leitura na interface gráfica.
*   **Monitoramento de Posição (Alerta):** O sistema deve analisar continuamente o mapa de força (distribuição de pressão). Se a pessoa mantiver a mesma distribuição de peso/centro de massa (dentro de uma margem de tolerância) por **1 minuto ininterrupto**, o sistema deve emitir um alerta indicando inatividade prolongada na mesma postura.

---

## 4. Estrutura do Software Front/Back

*   **Linguagem Base:** Python.
*   **Diretório:** Todos os códigos e dependências Python devem ficar restritos à pasta `/Python`.
*   **Interface Gráfica (UI):** Deve ser construída utilizando uma tecnologia de **Webview** (ex: `Eel`, `pywebview` ou `Dash/Streamlit`).
*   **Elementos da UI:**
    *   Mapa de calor (Heatmap) renderizando o espectro de pressão em tempo real (escala automática do protótipo 3x3 para 16x16).
    *   Exibição numérica do Peso Total (Kg).
    *   Cronômetro de inatividade / postura estática, acoplado ao alerta visual de 1 minuto.
    *   Painel interativo para edição de variáveis de calibração em tempo real (VCC, Resolução ADC, Resistor Fixo, Fator M, Offset B e limiar de Zona Morta).