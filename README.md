# Painel de Previsão de Preços com ARIMA

Este projeto apresenta um painel interativo desenvolvido em **Python** com **Streamlit** para previsão de preços agropecuários utilizando o modelo **ARIMA** (AutoRegressive Integrated Moving Average).

O painel permite:

* Selecionar produtos agropecuários de uma base de dados CEPEA unificada;
* Definir o horizonte de previsão (6, 12, 24 ou 48 meses);
* Ajustar manualmente os parâmetros ARIMA (p, d, q) ou utilizar AutoARIMA para seleção automática;
* Visualizar gráficos interativos com intervalo de confiança;
* Avaliar o desempenho do modelo com a métrica RMSE (Root Mean Squared Error).

---

## 🎓 Contexto Acadêmico

Este painel foi desenvolvido como produto técnico-científico dos alunos **Lucas França** e **Paola Conti**, vinculados ao **Programa de Pós-Graduação em Economia** da **Universidade Federal de Santa Maria (UFSM)**.

### Objetivos Acadêmicos:

* Aplicar métodos quantitativos em séries temporais com foco no setor agropecuário;
* Criar ferramenta interativa de apoio à decisão com transparência metodológica;
* Desenvolver produto tecnológico replicável com fins de extensão ou aplicação prática.

> 📅 **Ano de desenvolvimento:** 2025
> 👩‍🎓 **Orientador(a):** Prof. Dr. Nome do Orientador
> 📄 Este projeto poderá ser incluído em anexo a um TCC, dissertação ou relatório final de projeto de pesquisa.

---

## 📊 Modelagem Utilizada

* **ARIMA(p,d,q)**: modelo estatístico para séries temporais univariadas.

  * `p`: número de termos autoregressivos
  * `d`: número de diferenças para tornar a série estacionária
  * `q`: número de termos de média móvel
* **AutoARIMA**: seleção automática dos parâmetros baseada em AIC/BIC
* **RMSE**: erro médio quadrático aplicado aos últimos 12 meses para medir a performance

---

## 🚀 Executando o Projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/painel-arima.git
cd painel-arima
```

### 2. Instalar as dependências

Crie um ambiente virtual e instale as dependências:

```bash
pip install -r requirements.txt
```

### 3. Rodar o painel

```bash
streamlit run app.py
```

---

## 🏆 Autoria

**Desenvolvido por:**

* Lucas França
* Paola Conti

**Instituição:** Universidade Federal de Santa Maria (UFSM)
**Curso:** Mestrado em Economia
**Ano:** 2025

---

## 💌 Contato

Para dúvidas, colaborações ou referências:
**[contato@ufsm.com.br](mailto:contato@ufsm.com.br)**

---

> Este é um projeto acadêmico. Dados e resultados apresentados têm finalidade didática e exploratória.
