# Caso Pratico EMFI 1
**Corso:** Economia dei Mercati Finanziari e dell'Informazione  
**Docente:** Prof. Fulvio Corsi – Università di Pisa  
**Autori:** Leonardo Pratelli  
**Consegna:** fulvio.corsi@unipi.it — oggetto: "Caso pratico EMFI 1"

---

## Struttura del progetto

```
Caso-pratico-EMFI-1/
├── EMFI_prova applicata_parte 1_2026.pdf   # Traccia ufficiale
├── requirements.txt                         # Dipendenze Python
├── main.py                                  # Entry point (lancia A, B o C)
├── README.md
├── parte_A/                                 # Frontiera Efficiente (Punti 1–7)
│   ├── notebook_A.ipynb                     # Notebook interattivo
│   ├── parte_A.py                           # Script
│   └── *.png                                # Grafici generati
├── parte_B/                                 # CAPM e Single Index Model (Punti 8–9)
│   ├── notebook_B.ipynb                     # Notebook interattivo
│   ├── parte_B.py                           # Script
│   └── *.png
├── parte_C/                                 # Stabilità e Bootstrap (Punti 1–5)
│   ├── notebook_C.ipynb                     # Notebook interattivo
│   ├── parte_C.py                           # Script
│   └── *.png
└── latex/                                   # Report scritto (main.tex → main.pdf)
```

---

## Installazione dipendenze

```bash
pip install -r requirements.txt
```

| Libreria | Utilizzo |
|---|---|
| `yfinance` | Download dati storici da Yahoo Finance |
| `pandas` | Gestione serie temporali e DataFrame |
| `numpy` | Algebra lineare, calcoli matriciali |
| `scipy` | Ottimizzazione numerica e test statistici OLS |
| `matplotlib` | Grafici (frontiera efficiente, SML, CML, ecc.) |
| `seaborn` | Heatmap matrici di correlazione |

---

## Scelta degli asset (struttura 3-3-3-1)

| Settore | Ticker | Azienda | Motivazione |
|---|---|---|---|
| **Tecnologia** | AAPL | Apple | Alta capitalizzazione, elevata crescita |
| | MSFT | Microsoft | Cloud + AI, stabile e crescente |
| | NVDA | Nvidia | Semiconduttori/AI, alta volatilità e rendimento |
| **Healthcare** | JNJ | Johnson & Johnson | Difensivo, dividendi stabili |
| | PFE | Pfizer | Farmaceutico ciclico |
| | MRK | Merck | Oncologia, bassa correlazione con tech (ρ=0.04 con NVDA) |
| **Energy** | XOM | ExxonMobil | Major petrolifera USA |
| | CVX | Chevron | Diversificata, ottimo dividendo |
| | BP | BP | Europea, esposizione diversa da XOM/CVX |
| **Indice** | ^GSPC | S&P 500 | Proxy portafoglio di mercato (CAPM) |

**5 titoli selezionati per la frontiera (Parte A):** NVDA, MSFT, JNJ, MRK, XOM  
Razionale: massima diversificazione settoriale (tech + healthcare + energy) con correlazioni inter-settore basse.

---

## Parte A – Frontiera Efficiente (`parte_A/`)

| Blocco | Punto traccia | Contenuto | Output PNG |
|--------|---------------|-----------|------------|
| 0–2    | —             | Import, download, rendimenti logaritmici | — |
| 3      | 1             | Statistiche descrittive annualizzate | — |
| 4      | 1             | Matrice varianza-covarianza + heatmap | `covariance_heatmap.png` |
| 5      | 1             | Matrice di correlazione + heatmap | `correlation_heatmap.png` |
| 6      | 1             | Scatter rischio vs rendimento (10 asset) | `risk_return_scatter.png` |
| 7      | 2             | Selezione 5 titoli, scalari Markowitz, GMV | — |
| 8      | 2             | Frontiera analitica + numerica (50 punti) | `efficient_frontier_A2.png` |
| 9      | 3             | Frontiera long-only (no short selling) | `efficient_frontier_A3_comparison.png` |
| 10     | 4             | Portafoglio di mercato vs frontiera | `market_portfolio_A4.png` |
| 11     | 5             | GMV, EW5, EW10 + tabella confronto | `benchmarks_A5.png` |
| 12     | 6             | CML e portafoglio di tangenza | `tangency_CML_A6.png` |
| 13     | 7             | Portafoglio ottimo con A=3, curva indifferenza | `optimal_portfolio_A7.png` |

**Formule chiave:**
- Frontiera analitica: $\sigma^2(\mu_p) = (C\mu_p^2 - 2A\mu_p + B)/D$ con scalari $A=\mathbf{1}'\Sigma^{-1}\boldsymbol{\mu},\ B=\boldsymbol{\mu}'\Sigma^{-1}\boldsymbol{\mu},\ C=\mathbf{1}'\Sigma^{-1}\mathbf{1}$
- Tangenza: $\mathbf{w}_{tan} = \Sigma^{-1}(\mu - r_f\mathbf{1})\ /\ [\mathbf{1}'\Sigma^{-1}(\mu - r_f\mathbf{1})]$
- Portafoglio ottimo: $y^* = (\mu_{tan} - r_f)\ /\ (A \cdot \sigma^2_{tan})$

---

## Parte B – CAPM e Single Index Model (`parte_B/`)

| Blocco | Punto traccia | Contenuto | Output PNG |
|--------|---------------|-----------|------------|
| 0–1    | —             | Import, configurazione, download dati | — |
| 2      | 8             | Stima OLS beta e alpha (tutti i 10 titoli) + tabella significatività | — |
| 3      | 8             | Bar chart beta e alpha di Jensen | `beta_alpha_B8.png` |
| 4      | 8             | Security Market Line (SML) | `SML_B8.png` |
| 5      | 9             | Matrice Σ_SIM + confronto con Σ empirica | `correlation_SIM_vs_empirical_B9.png` |
| 6      | 9             | Rendimenti attesi CAPM vs empirici | `capm_vs_empirical_B9.png` |
| 7      | 9             | Frontiera SIM/CAPM vs frontiera empirica | `frontier_SIM_vs_empirical_B9.png` |

**Formule chiave:**
- OLS: $\hat{\beta}_i = \text{Cov}(r_i, r_m)/\text{Var}(r_m)$,\ $\hat{\alpha}_i = \bar{r}_i - \hat{\beta}_i\bar{r}_m$
- SIM covarianza: $\Sigma_{\text{SIM}} = \boldsymbol{\beta}\boldsymbol{\beta}'\sigma^2_m + \mathbf{D}$
- CAPM rendimenti: $\mathbb{E}[r_i] = r_f + \beta_i(\mu_m - r_f)$

---

## Parte C – Stabilità e Bootstrap (`parte_C/`)

Bootstrap i.i.d. con B=500 campioni (T=120 mesi) per due sottoinsiemi:
- **N=5**: gli stessi 5 titoli selezionati nella Parte A (NVDA, MSFT, JNJ, MRK, XOM)
- **N=10**: tutti i 10 titoli/indici

Analisi della stabilità dei pesi del portafoglio di tangenza e distribuzione dello Sharpe Ratio al variare del rapporto N/T.

| Punto | Contenuto | Output PNG |
|-------|-----------|------------|
| 3–4 (N=5)  | Boxplot pesi + istogramma SR bootstrap N=5 | `bootstrap_weights_N5_C4.png`, `bootstrap_sharpe_N5_C4.png` |
| 3–4 (N=10) | Boxplot pesi + istogramma SR bootstrap N=10 | `bootstrap_weights_N10_C4.png`, `bootstrap_sharpe_N10_C4.png` |
| 5 | Confronto stabilità pesi e distribuzione SR: N=5 vs N=10 | `bootstrap_stability_comparison_C5.png`, `bootstrap_sharpe_comparison_C5.png` |
