# Caso Pratico EMFI 1
**Autori:** Leonardo Pratelli e Sara Albotica
**Consegna:** fulvio.corsi@unipi.it

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