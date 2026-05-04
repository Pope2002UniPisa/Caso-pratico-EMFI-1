# Caso Pratico EMFI 1
**Corso:** Economia dei Mercati Finanziari e dell'Informazione  
**Docente:** Prof. Fulvio Corsi – Università di Pisa  
**Autori:** Leonardo Pratelli, Sara Albotica  
**Consegna:** fulvio.corsi@unipi.it — oggetto: "Caso pratico EMFI 1"

---

## Struttura del progetto

```
Caso-pratico-EMFI-1/
├── EMFI_prova applicata_parte 1_2026.pdf   # Traccia ufficiale
├── requiremets.txt                          # Dipendenze Python
├── main.py                                  # Entry point (chiama le parti)
├── parte_A.py                               # Frontiera efficiente (punti 1–7)
├── parte_B.py                               # CAPM e Single Index Model (punti 8–9)
├── parte_C.py                               # Stabilità e bootstrap (punti 1–5)
├── latex/                                   # Report scritto (main.tex → main.pdf)
└── README.md                                # Questo file
```

---

## Installazione dipendenze

```bash
pip install -r requiremets.txt
```

Librerie usate:

| Libreria | Utilizzo |
|---|---|
| `yfinance` | Download dati storici da Yahoo Finance |
| `pandas` | Gestione serie temporali e DataFrame |
| `numpy` | Algebra lineare, calcoli matriciali |
| `scipy` | Ottimizzazione numerica (minimize) |
| `matplotlib` | Grafici (frontiera efficiente, ecc.) |
| `seaborn` | Heatmap della matrice di correlazione |

---

## Scelta degli asset (struttura 3-3-3-1)

La traccia richiede 10 titoli con buona diversificazione settoriale.  
La struttura scelta è **3 settori × 3 aziende + 1 indice**:

| Settore | Ticker | Azienda | Motivazione |
|---|---|---|---|
| **Tecnologia** | AAPL | Apple | Alta capitalizzazione, elevata crescita |
| | MSFT | Microsoft | Cloud + AI, stabile e crescente |
| | NVDA | Nvidia | Semiconduttori/AI, alta volatilità e rendimento |
| **Healthcare** | JNJ | Johnson & Johnson | Difensivo, dividendi stabili |
| | PFE | Pfizer | Farmaceutico ciclico (COVID + pipeline) |
| | MRK | Merck | Oncologia, meno ciclico di Pfizer |
| **Energy** | XOM | ExxonMobil | Major petrolifera USA, legata al prezzo del greggio |
| | CVX | Chevron | Diversificata, ottimo dividendo |
| | BP | BP | Europea, esposizione diversa da XOM/CVX |
| **Indice** | ^GSPC | S&P 500 | Proxy del portafoglio di mercato (usato nel CAPM) |

**Razionale della diversificazione:**  
Tech e Healthcare hanno bassa correlazione storica (settori growth vs difensivo).  
Energy è legata al ciclo delle materie prime e ha dinamiche quasi indipendenti.  
L'S&P 500 viene usato come benchmark di mercato per la stima dei beta (Parte B).

---

## Parte A – Frontiera Efficiente (`parte_A.py`)

### Punto 1 – Download dati e statistiche descrittive

**Cosa si fa:**  
Si scaricano i prezzi mensili *adjusted* (rettificati per dividendi e split) di tutti e 10 i titoli per il periodo **gennaio 2015 – gennaio 2025**, ottenendo circa 120 osservazioni mensili come richiesto dalla traccia.

**Passaggi nel codice:**

1. **`download_prices()`**  
   - Usa `yf.download()` con `auto_adjust=True` per ottenere i prezzi *total return* (dividendi inclusi).  
   - Aggrega i dati giornalieri a mensili con `.resample("ME").last()`: prende il prezzo di chiusura dell'ultimo giorno di borsa di ogni mese.  
   - Rinomina `^GSPC` in `SP500` per evitare problemi nei nomi delle colonne.

2. **`compute_returns()`**  
   - Calcola i **rendimenti logaritmici** mensili: `r_t = ln(P_t / P_{t-1})`  
   - Perché logaritmici? Sono additivi nel tempo, simmetrici e approssimano meglio la normalità rispetto ai rendimenti semplici — ipotesi necessaria per la teoria di Markowitz.

3. **`compute_statistics()`**  
   - **Vettore dei rendimenti medi** `μ`: media campionaria di ogni serie di rendimenti.  
   - **Matrice di varianza-covarianza** `Σ`: misura quanto i rendimenti variano insieme. L'elemento diagonale è la varianza di ogni titolo; gli elementi off-diagonali misurano la co-movimentazione tra coppie di titoli. Pandas usa il divisore `(T-1)` (stima non distorta).  
   - **Matrice di correlazione** `ρ`: versione normalizzata di `Σ`, valori in `[-1, +1]`. Rho vicino a 0 tra settori diversi → forte diversificazione.

4. **`print_summary()`**  
   - Stampa le statistiche annualizzate (× 12 per i rendimenti, × √12 per le deviazioni standard), valide sotto l'ipotesi i.i.d. dei rendimenti mensili.  
   - Calcola anche uno Sharpe Ratio grezzo (con `rf = 0`) come primo confronto tra i titoli.

5. **`plot_correlation()`**  
   - Produce una heatmap della matrice di correlazione (salvata in `correlation_heatmap.png`).  
   - Colori: rosso = correlazione alta (si muovono insieme), bianco = indipendenti, blu = correlazione negativa.

---

## Parte B – CAPM e Single Index Model (`parte_B.py`)

*In sviluppo.*

**Cosa si farà:**
- Stima OLS dei beta di ciascun titolo rispetto all'S&P 500 (portafoglio di mercato).
- Test di significatività dei coefficienti (beta e alpha di Jensen).
- Ricostruzione della matrice varianza-covarianza tramite Single Index Model.
- Confronto della frontiera efficiente SIM vs frontiera empirica della Parte A.

---

## Parte C – Stabilità e Bootstrap (`parte_C.py`)

*In sviluppo.*

**Cosa si farà:**
- Bootstrap i.i.d. con 500 campioni di T = 120 mesi (con reinserimento).
- Per N = 5 e N = 10 asset: stima del portafoglio di tangenza su ciascun campione.
- Analisi della distribuzione dei pesi e dello Sharpe Ratio al variare di N/T.
- Conclusione: al crescere di N (con T fisso) l'errore di stima della covarianza esplode → i pesi diventano instabili.

---

## Note metodologiche

- **Prezzi adjusted vs unadjusted:** Yahoo Finance con `auto_adjust=True` restituisce prezzi già rettificati — non è necessario scaricare una colonna separata.  
- **Frequenza mensile:** il campionamento mensile riduce il rumore del dato giornaliero e rende i rendimenti più vicini alla normalità, migliorando le stime della frontiera.  
- **Short selling:** nella Parte A si costruisce prima la frontiera senza vincoli (short selling ammesso) e poi con il vincolo di non-negatività dei pesi (no short selling), per confrontare le due.
