#
#   Settore 1 – Tecnologia (alta crescita, alta volatilità):
#       AAPL (Apple), MSFT (Microsoft), NVDA (Nvidia)
#
#   Settore 2 – Healthcare (difensivo, bassa correlazione con Tech/Energy):
#       JNJ (Johnson & Johnson), PFE (Pfizer), MRK (Merck)
#
#   Settore 3 – Energy (ciclico, legato al prezzo del petrolio):
#       XOM (ExxonMobil), CVX (Chevron), BP (BP)
#
#   Indice di mercato (benchmark per il CAPM nelle Parti B e C):
#       ^GSPC (S&P 500)
#
# La diversificazione tra Tech, Healthcare ed Energy garantisce correlazioni
# basse tra i settori, allargando la frontiera e migliorando il trade-off
# rischio-rendimento disponibile.
#
# LIBRERIE UTILIZZATE
# --------------------
#   yfinance  
#   pandas    
#   numpy     
#   matplotlib / seaborn 
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend non-interattivo: salva su file senza aprire finestre
import matplotlib.pyplot as plt
import seaborn as sns


# ── Configurazione asset e orizzonte temporale ─────────────────────────────────
#
# ALL_TICKERS: lista ordinata dei 10 ticker che useremo per l'intero progetto.
# START / END: finestra temporale di 10 anni → ~120 osservazioni mensili.
# Il dataset copre sia periodi di rialzo (bull market post-2015) che crisi
# (COVID 2020, rialzo tassi 2022), rendendo il campione statisticamente ricco.

TICKERS = {
    "Tech":       ["AAPL", "MSFT", "NVDA"],
    "Healthcare": ["JNJ",  "PFE",  "MRK"],
    "Energy":     ["XOM",  "CVX",  "BP"],
    "Index":      ["^GSPC"],
}
ALL_TICKERS = [t for group in TICKERS.values() for t in group]

START = "2015-01-01"
END   = "2025-01-01"


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 1 – download_prices
# ══════════════════════════════════════════════════════════════════════════════
def download_prices(tickers, start, end):
    """
    Scarica i prezzi storici giornalieri da Yahoo Finance e li aggrega
    a frequenza mensile prendendo l'ultimo giorno disponibile di ogni mese.

    Parametri
    ---------
    tickers : list[str]  – lista dei ticker Yahoo Finance
    start   : str        – data di inizio (YYYY-MM-DD)
    end     : str        – data di fine   (YYYY-MM-DD)

    Ritorna
    -------
    prices : pd.DataFrame
        DataFrame con indice mensile (ultimo giorno del mese) e una colonna
        per ciascun ticker. I prezzi sono già "adjusted" (dividendi e split
        inclusi) grazie al parametro auto_adjust=True di yfinance.

    Note tecniche
    -------------
    - auto_adjust=True: yfinance restituisce direttamente i prezzi rettificati
      per dividendi e split azionari (equivalente al "total return" richiesto
      dalla traccia). Non serve una colonna separata "Adj Close".
    - resample("ME").last(): aggrega i dati giornalieri a mensili prendendo
      il prezzo di chiusura dell'ultimo giorno di borsa aperta di ogni mese.
      "ME" sta per Month End (fine mese).
    - Il ticker ^GSPC viene rinominato "SP500" per evitare problemi con il
      carattere "^" in operazioni successive su colonne del DataFrame.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance restituisce un MultiIndex: prendiamo solo la colonna "Close"
    prices = raw["Close"].resample("ME").last()

    # Rinomina ^GSPC → SP500 per comodità
    prices.columns = [c if c != "^GSPC" else "SP500" for c in prices.columns]

    # Rimuove eventuali righe completamente vuote (es. fine dataset)
    prices.dropna(how="all", inplace=True)

    return prices


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 2 – compute_returns
# ══════════════════════════════════════════════════════════════════════════════
def compute_returns(prices):
    """
    Calcola i rendimenti logaritmici mensili a partire dai prezzi.

    Rendimento logaritmico: r_t = ln(P_t / P_{t-1})

    Perché i rendimenti logaritmici?
    ---------------------------------
    1. Sono additivi nel tempo: il rendimento su N periodi è la somma dei
       rendimenti periodali → comodo per aggregazioni.
    2. Approssimano bene i rendimenti semplici per variazioni piccole.
    3. Sono simmetrici: una perdita del 50% e un guadagno del 100% si
       compensano esattamente (non è vero con i rendimenti semplici).
    4. La distribuzione dei log-rendimenti è più vicina alla normalità,
       ipotesi spesso richiesta nei modelli di ottimizzazione di portafoglio.

    Parametri
    ---------
    prices : pd.DataFrame – prezzi mensili (adj close)

    Ritorna
    -------
    returns : pd.DataFrame – rendimenti logaritmici mensili (N-1 righe)
    """
    # np.log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
    # .shift(1) sposta i prezzi di un periodo verso il basso
    # .dropna() elimina la prima riga (NaN perché non esiste P_{t-1})
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 3 – compute_statistics
# ══════════════════════════════════════════════════════════════════════════════
def compute_statistics(returns):
    """
    Calcola le statistiche fondamentali del portafoglio:
      - Vettore dei rendimenti medi mensili (mu)
      - Matrice di varianza-covarianza (Sigma)
      - Matrice di correlazione (rho)

    Matrice di varianza-covarianza (Sigma)
    ---------------------------------------
    Elemento (i,j): Cov(r_i, r_j) = E[(r_i - mu_i)(r_j - mu_j)]
    Elemento diagonale (i,i): Var(r_i) = sigma_i^2
    Pandas usa il divisore (T-1) per la stima campionaria (non distorta).

    Matrice di correlazione (rho)
    ------------------------------
    rho_{ij} = Cov(r_i, r_j) / (sigma_i * sigma_j)
    Valori in [-1, +1]. Indica quanto i titoli si muovono insieme.
    Un rho basso tra settori diversi è il fondamento della diversificazione.

    Ritorna
    -------
    mean_returns : pd.Series      – rendimenti medi mensili per asset
    cov_matrix   : pd.DataFrame   – matrice varianza-covarianza (N×N)
    corr_matrix  : pd.DataFrame   – matrice di correlazione (N×N)
    """
    mean_returns = returns.mean()        # mu_i = (1/T) * sum(r_{i,t})
    cov_matrix   = returns.cov()         # Sigma: stima campionaria con (T-1)
    corr_matrix  = returns.corr()        # rho: normalizzazione per std dev
    return mean_returns, cov_matrix, corr_matrix


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 4 – plot_correlation
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation(corr_matrix):
    """
    Visualizza la matrice di correlazione come heatmap colorata.

    La heatmap usa una scala cromatica divergente (coolwarm):
      - Rosso scuro  → correlazione vicina a +1 (si muovono insieme)
      - Bianco       → correlazione vicina a  0 (indipendenti)
      - Blu scuro    → correlazione vicina a -1 (si muovono in senso opposto)

    Un buon portafoglio diversificato mostra blocchi off-diagonal chiari
    (bassa correlazione tra settori diversi).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,          # scrive il valore numerico in ogni cella
        fmt=".2f",           # formato: 2 decimali
        cmap="coolwarm",     # palette divergente centrata su 0
        center=0,            # 0 → bianco (nessuna correlazione)
        square=True,         # celle quadrate per leggibilità
        linewidths=0.5,      # bordi sottili tra celle
        ax=ax,
        annot_kws={"size": 9}
    )
    ax.set_title("Matrice di Correlazione – Rendimenti Mensili (2015–2025)", fontsize=13)
    fig.tight_layout()
    fig.savefig("correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("Heatmap salvata: correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 5 – print_summary
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(mean_returns, cov_matrix, corr_matrix, returns):
    """
    Stampa a console un riepilogo delle statistiche calcolate.

    Annualizzazione
    ---------------
    Passando da mensile ad annuale si usano le seguenti relazioni:
      - Rendimento atteso annuo   = mu_mensile  * 12
      - Deviazione standard annua = sigma_mensile * sqrt(12)
    Questo vale sotto l'ipotesi che i rendimenti mensili siano i.i.d.
    (indipendenti e identicamente distribuiti), standard nella teoria
    classica di portafoglio di Markowitz.

    Sharpe Ratio (con rf = 0 come primo riferimento)
    --------------------------------------------------
    SR = (mu_annuo - rf) / sigma_annuo
    Con rf = 0 è un indice grezzo di rendimento per unità di rischio.
    Nella Parte A (punto 6) useremo il tasso risk-free reale (BTP a breve).
    """
    ann = 12  # fattore di annualizzazione (12 mesi)

    print("=" * 65)
    print("STATISTICHE DESCRITTIVE (mensili → annualizzate)")
    print("=" * 65)
    summary = pd.DataFrame({
        "Rend. medio mensile (%)":  (mean_returns * 100).round(3),
        "Rend. atteso annuo (%)":   (mean_returns * ann * 100).round(2),
        "Std Dev mensile (%)":      (returns.std() * 100).round(3),
        "Std Dev annua (%)":        (returns.std() * np.sqrt(ann) * 100).round(2),
        "Sharpe (annuo, rf=0)":     ((mean_returns * ann) / (returns.std() * np.sqrt(ann))).round(3),
    })
    print(summary.to_string())

    print("\n" + "=" * 65)
    print("MATRICE DI VARIANZA-COVARIANZA (mensile, valori x100)")
    print("=" * 65)
    # Moltiplichiamo per 100 per leggibilità (ordine di grandezza ~0.01)
    print((cov_matrix * 100).round(4).to_string())

    print("\n" + "=" * 65)
    print("MATRICE DI CORRELAZIONE")
    print("=" * 65)
    print(corr_matrix.round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Download dati: {', '.join(ALL_TICKERS)}  [{START} → {END}]")

    # Step 1: scarica i prezzi mensili adjusted
    prices = download_prices(ALL_TICKERS, START, END)

    # Step 2: calcola rendimenti logaritmici mensili
    returns = compute_returns(prices)

    print(f"Osservazioni mensili disponibili: {len(returns)}")
    print(f"Periodo effettivo: {returns.index[0].date()} → {returns.index[-1].date()}\n")

    # Step 3: calcola statistiche (mu, Sigma, rho)
    mean_returns, cov_matrix, corr_matrix = compute_statistics(returns)

    # Step 4: stampa riepilogo e mostra heatmap
    print_summary(mean_returns, cov_matrix, corr_matrix, returns)
    plot_correlation(corr_matrix)
