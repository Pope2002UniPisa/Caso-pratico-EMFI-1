# =============================================================================
# CASO PRATICO EMFI 1 – PARTE C: Stabilità del portafoglio di tangenza
# Autori: Leonardo Pratelli – Università di Pisa
# =============================================================================
#
# OBIETTIVO
# ---------
# Analizzare la stabilità dei portafogli ottimali rispetto all'errore di stima
# e studiare il ruolo del rapporto tra numero di asset (N) e numero di
# osservazioni (T), tramite bootstrap i.i.d.
#
#   Punto 1 – Fissare T = 120 mesi (intero campione).
#
#   Punto 2 – Due casi:
#       Caso 1: N = 5  (primi 5 titoli in ordine alfabetico)
#       Caso 2: N = 10 (tutti i 10 titoli)
#
#   Punto 3 – Per ciascun caso, bootstrap i.i.d.:
#       - 500 campioni di T = 120 mesi con reinserimento
#       - Per ogni campione: stima mu e Sigma, calcola portafoglio di tangenza
#         usando la formula analitica (short selling ammesso):
#
#           w_tan = Sigma^{-1}(mu - rf*iota) / [iota' Sigma^{-1}(mu - rf*iota)]
#
#         Il portafoglio di tangenza massimizza lo Sharpe Ratio.
#         Lo Sharpe Ratio annualizzato del portafoglio:
#
#           SR = (mu_tan - rf) / sigma_tan * sqrt(12)
#
#   Punto 4 – Per ciascun caso:
#       - Media e deviazione standard dei pesi (tabella + boxplot)
#       - Distribuzione dello Sharpe Ratio (istogramma)
#
#   Punto 5 – Commento:
#       - Come cambia la stabilità dei pesi al crescere di N (T fisso)?
#       - Quale è il ruolo del rapporto N/T?
#
# CHIAVE CONCETTUALE
# ------------------
# Con T fisso (120 mesi) e N crescente, il rapporto N/T aumenta:
#   N=5:  N/T = 5/120 ≈ 0.042  → molti gradi di libertà, stime stabili
#   N=10: N/T = 10/120 ≈ 0.083 → stima di Sigma più difficile, ottimizzatore
#                                  può amplificare errori di stima
# Per N → T la matrice Sigma diventa quasi singolare e i pesi esplodono.
# Questo è il problema dell'"errore di stima" in portafoglio: la frontiera
# efficiente è estremamente sensibile alle stime di mu e Sigma.
#
# LIBRERIE
# ---------
#   yfinance   → download dati Yahoo Finance
#   pandas     → gestione DataFrame
#   numpy      → algebra lineare, bootstrap
#   scipy      → linalg (solve, inv)
#   matplotlib → grafici (boxplot, istogrammi, bar chart)
#
# COME ESEGUIRE
# -------------
#   cd parte_C && python parte_C.py
#   I grafici vengono salvati nella cartella parte_C/ (stessa del file).
# =============================================================================

import matplotlib
matplotlib.use("Agg")   # backend non-interattivo: salva su file senza finestre

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import linalg

# ── Configurazione (identica alle Parti A e B) ─────────────────────────────────

TICKERS = {
    "Tech":       ["AAPL", "MSFT", "NVDA"],
    "Healthcare": ["JNJ",  "PFE",  "MRK"],
    "Energy":     ["XOM",  "CVX",  "BP"],
    "Index":      ["^GSPC"],
}
ALL_TICKERS = [t for group in TICKERS.values() for t in group]

rf_annual = 0.02        # tasso risk-free annuo: proxy BOT 3 mesi 2015-2025
ann       = 12          # fattore di annualizzazione
rf        = rf_annual / ann    # tasso risk-free mensile

START = "2015-01-01"
END   = "2025-01-01"

T_BOOT = 120    # lunghezza di ogni campione bootstrap (mesi)
B      = 500    # numero di campioni bootstrap

np.random.seed(42)    # riproducibilità


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 1 – download_and_returns
# ══════════════════════════════════════════════════════════════════════════════
def download_and_returns(tickers, start, end):
    """
    Scarica i prezzi mensili adjusted da Yahoo Finance e calcola i rendimenti
    logaritmici mensili. Identica alla Parte A e B per coerenza.

    Ritorna
    -------
    returns : pd.DataFrame – rendimenti logaritmici mensili per tutti i titoli.
    """
    raw    = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)
    prices = raw["Close"].resample("ME").last()
    prices.columns = [c if c != "^GSPC" else "SP500" for c in prices.columns]
    prices.dropna(how="all", inplace=True)

    # Rendimenti logaritmici: r_t = ln(P_t / P_{t-1})
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 2 – tangency_portfolio
# ══════════════════════════════════════════════════════════════════════════════
def tangency_portfolio(mu, Sigma, rf):
    """
    Calcola il portafoglio di tangenza con la formula analitica (short selling
    ammesso), derivata massimizzando lo Sharpe Ratio.

    Formula (eq. 23 slides, con vincolo iota'w = 1):
        z     = Sigma^{-1} (mu - rf * iota)
        w_tan = z / (iota' z)

    Parametri
    ---------
    mu    : (N,) array – rendimenti attesi (mensili)
    Sigma : (N,N) array – matrice varianza-covarianza (mensile)
    rf    : float – tasso risk-free mensile

    Ritorna
    -------
    w   : (N,) array – pesi del portafoglio di tangenza (sum = 1)
    SR  : float – Sharpe Ratio annualizzato
    ok  : bool – False se il denominatore è ≤ 0 (excess return tutti negativi)
    """
    N    = len(mu)
    iota = np.ones(N)
    exc  = mu - rf * iota           # excess return rispetto al risk-free

    # Risolvi Sigma * z = exc (più stabile di inv(Sigma) @ exc)
    try:
        z = linalg.solve(Sigma, exc, assume_a="pos")
    except linalg.LinAlgError:
        return np.full(N, np.nan), np.nan, False

    denom = iota @ z    # = iota' Sigma^{-1} (mu - rf*iota)

    # Se il denominatore è negativo, il portafoglio di tangenza è sull'asse
    # opposto (tutti gli asset hanno rendimento atteso < rf): non ha senso
    # economico. Segnaliamo come campione invalido.
    if denom <= 0:
        return np.full(N, np.nan), np.nan, False

    w = z / denom

    # Sharpe Ratio annualizzato del portafoglio di tangenza
    mu_p  = float(w @ mu)
    sig_p = float(np.sqrt(w @ Sigma @ w))
    SR    = (mu_p - rf) / sig_p * np.sqrt(ann)

    return w, SR, True


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 3 – bootstrap_tangency
# ══════════════════════════════════════════════════════════════════════════════
def bootstrap_tangency(returns_df, rf, T_boot, B):
    """
    Bootstrap i.i.d. dei rendimenti: genera B campioni di T_boot osservazioni
    ciascuno (campionamento con reinserimento) e stima il portafoglio di
    tangenza per ogni campione.

    Parametri
    ---------
    returns_df : pd.DataFrame – rendimenti storici (T_tot × N)
    rf         : float – tasso risk-free mensile
    T_boot     : int   – lunghezza di ogni campione bootstrap
    B          : int   – numero di campioni bootstrap

    Ritorna
    -------
    weights_boot : (B, N) array – pesi tangenza per ogni campione (NaN se fallisce)
    sharpes_boot : (B,) array  – Sharpe ratio per ogni campione
    n_valid      : int         – numero di campioni con soluzione valida
    """
    T_tot = len(returns_df)
    N     = returns_df.shape[1]
    ret   = returns_df.values    # (T_tot, N) numpy array per velocità

    weights_boot = np.full((B, N), np.nan)
    sharpes_boot = np.full(B, np.nan)

    for b in range(B):
        # Campionamento con reinserimento: seleziona T_boot indici in [0, T_tot)
        idx    = np.random.randint(0, T_tot, size=T_boot)
        sample = ret[idx, :]              # (T_boot, N) campione bootstrap

        # Stima mu e Sigma sul campione bootstrap
        mu_b    = sample.mean(axis=0)    # (N,) rendimento medio
        # ddof=1: varianza campionaria (non distorta)
        Sigma_b = np.cov(sample, rowvar=False, ddof=1)   # (N, N)

        w, SR, ok = tangency_portfolio(mu_b, Sigma_b, rf)
        if ok:
            weights_boot[b, :] = w
            sharpes_boot[b]    = SR

    n_valid = np.sum(~np.isnan(sharpes_boot))
    return weights_boot, sharpes_boot, n_valid


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 4 – print_weight_table
# ══════════════════════════════════════════════════════════════════════════════
def print_weight_table(weights_boot, tickers, N):
    """
    Stampa la tabella di media e deviazione standard dei pesi bootstrap
    (Punto 4 della traccia).
    """
    valid = weights_boot[~np.isnan(weights_boot[:, 0]), :]
    means = np.nanmean(weights_boot, axis=0)
    stds  = np.nanstd(weights_boot,  axis=0)

    print(f"\nDistribuzione pesi portafoglio di tangenza – N={N} asset "
          f"({valid.shape[0]} campioni validi su {weights_boot.shape[0]})\n")
    tab = pd.DataFrame({
        "Ticker":       tickers,
        "Media pesi":   means.round(4),
        "Std Dev pesi": stds.round(4),
        "CV (std/|media|)": (stds / np.abs(means)).round(3),
    }).set_index("Ticker")
    print(tab.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 5 – plot_bootstrap_weights
# ══════════════════════════════════════════════════════════════════════════════
def plot_bootstrap_weights(weights_boot, tickers, N, fname):
    """
    Boxplot della distribuzione dei pesi del portafoglio di tangenza
    per ciascun asset, su B campioni bootstrap.

    Il boxplot mostra mediana, quartili e outlier: tanto più stretto il box,
    tanto più stabile è il peso di quell'asset.
    """
    # Rimuovi righe con NaN (campioni con denom ≤ 0)
    valid = weights_boot[~np.isnan(weights_boot[:, 0]), :]

    fig, ax = plt.subplots(figsize=(max(9, N * 1.1), 6))

    # Boxplot con whiskers a 5°–95° percentile per evidenziare gli outlier estremi
    bp = ax.boxplot([valid[:, i] for i in range(N)],
                    labels=tickers, patch_artist=True,
                    whis=[5, 95], showfliers=True)

    # Colori alternati per leggibilità
    colors = plt.cm.tab10(np.linspace(0, 1, N))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5, label="w = 0")
    ax.axhline(1/N, color="grey", lw=1, ls=":", alpha=0.7,
               label=f"Equally weighted (1/{N} = {1/N:.2f})")

    ax.set_title(f"Distribuzione pesi portafoglio di tangenza – N={N}\n"
                 f"Bootstrap i.i.d.: B={valid.shape[0]} campioni validi, T={T_BOOT} mesi",
                 fontsize=12)
    ax.set_ylabel("Peso portafoglio (w_i)", fontsize=11)
    ax.set_xlabel("Asset", fontsize=11)
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Salvato: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 6 – plot_bootstrap_sharpe
# ══════════════════════════════════════════════════════════════════════════════
def plot_bootstrap_sharpe(sharpes_boot, N, fname, sr_storico=None):
    """
    Istogramma della distribuzione dello Sharpe Ratio annualizzato del
    portafoglio di tangenza su B campioni bootstrap.

    Mostra anche il valore storico (SR calcolato sull'intero campione) per
    evidenziare l'errore di stima.
    """
    sr_valid = sharpes_boot[~np.isnan(sharpes_boot)]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(sr_valid, bins=40, color="steelblue", edgecolor="white",
            alpha=0.8, density=True, label=f"SR bootstrap (B={len(sr_valid)})")

    # Linee verticali: media e ±1 std
    mu_sr  = np.mean(sr_valid)
    std_sr = np.std(sr_valid)
    ax.axvline(mu_sr, color="navy", lw=2, ls="-",
               label=f"Media: {mu_sr:.3f}")
    ax.axvline(mu_sr + std_sr, color="navy", lw=1, ls="--", alpha=0.7,
               label=f"±1σ: [{mu_sr-std_sr:.2f}, {mu_sr+std_sr:.2f}]")
    ax.axvline(mu_sr - std_sr, color="navy", lw=1, ls="--", alpha=0.7)

    if sr_storico is not None:
        ax.axvline(sr_storico, color="tomato", lw=2.5, ls="-",
                   label=f"SR storico: {sr_storico:.3f}")

    ax.set_title(f"Distribuzione Sharpe Ratio portafoglio di tangenza – N={N}\n"
                 f"Bootstrap i.i.d. annualizzato  (T={T_BOOT} mesi per campione)",
                 fontsize=12)
    ax.set_xlabel("Sharpe Ratio annualizzato", fontsize=11)
    ax.set_ylabel("Densità", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Salvato: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 7 – plot_stability_comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_stability_comparison(stds5, stds10, tickers5, tickers10, fname):
    """
    Confronto della deviazione standard dei pesi tra N=5 e N=10.
    Una std alta = peso instabile = alta sensibilità all'errore di stima.

    Permette di visualizzare l'effetto del rapporto N/T sulla stabilità.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Deviazione standard pesi per N=5
    ax = axes[0]
    ax.bar(tickers5, stds5, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title(f"Instabilità pesi – N=5 (N/T={5/T_BOOT:.3f})", fontsize=12)
    ax.set_ylabel("Std Dev dei pesi bootstrap", fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(stds5):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    # Deviazione standard pesi per N=10
    ax = axes[1]
    ax.bar(tickers10, stds10, color="tomato", edgecolor="black", alpha=0.8)
    ax.set_title(f"Instabilità pesi – N=10 (N/T={10/T_BOOT:.3f})", fontsize=12)
    ax.set_ylabel("Std Dev dei pesi bootstrap", fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(stds10):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        "Confronto stabilità pesi portafoglio di tangenza: N=5 vs N=10\n"
        f"Bootstrap i.i.d. B={B} campioni, T={T_BOOT} mesi – "
        "una std maggiore indica maggiore sensibilità all'errore di stima",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Salvato: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 8 – plot_sharpe_comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_sharpe_comparison(sr5, sr10, sr_storico5, sr_storico10, fname):
    """
    Confronto diretto delle distribuzioni di Sharpe Ratio per N=5 vs N=10.
    Mostra come il rumore nella stima aumenta con N/T.
    """
    sr5_v  = sr5[~np.isnan(sr5)]
    sr10_v = sr10[~np.isnan(sr10)]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Istogrammi sovrapposti
    ax.hist(sr5_v,  bins=40, alpha=0.6, color="steelblue", density=True,
            label=f"N=5  (N/T={5/T_BOOT:.3f}),  σ(SR)={np.std(sr5_v):.3f}")
    ax.hist(sr10_v, bins=40, alpha=0.6, color="tomato",    density=True,
            label=f"N=10 (N/T={10/T_BOOT:.3f}), σ(SR)={np.std(sr10_v):.3f}")

    ax.axvline(np.mean(sr5_v),  color="navy",   lw=2, ls="-",
               label=f"Media N=5:  {np.mean(sr5_v):.3f}")
    ax.axvline(np.mean(sr10_v), color="darkred", lw=2, ls="-",
               label=f"Media N=10: {np.mean(sr10_v):.3f}")

    if sr_storico5 is not None:
        ax.axvline(sr_storico5, color="navy",   lw=1.5, ls=":",
                   label=f"SR storico N=5:  {sr_storico5:.3f}")
    if sr_storico10 is not None:
        ax.axvline(sr_storico10, color="darkred", lw=1.5, ls=":",
                   label=f"SR storico N=10: {sr_storico10:.3f}")

    ax.set_title("Confronto distribuzioni Sharpe Ratio: N=5 vs N=10\n"
                 f"Bootstrap B={B}, T={T_BOOT} – effetto del rapporto N/T",
                 fontsize=12)
    ax.set_xlabel("Sharpe Ratio annualizzato", fontsize=11)
    ax.set_ylabel("Densità", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Salvato: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Step 1: Download dati ────────────────────────────────────────────────
    print(f"Download: {ALL_TICKERS}  [{START} → {END}]")
    returns = download_and_returns(ALL_TICKERS, START, END)
    ALL_COLS = list(returns.columns)   # ordine alfabetico yfinance
    T_tot    = len(returns)
    print(f"Osservazioni: {T_tot} mesi  ({returns.index[0].date()} → {returns.index[-1].date()})")
    print(f"Tutti i titoli ({len(ALL_COLS)}): {ALL_COLS}")

    # ── Step 2: Definizione dei sottoinsiemi (Punto 2) ───────────────────────
    # "primi 5 titoli" = prime 5 colonne in ordine alfabetico (come restituisce yfinance)
    TICKERS_5  = ALL_COLS[:5]     # N=5: sottocaso più parsimo
    TICKERS_10 = ALL_COLS         # N=10: tutti i titoli
    print(f"\nCaso 1 – N=5  (primi 5): {TICKERS_5}")
    print(f"Caso 2 – N=10 (tutti):   {TICKERS_10}")
    print(f"T = {T_BOOT} mesi, B = {B} campioni bootstrap")
    print(f"Rapporto N/T: {5}/{T_BOOT}={5/T_BOOT:.3f}  vs  {10}/{T_BOOT}={10/T_BOOT:.3f}\n")

    # ── Step 3a: Portafoglio di tangenza storico (benchmark) ─────────────────
    # Calcolato sull'intero campione, usato come riferimento nei grafici

    # N=5
    mu5_stor  = returns[TICKERS_5].mean().values
    Sig5_stor = returns[TICKERS_5].cov().values
    _, sr5_stor,  ok5  = tangency_portfolio(mu5_stor,  Sig5_stor,  rf)

    # N=10
    mu10_stor  = returns[TICKERS_10].mean().values
    Sig10_stor = returns[TICKERS_10].cov().values
    _, sr10_stor, ok10 = tangency_portfolio(mu10_stor, Sig10_stor, rf)

    print(f"SR storico (intero campione): N=5 → {sr5_stor:.3f}  |  N=10 → {sr10_stor:.3f}")

    # ── Step 3b: Bootstrap N=5 (Punto 3) ────────────────────────────────────
    print(f"\nBootstrap N=5  ({B} campioni × {T_BOOT} mesi)...")
    ws5, srs5, n_valid5 = bootstrap_tangency(returns[TICKERS_5], rf, T_BOOT, B)
    print(f"Campioni validi: {n_valid5}/{B}")

    # ── Step 3c: Bootstrap N=10 (Punto 3) ───────────────────────────────────
    print(f"\nBootstrap N=10 ({B} campioni × {T_BOOT} mesi)...")
    ws10, srs10, n_valid10 = bootstrap_tangency(returns[TICKERS_10], rf, T_BOOT, B)
    print(f"Campioni validi: {n_valid10}/{B}")

    # ── Step 4a: Tabelle peso (Punto 4) ──────────────────────────────────────
    print_weight_table(ws5,  TICKERS_5,  5)
    print_weight_table(ws10, TICKERS_10, 10)

    # Statistiche Sharpe
    print(f"\nSharpe Ratio bootstrap N=5:  media={np.nanmean(srs5):.3f}, "
          f"std={np.nanstd(srs5):.3f}")
    print(f"Sharpe Ratio bootstrap N=10: media={np.nanmean(srs10):.3f}, "
          f"std={np.nanstd(srs10):.3f}")

    # ── Step 4b: Grafici pesi (Punto 4) ──────────────────────────────────────
    plot_bootstrap_weights(ws5,  TICKERS_5,  5,  "bootstrap_weights_N5_C4.png")
    plot_bootstrap_weights(ws10, TICKERS_10, 10, "bootstrap_weights_N10_C4.png")

    # ── Step 4c: Grafici Sharpe (Punto 4) ────────────────────────────────────
    plot_bootstrap_sharpe(srs5,  5,  "bootstrap_sharpe_N5_C4.png",  sr5_stor)
    plot_bootstrap_sharpe(srs10, 10, "bootstrap_sharpe_N10_C4.png", sr10_stor)

    # ── Step 5: Confronto stabilità N=5 vs N=10 (Punto 5) ───────────────────
    stds5  = np.nanstd(ws5,  axis=0)
    stds10 = np.nanstd(ws10, axis=0)
    plot_stability_comparison(stds5, stds10, TICKERS_5, TICKERS_10,
                              "bootstrap_stability_comparison_C5.png")
    plot_sharpe_comparison(srs5, srs10, sr5_stor, sr10_stor,
                           "bootstrap_sharpe_comparison_C5.png")

    # ── Commento riassuntivo (Punto 5) ───────────────────────────────────────
    print("\n" + "="*65)
    print("COMMENTO – Stabilità e ruolo del rapporto N/T")
    print("="*65)
    print(f"N=5:  N/T = {5}/{T_BOOT} = {5/T_BOOT:.3f}  →  σ(SR) = {np.nanstd(srs5):.3f}")
    print(f"N=10: N/T = {10}/{T_BOOT} = {10/T_BOOT:.3f} →  σ(SR) = {np.nanstd(srs10):.3f}")
    print()
    print("Al crescere di N (con T fisso):")
    print("  - La matrice Sigma ha più parametri da stimare → stime meno precise")
    print("  - L'ottimizzatore amplifica l'errore di stima (error maximization)")
    print("  - I pesi del portafoglio di tangenza diventano più variabili")
    print("  - La distribuzione dello Sharpe Ratio si allarga")
    print()
    print("Regola pratica: mantenere N/T < 0.1–0.2 per stime stabili.")
    print("Con N = T la matrice Sigma non è invertibile (rango deficiente).")
    print("\nParte C completata. Grafici salvati in parte_C/")
