# Questa parte implementa i Punti 8 e 9 della traccia:
#
#   Punto 8 – Stima OLS dei beta e alpha:
#   Punto 9 – Single Index Model e CAPM:
#
# LIBRERIE
# ---------
#   yfinance   
#   pandas     
#   numpy      
#   scipy      
#   matplotlib 
#   seaborn    
# =============================================================================

import matplotlib
matplotlib.use("Agg")  
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize


# ── Configurazione (identica alla Parte A)

TICKERS = {
    "Tech":       ["AAPL", "MSFT", "NVDA"],
    "Healthcare": ["JNJ",  "PFE",  "MRK"],
    "Energy":     ["XOM",  "CVX",  "BP"],
    "Index":      ["^GSPC"],
}
ALL_TICKERS = [t for group in TICKERS.values() for t in group]

SELECTED = ["NVDA", "MSFT", "JNJ", "MRK", "XOM"]

rf_annual = 0.02        # tasso risk-free: BOT 3 mesi 2015-2025
ann       = 12          # fattore di annualizzazione
rf        = rf_annual / ann   # tasso mensile

START = "2015-01-01"
END   = "2025-01-01"


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 1 – download_and_returns
# ══════════════════════════════════════════════════════════════════════════════
def download_and_returns(tickers, start, end):
    """
    Scarica i prezzi mensili adjusted da Yahoo Finance e calcola i rendimenti
    logaritmici mensili. Identica alla Parte A per garantire coerenza dei dati.

    Ritorna
    -------
    returns : pd.DataFrame
        Rendimenti logaritmici mensili per tutti i titoli.
        Colonne: ticker names (^GSPC → SP500).
    """
    raw     = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
    prices  = raw["Close"].resample("ME").last()
    prices.columns = [c if c != "^GSPC" else "SP500" for c in prices.columns]
    prices.dropna(how="all", inplace=True)

    # Rendimenti logaritmici: r_t = ln(P_t / P_{t-1})
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 2 – ols_stats
# ══════════════════════════════════════════════════════════════════════════════
def ols_stats(y_series, x_series):
    """
    Stima OLS del modello: y_t = alpha + beta * x_t + eps_t

    Calcola stime, errori standard, t-statistiche e p-value sia per il
    coefficiente beta (rischio sistematico) sia per l'intercetta alpha
    (alpha di Jensen).

    Parametri
    ---------
    y_series : array-like – rendimenti dell'asset i
    x_series : array-like – rendimenti del mercato (proxy SP500)

    Ritorna
    -------
    dict con:
        alpha, beta         : stime OLS
        se_a, se_b          : errori standard
        t_a, t_b            : t-statistiche
        p_a, p_b            : p-value (due code, dist. t con T-2 dof)
        R2                  : coefficiente di determinazione
        eps                 : residui OLS (= componente idiosincratica)
        s2                  : varianza residua campionaria (diviso T-2)

    Formule
    -------
    beta  = Cov(y,x) / Var(x)                             [stima OLS]
    alpha = y_bar - beta * x_bar
    s2    = RSS / (T-2)                                    [varianza residua]
    SE(beta)  = sqrt(s2 / Sxx)   con Sxx = sum((x-x_bar)^2)
    SE(alpha) = sqrt(s2 * (1/T + x_bar^2/Sxx))
    t         = stima / SE(stima)   ~  t(T-2)
    R2        = 1 - RSS/TSS
    """
    y, x = np.array(y_series, dtype=float), np.array(x_series, dtype=float)
    T    = len(y)

    # ── Stime OLS ──────────────────────────────────────────────────────────────
    beta  = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha = y.mean() - beta * x.mean()

    # ── Residui e varianza residua ─────────────────────────────────────────────
    eps = y - alpha - beta * x     # componente idiosincratica dell'asset i
    s2  = (eps @ eps) / (T - 2)   # ddof=2: due parametri stimati (alpha, beta)

    # ── Errori standard (formula OLS classica) ─────────────────────────────────
    Sxx   = ((x - x.mean())**2).sum()
    se_b  = np.sqrt(s2 / Sxx)
    se_a  = np.sqrt(s2 * (1 / T + x.mean()**2 / Sxx))

    # ── t-stat e p-value bilaterale con T-2 gradi di libertà ──────────────────
    t_b, t_a = beta / se_b, alpha / se_a
    p_b = 2 * stats.t.sf(abs(t_b), df=T - 2)
    p_a = 2 * stats.t.sf(abs(t_a), df=T - 2)

    # ── R² ────────────────────────────────────────────────────────────────────
    TSS = ((y - y.mean())**2).sum()
    R2  = 1 - (eps @ eps) / TSS

    return dict(alpha=alpha, beta=beta,
                se_a=se_a,   se_b=se_b,
                t_a=t_a,     t_b=t_b,
                p_a=p_a,     p_b=p_b,
                R2=R2, eps=eps, s2=s2)


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 3 – build_sigma_sim
# ══════════════════════════════════════════════════════════════════════════════
def build_sigma_sim(betas, s2eps, sigma2_m):
    """
    Costruisce la matrice di varianza-covarianza secondo il Single Index Model.

    Formula:
        Sigma_SIM = beta * beta' * sigma2_m  +  diag(sigma2_eps)

    Dove:
        beta * beta' * sigma2_m  →  matrice N×N delle covarianze sistematiche
                                    (tutta la correlazione passa dal mercato)
        diag(sigma2_eps)         →  varianze idiosincratiche sulla diagonale

    Il SIM riduce drasticamente i parametri da stimare:
        empirico: N(N+1)/2 varianze e covarianze
        SIM:      N beta + N sigma2_eps + 1 sigma2_m = 2N+1 parametri

    Parametri
    ---------
    betas    : (N,) array di beta OLS
    s2eps    : (N,) array di varianze residue OLS (= sigma2_eps_i)
    sigma2_m : float – varianza mensile del mercato

    Ritorna
    -------
    Sigma_SIM : (N,N) ndarray
    """
    return np.outer(betas, betas) * sigma2_m + np.diag(s2eps)


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 4 – markowitz_frontier
# ══════════════════════════════════════════════════════════════════════════════
def markowitz_frontier(mu_vec, Sig_mat, n_points=300):
    """
    Calcola la frontiera efficiente analitica di Markowitz dato mu e Sigma.

    Utilizza le stesse formule della Parte A (scalari A, B, C, D).

    Ritorna
    -------
    sig_ann : (n_points,) – std dev annualizzata sulla frontiera (%)
    mu_ann  : (n_points,) – rendimento atteso annualizzato (%)
    mu_gmv  : float – rendimento GMV mensile
    """
    n    = len(mu_vec)
    iota = np.ones(n)
    Sinv = np.linalg.inv(Sig_mat)

    # Scalari di Markowitz
    A_mk = float(iota @ Sinv @ iota)
    B_mk = float(iota @ Sinv @ mu_vec)
    C_mk = float(mu_vec @ Sinv @ mu_vec)
    D_mk = A_mk * C_mk - B_mk**2

    mu_gmv  = B_mk / A_mk
    mu_grid = np.linspace(mu_gmv, mu_vec.max() * 1.6, n_points)

    # Varianza sulla frontiera: sigma^2(mu_p) = (A*mu_p^2 - 2B*mu_p + C) / D
    sig_grid = np.sqrt((A_mk * mu_grid**2 - 2*B_mk*mu_grid + C_mk) / D_mk)

    return sig_grid * np.sqrt(ann) * 100, mu_grid * ann * 100, mu_gmv


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 5 – plot_beta_alpha
# ══════════════════════════════════════════════════════════════════════════════
def plot_beta_alpha(risky, ols):
    """
    Bar chart affiancati: beta (rischio sistematico) e alpha annualizzato.

    Legenda colori:
        Beta:  rosso  → beta > 1 (asset aggressivo, amplifica il mercato)
               blu    → beta < 1 (asset difensivo, attutisce il mercato)
        Alpha: verde  → alpha > 0 (outperformance rispetto al CAPM)
               salmone → alpha < 0 (underperformance)
    """
    betas_all = [ols[t]["beta"]          for t in risky]
    alphas_ann= [ols[t]["alpha"]*ann*100 for t in risky]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Beta
    ax = axes[0]
    cols = ["tomato" if b > 1 else "steelblue" for b in betas_all]
    bars = ax.bar(risky, betas_all, color=cols, edgecolor="black", lw=0.5)
    ax.axhline(1.0, color="black", lw=1.5, ls="--")
    for bar, b in zip(bars, betas_all):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Beta dei 10 titoli vs S&P 500"); ax.set_ylabel("β")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(handles=[mpatches.Patch(color="tomato",    label="β>1 aggressivo"),
                       mpatches.Patch(color="steelblue", label="β<1 difensivo")], fontsize=9)

    # Alpha di Jensen annualizzato
    ax = axes[1]
    cols_a = ["limegreen" if a > 0 else "salmon" for a in alphas_ann]
    bars_a = ax.bar(risky, alphas_ann, color=cols_a, edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=1)
    for bar, a in zip(bars_a, alphas_ann):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.3 if a >= 0 else -1.8),
                f"{a:.1f}%", ha="center", fontsize=8)
    ax.set_title("Alpha di Jensen annualizzato (%)"); ax.set_ylabel("α (%/anno)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(handles=[mpatches.Patch(color="limegreen", label="α>0 outperform"),
                       mpatches.Patch(color="salmon",    label="α<0 underperform")], fontsize=9)

    fig.suptitle("Beta e Alpha di Jensen – OLS (2015–2025)", fontsize=13)
    fig.tight_layout()
    fig.savefig("beta_alpha_B8.png", dpi=150)
    plt.close(fig)
    print("Salvato: beta_alpha_B8.png")


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 6 – plot_sml
# ══════════════════════════════════════════════════════════════════════════════
def plot_sml(risky, ols, returns, mu_m, rf, ann, palette, sector_map):
    """
    Disegna la Security Market Line (SML) e posiziona i 10 titoli.

    SML: E[r_i] = rf + beta_i * (mu_m - rf)  →  retta in spazio (beta, mu)

    Differenza SML vs CML:
        CML: asse x = sigma (std dev), valida solo per portafogli efficienti
        SML: asse x = beta,            valida per qualsiasi asset
    """
    betas_all  = np.array([ols[t]["beta"]  for t in risky])
    mu_emp_all = np.array([returns[t].mean() for t in risky])
    mu_capm_all = rf + betas_all * (mu_m - rf)

    beta_grid = np.linspace(min(betas_all) - 0.1, max(betas_all) + 0.1, 200)
    mu_sml    = rf + beta_grid * (mu_m - rf)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(beta_grid, mu_sml * ann * 100, "b-", lw=2,
            label=f"SML: rf={rf*ann*100:.1f}%, μ_m={mu_m*ann*100:.1f}%")

    for i, t in enumerate(risky):
        color = palette[sector_map[t]]
        ax.scatter(betas_all[i], mu_emp_all[i] * ann * 100,
                   color=color, s=120, zorder=5, edgecolors="black", lw=0.5)
        ax.annotate(t, (betas_all[i], mu_emp_all[i] * ann * 100),
                    textcoords="offset points", xytext=(6, 3), fontsize=9)
        # Freccia alpha: differenza tra rendimento reale e CAPM
        if abs((mu_emp_all[i] - mu_capm_all[i]) * ann * 100) > 1.5:
            ax.annotate("", xy=(betas_all[i], mu_emp_all[i] * ann * 100),
                        xytext=(betas_all[i], mu_capm_all[i] * ann * 100),
                        arrowprops=dict(arrowstyle="->", color="grey", lw=1.2))

    ax.scatter(0, rf * ann * 100, color="grey", s=100, marker="o", zorder=5)
    ax.set_xlabel("Beta (β)"); ax.set_ylabel("Rendimento atteso annuo (%)")
    ax.set_title("Security Market Line – CAPM vs Rendimenti Reali", fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("SML_B8.png", dpi=150)
    plt.close(fig)
    print("Salvato: SML_B8.png")


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 7 – plot_correlation_comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation_comparison(Corr_SIM, Corr_emp, risky):
    """
    Heatmap affiancate: correlazione SIM vs correlazione empirica.

    Il SIM impone che tutta la correlazione tra i titoli passi dal mercato:
        rho_SIM(i,j) = (beta_i * beta_j * sigma2_m) / (sigma_i * sigma_j)

    Se due titoli hanno correlazioni empiriche superiori a quelle del SIM,
    vuol dire che hanno fattori comuni non catturati dal mercato.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, corr, title in zip(axes,
                                [Corr_SIM, Corr_emp],
                                ["Correlazione SIM", "Correlazione Empirica"]):
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, square=True,
                    linewidths=0.5, ax=ax, annot_kws={"size": 8})
        ax.set_title(title, fontsize=12)
    fig.suptitle("Confronto matrici di correlazione: SIM vs Empirica", fontsize=13)
    fig.tight_layout()
    fig.savefig("correlation_SIM_vs_empirical_B9.png", dpi=150)
    plt.close(fig)
    print("Salvato: correlation_SIM_vs_empirical_B9.png")


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 8 – plot_frontier_comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_frontier_comparison(mu5_emp, Sig5_emp, mu5_capm, Sig5_sim, selected, returns):
    """
    Confronto grafico tra frontiera efficiente empirica (Parte A) e
    frontiera SIM/CAPM (Parte B) per i 5 titoli selezionati.

    Le due frontiere differiscono perché:
      1. μ_CAPM ≠ μ_empirico  (il CAPM azzera gli alpha, i rendimenti storici no)
      2. Σ_SIM ≠ Σ_empirica   (il SIM semplifica la struttura delle covarianze)
    """
    sig_emp_f, mu_emp_f, _ = markowitz_frontier(mu5_emp,  Sig5_emp)
    sig_sim_f, mu_sim_f, _ = markowitz_frontier(mu5_capm, Sig5_sim)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(sig_emp_f, mu_emp_f, "b-",  lw=2.5,
            label="Frontiera empirica (μ empirico, Σ empirica) – Parte A")
    ax.plot(sig_sim_f, mu_sim_f, "r--", lw=2.5,
            label="Frontiera SIM/CAPM (μ_CAPM, Σ_SIM) – Parte B")

    for t in selected:
        sig_a = returns[t].std()  * np.sqrt(ann) * 100
        mu_a  = returns[t].mean() * ann * 100
        ax.scatter(sig_a, mu_a, s=100, marker="o", color="steelblue",
                   zorder=5, edgecolors="black", lw=0.5)
        ax.annotate(t, (sig_a, mu_a), textcoords="offset points",
                    xytext=(5, 3), fontsize=9)

    ax.set_xlabel("Std Dev Annua (%)"); ax.set_ylabel("Rendimento Atteso Annuo (%)")
    ax.set_title("Frontiera Efficiente: Empirica vs SIM/CAPM\n"
                 "5 titoli selezionati – short selling ammesso", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("frontier_SIM_vs_empirical_B9.png", dpi=150)
    plt.close(fig)
    print("Salvato: frontier_SIM_vs_empirical_B9.png")


# ══════════════════════════════════════════════════════════════════════════════
# FUNZIONE 9 – print_ols_table
# ══════════════════════════════════════════════════════════════════════════════
def print_ols_table(risky, ols):
    """
    Stampa la tabella riassuntiva dei risultati OLS per tutti i titoli.
    Simboli di significatività: *** p<0.01, ** p<0.05, * p<0.10
    """
    def stars(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    rows = []
    for t in risky:
        o = ols[t]
        rows.append({
            "Ticker":        t,
            "Alpha ann (%)": round(o["alpha"] * ann * 100, 3),
            "t(α)":          round(o["t_a"], 3),
            "Sig α":         stars(o["p_a"]),
            "Beta":          round(o["beta"], 4),
            "t(β)":          round(o["t_b"], 3),
            "Sig β":         stars(o["p_b"]),
            "R²":            round(o["R2"], 4),
        })
    tab = pd.DataFrame(rows).set_index("Ticker")
    T_actual = len(ols[risky[0]]["eps"])   # ricavato dai residui OLS
    print("\nStima OLS: r_i = alpha + beta * r_m + eps")
    print(f"T = {T_actual} osservazioni mensili\n")
    print(tab.to_string())
    print("\nLegenda: *** p<0.01  ** p<0.05  * p<0.10")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Step 1: Download dati ────────────────────────────────────────────────
    print(f"Download: {ALL_TICKERS}  [{START} → {END}]")
    returns  = download_and_returns(ALL_TICKERS, START, END)
    r_m      = returns["SP500"]
    RISKY    = list(returns.columns)
    mu_m     = r_m.mean()
    sigma2_m = np.var(r_m, ddof=1)

    print(f"Osservazioni: {len(returns)} mesi\n")

    # ── Step 2: Stima OLS dei beta (Punto 8) ─────────────────────────────────
    # r_i = alpha_i + beta_i * r_m + eps_i
    ols = {t: ols_stats(returns[t], r_m) for t in RISKY}
    print_ols_table(RISKY, ols)

    # Palette e mappa settore→colore per i grafici
    palette = {"Tech": "steelblue", "Healthcare": "seagreen",
               "Energy": "tomato",  "Index": "black"}
    sector_map = {}
    for sector, tickers in TICKERS.items():
        for t in tickers:
            sector_map["SP500" if t == "^GSPC" else t] = sector

    # ── Step 3: Bar chart beta e alpha ───────────────────────────────────────
    plot_beta_alpha(RISKY, ols)

    # ── Step 4: Security Market Line (Punto 8) ───────────────────────────────
    plot_sml(RISKY, ols, returns, mu_m, rf, ann, palette, sector_map)

    # ── Step 5: Costruzione Sigma_SIM (Punto 9) ──────────────────────────────
    # Sigma_SIM = beta * beta' * sigma2_m + diag(sigma2_eps)
    betas_arr = np.array([ols[t]["beta"] for t in RISKY])
    s2eps_arr = np.array([ols[t]["s2"]   for t in RISKY])
    Sigma_SIM = build_sigma_sim(betas_arr, s2eps_arr, sigma2_m)

    # Matrice di correlazione SIM
    D_inv    = np.diag(1 / np.sqrt(np.diag(Sigma_SIM)))
    Corr_SIM = pd.DataFrame(D_inv @ Sigma_SIM @ D_inv, index=RISKY, columns=RISKY)
    Corr_emp = returns.corr()
    plot_correlation_comparison(Corr_SIM, Corr_emp, RISKY)

    # ── Step 6: Rendimenti CAPM, std dev SIM e confronto frontiere (Punto 9) ─
    # CAPM: E[r_i] = rf + beta_i * (mu_m - rf)
    mu_capm_arr = rf + betas_arr * (mu_m - rf)

    # Deviazioni standard SIM (annualizzate): sigma_i = sqrt(Sigma_SIM[i,i]) * sqrt(12)
    # Per costruzione del SIM: Var(r_i) = beta_i^2*Var(r_m) + Var(eps_i)
    # → diagonale di Sigma_SIM ≈ varianza empirica → sigma_SIM ≈ sigma_emp
    sigma_sim_ann = np.sqrt(np.diag(Sigma_SIM)) * np.sqrt(ann) * 100
    sigma_emp_ann = np.array([returns[t].std() for t in RISKY]) * np.sqrt(ann) * 100
    mu_emp_arr    = np.array([returns[t].mean() for t in RISKY])

    # Tabella riepilogativa (Punto 9): mu CAPM, sigma SIM, alpha
    print("\nPunto 9 – Statistiche SIM/CAPM vs Empiriche (annualizzate):")
    tab_sim = pd.DataFrame({
        "Beta":                 betas_arr.round(4),
        "μ CAPM (%)":          (mu_capm_arr * ann * 100).round(2),
        "μ Empirico (%)":      (mu_emp_arr  * ann * 100).round(2),
        "σ SIM (%)":           sigma_sim_ann.round(2),
        "σ Empirica (%)":      sigma_emp_ann.round(2),
        "Alpha Jensen (%)":    ((mu_emp_arr - mu_capm_arr) * ann * 100).round(2),
    }, index=RISKY)
    print(tab_sim.to_string())
    print("\nNota: σ SIM ≈ σ empirica per asset singoli (diagonale preservata dal SIM).")
    print("      Le differenze sulle frontiere emergono dai termini off-diagonali (covarianze).")

    # Estrai parametri per i 5 titoli selezionati
    sel_idx  = [RISKY.index(t) for t in SELECTED]
    mu5_emp  = np.array([returns[t].mean() for t in SELECTED])
    mu5_capm = mu_capm_arr[sel_idx]
    Sig5_emp = returns[SELECTED].cov().values
    Sig5_SIM = Sigma_SIM[np.ix_(sel_idx, sel_idx)]

    # Confronto frontiere
    plot_frontier_comparison(mu5_emp, Sig5_emp, mu5_capm, Sig5_SIM, SELECTED, returns)

    print("\nParte B completata. Grafici salvati in parte_B/")
