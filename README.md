# Caso-pratico-EMFI-1
fulvio.corsi@unipi.it 

Con il titolo “Caso pratico EMFI 1” Si deve inviare: foglio excel o codice matlab con le serie dei rendimenti e calcoli; documento word o PDF con le risposte alle domande. Questo file deve essere un report leggibile senza fare riferimento al foglio excel o codice matlab.


Parte A – Frontiera efficiente (punti 1-7): Si scaricano i rendimenti mensili total return dividend-adjusted di 10 azioni per 10 anni (120 osservazioni), scegliendo titoli diversificati per settore. Si costruisce la frontiera efficiente media-deviazione standard su un sottoinsieme di 5 titoli sia analiticamente che tramite ottimizzazione numerica, prima senza vincoli (short selling ammesso) poi con non-negatività dei pesi. Si calcolano e si confrontano il portafoglio di minima varianza globale (GMV), i portafogli equally weighted (EW5, EW10) e una proxy del portafoglio di mercato rispetto alla frontiera. Infine, includendo un tasso risk-free (BTP a breve), si identifica il portafoglio di tangenza e il portafoglio ottimo che massimizza l'utilità media-varianza per un dato coefficiente di avversione al rischio A.

Parte B – CAPM e Single Index Model (punti 8-9): Si stimano i beta dei 10 titoli rispetto al portafoglio di mercato, verificandone la significatività statistica (incluso l'alpha). Si ricostruisce la matrice varianza-covarianza tramite il Single Index Model/CAPM e si confronta la frontiera efficiente risultante con quella empirica della Parte A.

Parte C – Stabilità e ruolo della dimensionalità (punti 1-5): Tramite bootstrap i.i.d. (500 campioni di T=120 mesi con reinserimento), si analizza la stabilità del portafoglio di tangenza al variare della dimensione N del problema (N=5 vs N=10), studiando come la variabilità dei pesi ottimali e la distribuzione dello Sharpe ratio cambiano al crescere del rapporto N/T.