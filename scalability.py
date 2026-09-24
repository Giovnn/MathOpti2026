"""
scalability.py - Analisi di scalabilita' di Model I e Model II.

Risolve con entrambi i modelli (obiettivo del paper: costo di routing f_c) istanze
Cordeau via via piu' grandi e confronta costi e tempi con le Tabelle 5 e 6 di Gaul,
Klamroth & Stiglmayr (2022).

La scala di istanze: tutte con lo stesso orizzonte di servizio (T = 720 minuti); a ogni
gradino 12 utenti e un veicolo in piu'.
    serie a: capacita' 3, un posto per utente
    serie b: capacita' 6, da 1 a 6 posti per utente

Due modalita':
    cosi' com'e'                              fino a 84 utenti, pochi minuti
    togliendo il '#' davanti alle righe       fino a 96 utenti, circa 40 minuti
    di a8-96 e b8-96 in ISTANZE               (nel paper Model I impiega un'ora su a8-96,
                                               Model II otto minuti)

I tempi non sono confrontabili in assoluto con quelli del paper (CPLEX 12.10 su un altro
computer): si confrontano l'andamento al crescere delle istanze e quale modello e' piu'
veloce. Sotto il secondo le differenze fra i due modelli sono rumore.

Esecuzione:  python scalability.py
Produce:     la tabella a schermo, risultati/scalability.csv, risultati/scalability.png
"""

import csv
import sys
import time
from pathlib import Path

from instances import leggi_istanza
from graph import Grafo
from model import costruisci_modello
from results import risolvi


# ----------------------------------------------------------------------
# Parametri
# ----------------------------------------------------------------------

CARTELLA = Path(__file__).resolve().parent
CARTELLA_DATI = CARTELLA / "dati_milp"
CARTELLA_RISULTATI = CARTELLA / "risultati"
FILE_CSV = CARTELLA_RISULTATI / "scalability.csv"
FILE_GRAFICO = CARTELLA_RISULTATI / "scalability.png"

LIMITE_TEMPO = 7200   # secondi per esecuzione (30 minuti)
MIP_GAP = 1e-4        # tolleranza standard dei solver, la stessa di default di CPLEX
                      # (model.py, se non indicato, chiede l'ottimo esatto)

# Dalle Tabelle 5 e 6 del paper:
#   istanza: (costo f_c, tempo Model I in secondi, tempo Model II in secondi)
ISTANZE = {
    "a2-24": (431.1, 0.06, 0.05),
    "a3-36": (583.2, 0.09, 0.11),
    "a4-48": (668.8, 0.67, 0.50),
    "a5-60": (808.4, 1.41, 1.03),
    "a6-72": (916.1, 17.88, 13.87),
    "a7-84": (1033.3, 35.70, 5.89),
    "a8-96": (1229.7, 3593.0, 461.0),     # run completo: togliere il '#'
    "b2-24": (444.7, 0.06, 0.05),
    "b3-36": (603.8, 0.07, 0.09),
    "b4-48": (673.8, 1.14, 0.94),
    "b5-60": (902.0, 4.01, 0.92),
    "b6-72": (978.5, 3.48, 5.11),
    "b7-84": (1203.4, 2.93, 2.72),
    "b8-96": (1185.6, 27.42, 24.26),      # run completo: togliere il '#'
}

# Colonne della tabella, uguali a schermo e nel CSV.
COLONNE = ["istanza", "serie", "n", "archi", "modello", "esito", "costo", "gap_perc",
           "tempo_s", "costo_paper", "tempo_paper_s"]


# ----------------------------------------------------------------------
# Una misura: un'istanza risolta con un modello
# ----------------------------------------------------------------------

def misura(grafo: Grafo, variante: str) -> dict:
    """Risolve il modello e restituisce esito, costo, gap (in %) e tempo del solver."""
    m = costruisci_modello(grafo, variante=variante, time_limit=LIMITE_TEMPO,
                           mip_gap=MIP_GAP, log=False)
    try:
        r = risolvi(m)
    finally:
        m.dispose()           # libera la memoria di Gurobi prima dell'istanza successiva

    # "N/A" come nel paper: nessuna soluzione trovata entro il limite di tempo.
    if r.stato == "time_limit":
        esito = "tempo scaduto" if r.ha_soluzione else "N/A"
    else:
        esito = r.stato       # "ottimo", "infeasible", "interrotto", ...

    return {
        "esito": esito,
        "costo": None if r.obj is None else round(r.obj, 4),
        "gap_perc": None if r.gap is None else round(100 * r.gap, 3),
        "tempo_s": round(r.tempo, 3),
    }


# ----------------------------------------------------------------------
# Tabella a schermo e CSV
# ----------------------------------------------------------------------

def testo(valore, formato: str) -> str:
    """Numero formattato, oppure '-' se manca."""
    return "-" if valore is None else format(valore, formato)


def al_limite(riga: dict) -> bool:
    """Il modello si e' fermato per il limite di tempo (con o senza soluzione)."""
    return riga["esito"] in ("tempo scaduto", "N/A")


def stampa_intestazione() -> None:
    print(f"{'istanza':<8} {'n':>3} {'archi':>6} {'modello':<7} {'esito':<14} "
          f"{'costo':>10} {'gap %':>7} {'tempo s':>9} | "
          f"{'paper: costo':>12} {'tempo s':>8}")


def stampa_riga(riga: dict) -> None:
    print(f"{riga['istanza']:<8} {riga['n']:>3} {riga['archi']:>6} {riga['modello']:<7} "
          f"{riga['esito']:<14} {testo(riga['costo'], '.4f'):>10} "
          f"{testo(riga['gap_perc'], '.3f'):>7} {testo(riga['tempo_s'], '.2f'):>9} | "
          f"{riga['costo_paper']:12.1f} {riga['tempo_paper_s']:8.2f}", flush=True)


def crea_csv() -> None:
    CARTELLA_RISULTATI.mkdir(exist_ok=True)
    with open(FILE_CSV, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLONNE).writeheader()


def aggiungi_al_csv(riga: dict) -> None:
    """Una riga alla volta: se il run si interrompe, le righe gia' scritte restano."""
    with open(FILE_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLONNE).writerow(riga)


# ----------------------------------------------------------------------
# Ciclo principale
# ----------------------------------------------------------------------

def esegui(righe: list[dict]) -> None:
    """Risolve tutte le istanze con i due modelli, in ordine di dimensione crescente."""
    for nome, (costo_paper, tempo_I, tempo_II) in ISTANZE.items():
        grafo = Grafo.costruisci(leggi_istanza(CARTELLA_DATI / f"{nome}.txt"))

        for variante, tempo_paper in (("I", tempo_I), ("II", tempo_II)):
            riga = {"istanza": nome, "serie": nome[0], "n": grafo.istanza.n,
                    "archi": len(grafo.archi), "modello": variante,
                    "esito": None, "costo": None, "gap_perc": None, "tempo_s": None,
                    "costo_paper": costo_paper, "tempo_paper_s": tempo_paper}
            try:
                riga.update(misura(grafo, variante))
            except Exception as errore:            # un errore non ferma tutto il run
                riga["esito"] = f"errore: {errore}"

            righe.append(riga)
            stampa_riga(riga)
            aggiungi_al_csv(riga)

            # Ctrl+C durante la risoluzione: Gurobi si ferma e restituisce "interrotto".
            if riga["esito"].startswith("interrotto"):
                raise KeyboardInterrupt


# ----------------------------------------------------------------------
# Riepilogo: Model I contro Model II
# ----------------------------------------------------------------------

def chi_vince(t_I: float | None, t_II: float | None) -> str:
    """Il modello piu' veloce e di quante volte (tempi minimi a 0.01 s)."""
    if t_I is None or t_II is None:
        return "-"
    t_I, t_II = max(t_I, 0.01), max(t_II, 0.01)
    if t_II < t_I:
        return f"II (x{t_I / t_II:.1f})"
    if t_I < t_II:
        return f"I (x{t_II / t_I:.1f})"
    return "pari"


def stampa_riepilogo(righe: list[dict]) -> None:
    print("\nModel I contro Model II: il piu' veloce e di quante volte")
    print(f"{'istanza':<8} {'noi':>20} {'paper':>20}")
    nota = False
    for nome, (_, tempo_I, tempo_II) in ISTANZE.items():
        nostri = {r["modello"]: r for r in righe if r["istanza"] == nome}
        if set(nostri) != {"I", "II"}:
            continue                                 # istanza non completata
        quanti_al_limite = sum(al_limite(r) for r in nostri.values())
        if quanti_al_limite == 2:
            noi = "entrambi al limite"
        else:
            noi = chi_vince(nostri["I"]["tempo_s"], nostri["II"]["tempo_s"])
            if quanti_al_limite == 1:
                noi += " *"
                nota = True
        print(f"{nome:<8} {noi:>20} {chi_vince(tempo_I, tempo_II):>20}")
    if nota:
        print("* un modello ha raggiunto il limite di tempo: il rapporto e' un minimo")


# ----------------------------------------------------------------------
# Grafico: tempo in funzione del numero di utenti
# ----------------------------------------------------------------------

def disegna_grafico(righe: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")                        # nessuna finestra: salva solo il file
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non installato: grafico non creato (il CSV c'e').")
        return

    colori = {"I": "tab:red", "II": "tab:blue"}
    fig, pannelli = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for pannello, serie, titolo in zip(pannelli, ("a", "b"),
                                       ("Serie a (capacita' 3)", "Serie b (capacita' 6)")):
        for variante, colore in colori.items():
            punti = [r for r in righe if r["serie"] == serie and r["modello"] == variante
                     and r["tempo_s"] is not None]
            if not punti:
                continue
            n = [r["n"] for r in punti]
            pannello.plot(n, [max(r["tempo_s"], 1e-3) for r in punti], "o-",
                          color=colore, label=f"Model {variante} (noi)")
            pannello.plot(n, [r["tempo_paper_s"] for r in punti], "s--",
                          color=colore, alpha=0.5, label=f"Model {variante} (paper)")
            pannello.set_xticks(n)

        if any(r["serie"] == serie and al_limite(r) for r in righe):
            pannello.axhline(LIMITE_TEMPO, color="grey", linestyle=":",
                             label="limite di tempo")

        pannello.set_yscale("log")
        pannello.set_title(titolo)
        pannello.set_xlabel("numero di utenti n")
        pannello.grid(True, which="both", alpha=0.3)
        if pannello.lines:                           # serie vuota: niente legenda
            pannello.legend(fontsize=8)

    pannelli[0].set_ylabel("tempo del solver [s] (scala log)")
    fig.suptitle("Scalabilita' di Model I e Model II (obiettivo f_c)")
    fig.tight_layout()
    fig.savefig(FILE_GRAFICO, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Esecuzione
# ----------------------------------------------------------------------

def main() -> int:
    mancanti = [n for n in ISTANZE if not (CARTELLA_DATI / f"{n}.txt").exists()]
    if mancanti:
        print(f"Mancano questi file in {CARTELLA_DATI}: {', '.join(mancanti)}")
        return 1

    print(f"Scalabilita' di Model I e Model II: {len(ISTANZE)} istanze, "
          f"limite di tempo {LIMITE_TEMPO} s per esecuzione\n")
    crea_csv()
    stampa_intestazione()

    righe: list[dict] = []
    inizio = time.perf_counter()
    try:
        esegui(righe)
    except KeyboardInterrupt:
        print("\nInterrotto: riepilogo e grafico con le istanze gia' completate.")

    stampa_riepilogo(righe)
    disegna_grafico(righe)
    print(f"\nTempo totale: {time.perf_counter() - inizio:.1f} s")
    print(f"Risultati salvati in: {FILE_CSV}")
    if FILE_GRAFICO.exists():
        print(f"                      {FILE_GRAFICO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
