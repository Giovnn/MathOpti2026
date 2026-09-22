"""
test.py - Esecuzione dei modelli su istanze piccole.

Linee guida d'esame: "un file test.py in cui tutti gli algoritmi proposti vengono
eseguiti su istanze di piccole dimensioni".

Parte 1  Model I e Model II con l'obiettivo del paper, il costo di routing f_c, su sei
         istanze Cordeau piccole. Il costo trovato si confronta con le Tabelle 5 e 6 di
         Gaul, Klamroth & Stiglmayr (2022).
Parte 2  Le altre sei funzioni obiettivo del paper (Sez. 3.5) su a2-16, con entrambi i
         modelli. Il paper non riporta valori per queste istanze: si controlla che i due
         modelli, equivalenti per costruzione, trovino lo stesso ottimo.

Esecuzione:  python test.py        (pochi secondi)
Le istanze vanno nella cartella dati_milp, accanto a questo file.
"""

import sys
import time
from pathlib import Path

from instances import leggi_istanza
from graph import Grafo
from model import costruisci_modello, VARIANTI
from objectives import costruisci_con_obiettivo
from results import risolvi, controlli


# ----------------------------------------------------------------------
# Parametri
# ----------------------------------------------------------------------

CARTELLA_DATI = Path(__file__).resolve().parent / "dati_milp"

LIMITE_TEMPO = 60          # secondi per esecuzione: solo una rete di sicurezza
TOLLERANZA_PAPER = 0.1     # il paper riporta i costi con un solo decimale
TOLLERANZA_MODELLI = 1e-3  # Model I e Model II devono trovare lo stesso ottimo

# Costo di routing f_c dalle Tabelle 5 (serie a) e 6 (serie b) del paper.
COSTI_PAPER = {
    "a2-16": 294.3,
    "a3-18": 300.5,
    "a4-16": 282.7,
    "b2-16": 309.4,
    "b3-18": 301.6,
    "b4-16": 297.0,
}

# Parte 2: le altre sei funzioni obiettivo, con i pesi del paper
# (alfa = 1, beta = n/5, gamma = 20).
#   fn     richieste rifiutate          fcr     costo + regret totale
#   fr     regret totale                fcrmax  costo + regret massimo
#   frmax  regret massimo               frcr    costo + regret + rifiuti
ISTANZA_OBIETTIVI = "a2-16"
ALTRI_OBIETTIVI = ("fn", "fr", "frmax", "fcr", "fcrmax", "frcr")


# ----------------------------------------------------------------------
# Funzioni di supporto
# ----------------------------------------------------------------------

# Ogni controllo eseguito finisce qui come (descrizione, superato).
esiti: list[tuple[str, bool]] = []


def verifica(superato: bool, descrizione: str) -> bool:
    """Registra l'esito di un controllo e lo restituisce."""
    esiti.append((descrizione, superato))
    return superato


def carica_grafo(nome: str) -> Grafo:
    """Legge l'istanza e costruisce il suo grafo event-based."""
    return Grafo.costruisci(leggi_istanza(CARTELLA_DATI / f"{nome}.txt"))


def ottimo_e_coerente(r, descrizione: str) -> bool:
    """Il solver ha dimostrato l'ottimo e la soluzione supera i controlli di results.py."""
    problemi = controlli(r)
    for problema in problemi:
        print(f"      {problema}")
    return verifica(r.ottimo and not problemi,
                    f"{descrizione}: risolto all'ottimo e soluzione coerente")


def stesso_valore(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) <= TOLLERANZA_MODELLI


def testo(valore: float | None, formato: str) -> str:
    """Numero formattato, oppure '-' se manca (nessuna soluzione trovata)."""
    return "-" if valore is None else format(valore, formato)


# ----------------------------------------------------------------------
# Parte 1 - costo di routing, confronto con le Tabelle 5 e 6
# ----------------------------------------------------------------------

def parte_1() -> None:
    print("PARTE 1 - costo di routing f_c, confronto con le Tabelle 5 e 6 del paper")
    print(f"{'istanza':<8} {'modello':<8} {'costo':>10} {'paper':>7} {'diff':>7} "
          f"{'tempo':>8}  esito")

    for nome, costo_paper in COSTI_PAPER.items():
        grafo = carica_grafo(nome)          # un solo grafo per entrambi i modelli
        costi = {}

        for variante in VARIANTI:
            m = costruisci_modello(grafo, variante=variante,
                                   time_limit=LIMITE_TEMPO, log=False)
            r = risolvi(m)
            chi = f"{nome} Model {variante}"
            diff = r.obj - costo_paper if r.ha_soluzione else None

            ok_ottimo = ottimo_e_coerente(r, chi)
            ok_paper = verifica(diff is not None and abs(diff) <= TOLLERANZA_PAPER,
                                f"{chi}: costo entro {TOLLERANZA_PAPER} dal paper")
            costi[variante] = r.obj

            esito = "OK" if ok_ottimo and ok_paper else "ERRORE"
            print(f"{nome:<8} {variante:<8} {testo(r.obj, '.4f'):>10} {costo_paper:7.1f} "
                  f"{testo(diff, '+.3f'):>7} {r.tempo:7.2f}s  {esito}")

        if not verifica(stesso_valore(costi["I"], costi["II"]),
                        f"{nome}: Model I e Model II con lo stesso costo"):
            print(f"         ATTENZIONE: Model I e Model II trovano costi diversi su {nome}")


# ----------------------------------------------------------------------
# Parte 2 - le altre funzioni obiettivo, Model I contro Model II
# ----------------------------------------------------------------------

def parte_2() -> None:
    print(f"\nPARTE 2 - le altre funzioni obiettivo su {ISTANZA_OBIETTIVI} "
          f"(pesi del paper)")
    print(f"{'obiettivo':<10} {'Model I':>10} {'Model II':>10} {'servite':>8} "
          f"{'tempo I':>8} {'tempo II':>9}  esito")

    grafo = carica_grafo(ISTANZA_OBIETTIVI)

    for obiettivo in ALTRI_OBIETTIVI:
        risultati, tutti_ok = {}, True

        for variante in VARIANTI:
            m = costruisci_con_obiettivo(grafo, obiettivo, variante=variante,
                                         time_limit=LIMITE_TEMPO, log=False)
            r = risolvi(m)
            risultati[variante] = r
            tutti_ok = ottimo_e_coerente(r, f"{obiettivo} Model {variante}") and tutti_ok

        r1, r2 = risultati["I"], risultati["II"]
        tutti_ok = verifica(stesso_valore(r1.obj, r2.obj),
                            f"{obiettivo}: Model I e Model II con lo stesso ottimo") and tutti_ok

        servite = (f"{r2.n - round(r2.canonici['rifiuti'])}/{r2.n}"
                   if r2.ha_soluzione else "-")
        print(f"{obiettivo:<10} {testo(r1.obj, '.4f'):>10} {testo(r2.obj, '.4f'):>10} "
              f"{servite:>8} {r1.tempo:7.2f}s {r2.tempo:8.2f}s  "
              f"{'OK' if tutti_ok else 'ERRORE'}")


# ----------------------------------------------------------------------
# Esecuzione
# ----------------------------------------------------------------------

def main() -> int:
    richieste = sorted(set(COSTI_PAPER) | {ISTANZA_OBIETTIVI})
    mancanti = [n for n in richieste if not (CARTELLA_DATI / f"{n}.txt").exists()]
    if mancanti:
        print(f"Mancano questi file in {CARTELLA_DATI}: {', '.join(mancanti)}")
        return 1

    inizio = time.perf_counter()
    parte_1()
    parte_2()

    falliti = [descrizione for descrizione, superato in esiti if not superato]
    print(f"\nRIEPILOGO: {len(esiti) - len(falliti)}/{len(esiti)} controlli superati "
          f"in {time.perf_counter() - inizio:.1f} s")
    for descrizione in falliti:
        print(f"  FALLITO: {descrizione}")
    return 1 if falliti else 0


if __name__ == "__main__":
    sys.exit(main())
