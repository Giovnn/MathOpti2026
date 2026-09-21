"""
test.py — Test automatici del progetto.

Esecuzione:   python test.py        (nessuna libreria in piu' richiesta)
              pytest test.py        (se pytest e' installato: stessi test)

Ogni funzione il cui nome inizia con "test_" e' un test: passa se arriva in fondo,
fallisce alla prima assert falsa o alla prima eccezione. I valori di riferimento sono
valori ottimi, verificati con due solver indipendenti (Gurobi e HiGHS): non dipendono
dal solver. Valori NON unici (costo sotto fr, regret sotto fc, sequenze delle rotte)
non compaiono mai in un'assert.
"""

import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path

from instances import leggi_istanza
from graph import Grafo, schedule_minimo
from model import costruisci_modello, VARIANTI
from objectives import (OBIETTIVI, Pesi, coefficienti, costruisci_con_obiettivo,
                        imposta_somma_pesata)
from results import risolvi, controlli, risolvi_lessicografico

# Cartella delle istanze calcolata rispetto a QUESTO file: i test funzionano da
# qualunque cartella li si lanci (terminale, PyCharm, macchina del docente).
CARTELLA_DATI = Path(__file__).resolve().parent / "dati_milp"

TOL = 1e-3          # sui valori ottimi riportati con 4 decimali

# f_c, stesso valore per Model I e Model II. Coincidono con le Tabelle 5 e 6 del
# paper arrotondate a un decimale, con l'eccezione di a3-24 (tabella: 344.9),
# discussa in analisi_funzioni_obiettivo.md.
VALORI_FC = {
    "a2-16": 294.2480,
    "a2-20": 344.8341,
    "a3-24": 344.8336,
    "b2-16": 309.4057,
    "b2-20": 332.6382,
}

# Sette obiettivi con i pesi del paper (alpha = 1, beta = n/5, gamma = 20).
VALORI_OBIETTIVI = {
    "a2-16": {"fc": 294.2480, "fn": 0.0, "fr": 14.1864, "frmax": 5.5939,
              "fcr": 330.2431, "fcrmax": 329.9129, "frcr": 267.7970},
    "b2-16": {"fc": 309.4057, "fn": 0.0, "fr": 10.7556, "frmax": 5.4985,
              "fcr": 338.1290, "fcrmax": 343.8774, "frcr": 283.6275},
    "a3-24": {"fc": 344.8336, "fn": 0.0, "fr": 8.1950, "frmax": 4.9672,
              "fcr": 382.7216, "fcrmax": 390.7776, "frcr": 328.1593},
}


# ----------------------------------------------------------------------
# Funzioni di supporto
# ----------------------------------------------------------------------

@lru_cache(maxsize=None)
def grafo(nome: str) -> Grafo:
    """Grafo dell'istanza 'nome', costruito una volta sola e poi riusato."""
    return Grafo.costruisci(leggi_istanza(CARTELLA_DATI / f"{nome}.txt"))


def vicino(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def soluzione(nome: str, obiettivo: str = "fc", variante: str = "II",
              pesi: Pesi | None = None):
    """Risolve, e pretende ottimo dimostrato e controlli di coerenza superati."""
    m = costruisci_con_obiettivo(grafo(nome), obiettivo, variante=variante,
                                 pesi=pesi, log=False)
    r = risolvi(m)
    assert r.ottimo, f"{nome} {obiettivo} Model {variante}: stato {r.stato}"
    problemi = controlli(r)
    assert not problemi, f"{nome} {obiettivo} Model {variante}: {problemi}"
    return r


def solleva(eccezione, funzione, *args, **kwargs) -> None:
    """Verifica che funzione(*args, **kwargs) sollevi l'eccezione indicata."""
    try:
        funzione(*args, **kwargs)
    except eccezione:
        return
    raise AssertionError(f"{funzione.__name__} doveva sollevare {eccezione.__name__}")


# ----------------------------------------------------------------------
# Dati e grafo
# ----------------------------------------------------------------------

def test_lettura_istanza():
    ist = grafo("a2-16").istanza
    assert (ist.n, ist.K, ist.Q) == (16, 2, 3)
    assert (ist.L, ist.T) == (30.0, 480.0)


def test_dimensioni_grafo():
    g = grafo("a2-16")
    assert (len(g.nodi), len(g.archi)) == (63, 344)


def test_schedule_minimo():
    ist = grafo("a2-16").istanza
    p1, d1 = ist.pickup(1).id, ist.delivery(1).id
    p9, d9 = ist.pickup(9).id, ist.delivery(9).id

    assert schedule_minimo([p1, d1], [(0, 1)], ist) == [369.0, 402.0]
    s = schedule_minimo([p9, d9, p1, d1], [(0, 1), (2, 3)], ist)
    assert all(vicino(a, b, 1e-6) for a, b in zip(s, [276.0, 286.901486, 369.0, 402.0]))
    assert schedule_minimo([p1, d1, p9, d9], [(0, 1), (2, 3)], ist) is None
    assert schedule_minimo([p9, d9], [(0, 1)], ist, tempi=[15.0]) == [276.0, 294.0]
    # soglia del viaggio diretto: t <= L
    assert schedule_minimo([p1, d1], [(0, 1)], ist, tempi=[30.0]) == [369.0, 402.0]
    assert schedule_minimo([p1, d1], [(0, 1)], ist, tempi=[31.0]) is None


# ----------------------------------------------------------------------
# Modelli: Tabelle 5 e 6
# ----------------------------------------------------------------------

def test_tabelle_5_6_model_I_e_II():
    for nome, atteso in VALORI_FC.items():
        for variante in VARIANTI:
            r = risolvi(costruisci_modello(grafo(nome), variante=variante, log=False))
            assert r.ottimo, f"{nome} Model {variante}: stato {r.stato}"
            assert vicino(r.obj, atteso), f"{nome} Model {variante}: {r.obj} != {atteso}"
            assert not controlli(r), f"{nome} Model {variante}: {controlli(r)}"
            assert r.canonici["rifiuti"] == 0


# ----------------------------------------------------------------------
# Funzioni obiettivo (analisi_funzioni_obiettivo.md, par. 6)
# ----------------------------------------------------------------------

def test_T1_preset_fc_uguale_al_default():
    g = grafo("a2-16")
    m1 = costruisci_modello(g, log=False)
    m2 = costruisci_con_obiettivo(g, "fc", log=False)
    assert m1._coefficienti == m2._coefficienti
    assert vicino(risolvi(m1).obj, risolvi(m2).obj, 1e-9)


def test_T2_T3_T5_sette_obiettivi():
    # T2: Model I e II danno lo stesso ottimo; T3 e T5 sono dentro controlli().
    for nome, valori in VALORI_OBIETTIVI.items():
        for obiettivo in OBIETTIVI:
            for variante in VARIANTI:
                r = soluzione(nome, obiettivo, variante)
                assert vicino(r.obj, valori[obiettivo]), \
                    f"{nome} {obiettivo} Model {variante}: {r.obj} != {valori[obiettivo]}"


def test_T6_catena_di_dominanza():
    # Valida per QUALSIASI ottimo restituito, anche quando non e' unico.
    xc, xcr, xr = (soluzione("a2-16", o) for o in ("fc", "fcr", "fr"))
    costo = lambda r: r.canonici["costo"]
    regret = lambda r: r.canonici["regret"]
    assert costo(xc) <= costo(xcr) + TOL <= costo(xr) + 2 * TOL
    assert regret(xr) <= regret(xcr) + TOL <= regret(xc) + 2 * TOL


def test_T7_i_rifiuti_non_peggiorano():
    assert soluzione("a2-16", "frcr").obj <= soluzione("a2-16", "fcr").obj + TOL


def test_T8_pesi_limite():
    r = soluzione("a2-16", "fcr", pesi=Pesi(alpha=0.001))
    assert vicino(r.canonici["costo"], 294.2480)

    r = soluzione("a2-16", "fcr", pesi=Pesi(alpha=1000.0))
    assert vicino(r.canonici["regret"], 14.1864)
    assert vicino(r.canonici["costo"], 317.9460)

    r = soluzione("a2-16", "frcr", pesi=Pesi(gamma=100.0))
    assert r.canonici["rifiuti"] == 0 and vicino(r.obj, 330.2431)

    # nessuno servito: l'obiettivo vale tutto gamma * n = 5 * 16
    r = soluzione("a2-16", "frcr", pesi=Pesi(gamma=5.0))
    assert r.canonici["rifiuti"] == 16 and vicino(r.obj, 80.0)


def test_T9_validazioni():
    g = grafo("a2-16")
    solleva(ValueError, coefficienti, "fcmax", Pesi())
    solleva(TypeError, costruisci_con_obiettivo, g, "fc", consenti_rifiuti=True)
    solleva(ValueError, costruisci_con_obiettivo, g, "frcr", pesi=Pesi(gamma=0.0), log=False)

    m = costruisci_modello(g, log=False)
    solleva(ValueError, imposta_somma_pesata, m)                  # tutti i pesi nulli
    solleva(ValueError, imposta_somma_pesata, m, costo=-1.0)      # peso negativo
    solleva(ValueError, imposta_somma_pesata, m, rifiuti=1.0)     # rifiuti senza p_i

    con_rifiuti = costruisci_modello(g, consenti_rifiuti=True, log=False)
    solleva(RuntimeError, risolvi, con_rifiuti)                   # obiettivo mancante


def test_T10_lessicografico():
    for primario, costo_atteso in (("fr", 317.9460), ("frmax", 315.9329)):
        prima, seconda = risolvi_lessicografico(grafo("a2-16"), primario, log=False)
        assert seconda is not None and seconda.ottimo
        assert vicino(seconda.obj, costo_atteso), f"{primario}>fc: {seconda.obj}"
        assert not controlli(seconda)


# ----------------------------------------------------------------------
# Esecuzione senza pytest
# ----------------------------------------------------------------------

def _esegui_tutti() -> int:
    tutti = [(nome, f) for nome, f in globals().items()
             if nome.startswith("test_") and callable(f)]
    falliti = []
    inizio = time.perf_counter()
    for nome, f in tutti:
        t0 = time.perf_counter()
        try:
            f()
            esito = "OK"
        except Exception:
            esito = "FALLITO"
            falliti.append(nome)
            traceback.print_exc()
        print(f"{esito:8} {nome:36} {time.perf_counter() - t0:6.2f} s")

    totale = time.perf_counter() - inizio
    print(f"\n{len(tutti) - len(falliti)}/{len(tutti)} test superati in {totale:.1f} s")
    return 1 if falliti else 0


if __name__ == "__main__":
    sys.exit(_esegui_tutti())
