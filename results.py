"""
results.py — Risoluzione dei modelli e lettura dei risultati.

Riceve un modello gia' costruito (model.py, eventualmente con l'obiettivo impostato da
objectives.py), lo risolve e ne ricava tutto cio' che serve a test, tabelle e relazione:
stato del solver, valore, bound, gap, tempi, rotte dei veicoli e i quattro criteri.

I criteri che dipendono dagli orari (regret, regret massimo) sono calcolati in due modi:
  - canonico: sullo schedule MINIMO di ogni rotta (graph.schedule_minimo). Dipende solo
    dalle rotte, quindi e' riproducibile e confrontabile fra solver. E' quello da
    riportare nelle tabelle.
  - grezzo: dai B.X restituiti dal solver. Quando il regret non e' nell'obiettivo i B.X
    sono arbitrari (analisi_funzioni_obiettivo.md, par. 3.1): serve solo per mostrarlo.
Vale sempre canonico <= grezzo, perche' lo schedule minimo anticipa ogni evento quanto
possibile.
"""

from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from graph import Grafo, Nodo, Arco, schedule_minimo
from objectives import (Pesi, costruisci_con_obiettivo, imposta_obiettivo,
                        aggiungi_vincolo_livello, soglia_con_tolleranza,
                        valuta_criteri, valore_obiettivo)

# Tolleranza relativa sui controlli di coerenza fra valori del solver e valori ricalcolati.
TOL = 1e-6


# ----------------------------------------------------------------------
# Blocco 1 — Stato del solver
# ----------------------------------------------------------------------

_NOMI_STATO = {
    GRB.OPTIMAL: "ottimo",
    GRB.INFEASIBLE: "infeasible",
    GRB.INF_OR_UNBD: "infeasible_o_illimitato",
    GRB.TIME_LIMIT: "time_limit",
    GRB.INTERRUPTED: "interrotto",
}


def nome_stato(codice: int) -> str:
    """Nome leggibile di un codice di stato Gurobi (m.Status)."""
    return _NOMI_STATO.get(codice, f"stato_{codice}")


# ----------------------------------------------------------------------
# Blocco 2 — Rotte
# ----------------------------------------------------------------------

@dataclass
class Rotta:
    """
    Il giro di un veicolo, dal deposito al deposito.

    nodi, localita e schedule hanno la stessa lunghezza e lo stesso indice: la posizione
    k e' il k-esimo evento della rotta. archi e tempi hanno un elemento in meno.
    """
    archi: list[Arco]
    nodi: list[Nodo]              # [deposito, v1, ..., vk, deposito]
    localita: list[int]           # id fisici; il deposito compare come iniziale e finale
    tempi: list[float]            # tempi di percorrenza degli archi (m._tempo)
    costo: float
    schedule: list[float]         # schedule minimo, uno per posizione

    @property
    def utenti(self) -> list[int]:
        """Utenti serviti, nell'ordine in cui salgono."""
        return [v[0] for v in self.nodi if v[0] > 0]

    def vincoli_ride(self) -> list[tuple[int, int]]:
        """Coppie (posizione del pickup, posizione del drop-off), una per utente."""
        salita = {v[0]: k for k, v in enumerate(self.nodi) if v[0] > 0}
        discesa = {-v[0]: k for k, v in enumerate(self.nodi) if v[0] < 0}
        return [(salita[i], discesa[i]) for i in salita]

    def arrivi(self) -> dict[int, float]:
        """Utente -> inizio del servizio al suo drop-off, sullo schedule minimo."""
        return {-v[0]: self.schedule[k] for k, v in enumerate(self.nodi) if v[0] < 0}


def _costruisci_rotta(m: gp.Model, archi: list[Arco]) -> Rotta:
    """Completa una rotta a partire dai suoi archi, schedule minimo compreso."""
    istanza = m._istanza
    nodi = [a[0] for a in archi] + [archi[-1][1]]
    # coda e testa differiscono solo sul deposito: primo nodo = deposito iniziale,
    # ultimo nodo = deposito finale. Per i nodi intermedi coincidono.
    localita = [m._coda[a[0]] for a in archi] + [m._testa[archi[-1][1]]]
    tempi = [m._tempo[a] for a in archi]
    costo = sum(m._costo[a] for a in archi)

    salite = {v[0] for v in nodi if v[0] > 0}
    discese = {-v[0] for v in nodi if v[0] < 0}
    if salite != discese:
        raise RuntimeError(f"rotta incoerente: salgono {sorted(salite)}, "
                           f"scendono {sorted(discese)}")

    provvisoria = Rotta(archi, nodi, localita, tempi, costo, schedule=[])
    schedule = schedule_minimo(localita, provvisoria.vincoli_ride(), istanza, tempi)
    if schedule is None:
        raise RuntimeError(
            "il solver ha restituito una rotta senza schedule ammissibile: "
            f"{[v[0] for v in nodi]}. Possibile violazione entro le tolleranze "
            "numeriche del solver, oppure incoerenza fra model.py e graph.py."
        )
    provvisoria.schedule = schedule
    return provvisoria


def estrai_rotte(m: gp.Model) -> list[Rotta]:
    """
    Ricostruisce le rotte dagli archi con x_a = 1.

    Ogni nodo diverso dal deposito ha al piu' un arco uscente attivo (ogni evento
    avviene una volta sola), quindi da ogni partenza dal deposito si segue la catena
    fino al rientro. Se avanzano archi, il solver ha restituito un sottociclo: con
    tempi di servizio positivi e' impossibile, quindi e' un errore da segnalare.
    """
    deposito = m._deposito
    attivi = [a for a in m._archi if m._x[a].X > 0.5]

    partenze = [a for a in attivi if a[0] == deposito]
    successivo: dict[Nodo, Arco] = {}
    for a in attivi:
        if a[0] == deposito:
            continue
        if a[0] in successivo:
            raise RuntimeError(f"il nodo {a[0]} ha due archi uscenti attivi")
        successivo[a[0]] = a

    rotte = []
    for arco in partenze:
        archi = [arco]
        while archi[-1][1] != deposito:
            nodo = archi[-1][1]
            if nodo not in successivo:
                raise RuntimeError(f"il nodo {nodo} non ha un arco uscente attivo")
            archi.append(successivo.pop(nodo))
        rotte.append(_costruisci_rotta(m, archi))

    if successivo:
        raise RuntimeError(f"archi attivi non raggiungibili dal deposito (sottociclo): "
                           f"{sorted(successivo)}")
    return rotte


# ----------------------------------------------------------------------
# Blocco 3 — Il risultato di una risoluzione
# ----------------------------------------------------------------------

@dataclass
class Risultato:
    istanza: str
    n: int
    variante: str
    obiettivo: str | None
    coefficienti: dict[str, float] | None
    stato: str
    tempo: float                                   # m.Runtime, secondi
    nodi_bb: float                                 # nodi del branch-and-bound
    obj: float | None = None                       # m.ObjVal
    bound: float | None = None                     # m.ObjBound
    gap: float | None = None                       # m.MIPGap (relativo)
    rotte: list[Rotta] = field(default_factory=list)
    canonici: dict[str, float] = field(default_factory=dict)   # schedule minimo
    grezzi: dict[str, float] = field(default_factory=dict)     # B.X del solver
    somma_d: float | None = None                   # somma d_i.X, se le d_i esistono

    @property
    def ha_soluzione(self) -> bool:
        return self.obj is not None

    @property
    def ottimo(self) -> bool:
        return self.stato == "ottimo"

    def riga(self) -> dict:
        """Dizionario piatto con i campi principali del risultato (per esportazioni e tabelle)."""
        c, g = self.canonici, self.grezzi
        return {
            "istanza": self.istanza, "n": self.n, "variante": self.variante,
            "obiettivo": self.obiettivo, "stato": self.stato,
            "obj": self.obj, "bound": self.bound, "gap": self.gap,
            "tempo": self.tempo, "nodi_bb": self.nodi_bb,
            "veicoli": len(self.rotte) if self.ha_soluzione else None,
            "costo": c.get("costo"),
            "regret": c.get("regret"), "regret_grezzo": g.get("regret"),
            "regret_max": c.get("regret_max"), "regret_max_grezzo": g.get("regret_max"),
            "rifiuti": c.get("rifiuti"), "ar": c.get("ar"),
            "regret_medio_servito": c.get("regret_medio_servito"),
        }


def _arrivi_grezzi(m: gp.Model, rotte: list[Rotta]) -> dict[int, float]:
    """Utente -> B.X al suo drop-off, letto dal solver."""
    return {-v[0]: m._B[v].X for r in rotte for v in r.nodi if v[0] < 0}


def risolvi(m: gp.Model) -> Risultato:
    """Risolve il modello e ne legge i risultati."""
    if m._richiede_obiettivo:
        raise RuntimeError("il modello consente rifiuti ma non ha ancora un obiettivo che "
                           "li penalizzi: impostarlo con objectives.py prima di risolvere")

    m.optimize()
    istanza = m._istanza
    r = Risultato(
        istanza=istanza.nome, n=istanza.n, variante=m._variante,
        obiettivo=m._obiettivo, coefficienti=dict(m._coefficienti),
        stato=nome_stato(m.Status), tempo=m.Runtime, nodi_bb=m.NodeCount,
    )
    if m.SolCount == 0:
        return r

    r.obj, r.bound, r.gap = m.ObjVal, m.ObjBound, m.MIPGap
    r.rotte = estrai_rotte(m)
    costo = sum(rotta.costo for rotta in r.rotte)
    arrivi = {i: b for rotta in r.rotte for i, b in rotta.arrivi().items()}
    r.canonici = valuta_criteri(istanza, costo, arrivi)
    r.grezzi = valuta_criteri(istanza, costo, _arrivi_grezzi(m, r.rotte))
    if m._d is not None:
        r.somma_d = sum(d.X for d in m._d.values())
    return r


# ----------------------------------------------------------------------
# Blocco 4 — Controlli di coerenza (usati da test.py)
# ----------------------------------------------------------------------

def _vicini(a: float, b: float) -> bool:
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def controlli(r: Risultato) -> list[str]:
    """
    Controlli che non richiedono valori di riferimento. Lista vuota = tutto coerente.

    Sempre:
      - regret canonico <= regret grezzo, a meno delle tolleranze del solver (lo
        schedule minimo non puo' essere peggiore);
    Solo all'ottimo dimostrato con gap nullo, perche' fuori dall'ottimo le variabili
    ausiliarie possono avere "gioco" e i valori letti non sono piu' vincolati:
      - T5: l'obiettivo ricalcolato dai criteri coincide con ObjVal;
      - T3: se il regret e' nell'obiettivo, somma d_i.X = regret canonico.
    """
    problemi = []
    if not r.ha_soluzione:
        return problemi

    # I B.X rispettano i vincoli solo entro la tolleranza di ammissibilita' del solver
    # (FeasibilityTol, 1e-6 di default): possono stare di poco SOTTO lo schedule
    # minimo esatto. Margine: 1e-5 per utente.
    margine = 1e-5 * max(1, r.n)
    if r.canonici["regret"] > r.grezzi["regret"] + margine:
        problemi.append(f"regret canonico {r.canonici['regret']:.6f} maggiore del "
                        f"grezzo {r.grezzi['regret']:.6f}")

    if r.ottimo and r.gap is not None and r.gap <= 1e-8:
        ricalcolato = valore_obiettivo(r.coefficienti, r.grezzi)
        if not _vicini(ricalcolato, r.obj):
            problemi.append(f"T5: obiettivo ricalcolato {ricalcolato:.6f} != {r.obj:.6f}")
        if r.somma_d is not None and r.coefficienti.get("regret", 0) > 0:
            if not _vicini(r.somma_d, r.canonici["regret"]):
                problemi.append(f"T3: somma d_i {r.somma_d:.6f} != regret canonico "
                                f"{r.canonici['regret']:.6f}")
    return problemi


# ----------------------------------------------------------------------
# Blocco 5 — Lessicografico in due fasi
# ----------------------------------------------------------------------

# Obiettivo "puro" -> criterio da congelare nella seconda fase.
CRITERIO_PRIMARIO = {"fr": "regret", "frmax": "regret_max", "fn": "rifiuti"}


def risolvi_lessicografico(grafo: Grafo, primario: str, secondario: str = "fc", *,
                           pesi: Pesi | None = None,
                           **opzioni_modello) -> tuple[Risultato, Risultato | None]:
    """
    Prima fase: ottimizza l'obiettivo puro `primario` e ne ricava il valore ottimo f*.
    Seconda fase: sullo stesso modello aggiunge  criterio <= f* (con tolleranza) e
    ottimizza `secondario`. Risultato: la soluzione migliore per `secondario` fra quelle
    ottime per `primario` (su a2-16, fr poi fc: costo 317.946).

    Se la prima fase non raggiunge l'ottimo, la seconda non ha senso: restituisce None.
    """
    if primario not in CRITERIO_PRIMARIO:
        raise ValueError(f"primario deve essere un obiettivo puro: "
                         f"{sorted(CRITERIO_PRIMARIO)}, ricevuto {primario!r}")

    m = costruisci_con_obiettivo(grafo, primario, pesi=pesi, **opzioni_modello)
    prima = risolvi(m)
    if not prima.ottimo:
        return prima, None

    aggiungi_vincolo_livello(m, CRITERIO_PRIMARIO[primario],
                             soglia_con_tolleranza(prima.obj))
    imposta_obiettivo(m, secondario, pesi)
    seconda = risolvi(m)
    seconda.obiettivo = f"{primario}>{secondario}"
    return prima, seconda


# ----------------------------------------------------------------------
# Blocco 6 — Stampa leggibile
# ----------------------------------------------------------------------

def descrivi(r: Risultato) -> str:
    """Riassunto testuale: intestazione, criteri, una riga per veicolo."""
    testa = (f"{r.istanza} | Model {r.variante} | {r.obiettivo} | {r.stato} | "
             f"{r.tempo:.2f} s")
    if not r.ha_soluzione:
        return testa + " | nessuna soluzione"
    c, g = r.canonici, r.grezzi
    righe = [
        testa,
        f"  obj {r.obj:.4f}  bound {r.bound:.4f}  gap {r.gap:.2e}  veicoli {len(r.rotte)}",
        f"  costo {c['costo']:.4f}  serviti {r.n - round(c['rifiuti'])}/{r.n}",
        f"  regret     canonico {c['regret']:9.4f}   grezzo {g['regret']:9.4f}",
        f"  regret max canonico {c['regret_max']:9.4f}   grezzo {g['regret_max']:9.4f}",
    ]
    for k, rotta in enumerate(r.rotte, start=1):
        eventi = " ".join(f"{'+' if v[0] > 0 else '-'}{abs(v[0])}" for v in rotta.nodi[1:-1])
        righe.append(f"  veicolo {k}: primo servizio {rotta.schedule[1]:7.2f}, rientro "
                     f"{rotta.schedule[-1]:7.2f}, costo {rotta.costo:8.3f} | {eventi}")
    return "\n".join(righe)


if __name__ == "__main__":
    import sys

    from instances import leggi_istanza
    from model import costruisci_modello

    percorso = sys.argv[1] if len(sys.argv) > 1 else "dati_milp/a2-16.txt"
    grafo = Grafo.costruisci(leggi_istanza(percorso))

    esiti = []
    r = risolvi(costruisci_modello(grafo, log=False))
    esiti.append(r)
    print(descrivi(r) + "\n")

    for nome in ("fr", "frcr"):
        r = risolvi(costruisci_con_obiettivo(grafo, nome, log=False))
        esiti.append(r)
        print(descrivi(r) + "\n")

    prima, seconda = risolvi_lessicografico(grafo, "fr", log=False)
    esiti += [prima, seconda]
    print(descrivi(seconda) + "\n")

    problemi = [f"{e.obiettivo}: {p}" for e in esiti if e is not None for p in controlli(e)]
    print("\n".join(problemi) if problemi else "Controlli di coerenza superati.")
