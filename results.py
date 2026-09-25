"""
Risoluzione dei modelli e lettura dei risultati: stato del solver, valore, bound,
gap, tempi, rotte dei veicoli e i quattro criteri.

Regret e regret massimo sono calcolati in due modi:
  - canonico: sullo schedule minimo di ogni rotta (graph.schedule_minimo). Dipende
    solo dalle rotte, quindi è riproducibile; è il valore riportato nelle tabelle.
  - grezzo: dai B.X del solver. Se il regret non è nell'obiettivo, i B.X possono
    assumere qualunque valore ammissibile, quindi il dato è solo indicativo.
Il canonico è sempre <= del grezzo, perché lo schedule minimo anticipa ogni evento
il più possibile.
"""

from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from graph import Grafo, Nodo, Arco, schedule_minimo
from objectives import costruisci_con_obiettivo, valuta_criteri, valore_obiettivo

# Tolleranza relativa sui controlli di coerenza fra valori del solver e valori ricalcolati.
TOL = 1e-6


# ----------------------------------------------------------------------
# Stato del solver
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
# Rotte
# ----------------------------------------------------------------------

@dataclass
class Rotta:
    """
    Giro di un veicolo dal deposito al deposito.
    nodi, localita e schedule sono allineati per posizione.
    """
    nodi: list[Nodo]              # [deposito, v1, ..., vk, deposito]
    localita: list[int]           # id fisici; il deposito compare come iniziale e finale
    costo: float
    schedule: list[float]         # schedule minimo, uno per posizione

    def vincoli_ride(self) -> list[tuple[int, int]]:
        """Coppie (posizione del pickup, posizione del drop-off), una per utente."""
        salita = {v[0]: k for k, v in enumerate(self.nodi) if v[0] > 0}
        discesa = {-v[0]: k for k, v in enumerate(self.nodi) if v[0] < 0}
        return [(salita[i], discesa[i]) for i in salita]

    def arrivi(self) -> dict[int, float]:
        """Utente -> inizio del servizio al suo drop-off, sullo schedule minimo."""
        return {-v[0]: self.schedule[k] for k, v in enumerate(self.nodi) if v[0] < 0}


def _costruisci_rotta(m: gp.Model, archi: list[Arco]) -> Rotta:
    """Costruisce la Rotta dai suoi archi e ne calcola lo schedule minimo."""
    istanza = m._istanza
    nodi = [a[0] for a in archi] + [archi[-1][1]]
    # coda e testa coincidono tranne che per il deposito (primo e ultimo nodo)
    localita = [m._coda[a[0]] for a in archi] + [m._testa[archi[-1][1]]]
    tempi = [m._tempo[a] for a in archi]
    costo = sum(m._costo[a] for a in archi)

    salite = {v[0] for v in nodi if v[0] > 0}
    discese = {-v[0] for v in nodi if v[0] < 0}
    if salite != discese:
        raise RuntimeError(f"rotta incoerente: salgono {sorted(salite)}, "
                           f"scendono {sorted(discese)}")

    provvisoria = Rotta(nodi, localita, costo, schedule=[])
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
    Ricostruisce le rotte dagli archi con x_a = 1. Ogni nodo diverso dal deposito ha
    al più un arco uscente attivo, quindi da ogni partenza si segue la catena fino al
    rientro. Archi avanzati indicherebbero un sottociclo, impossibile con tempi di
    servizio positivi.
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
# Risultato di una risoluzione
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
        """Campi principali del risultato come dizionario: una riga del CSV di esperimenti.py."""
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
# Controlli di coerenza (usati da test.py)
# ----------------------------------------------------------------------

def _vicini(a: float, b: float) -> bool:
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def controlli(r: Risultato) -> list[str]:
    """
    Controlli di coerenza che non richiedono valori di riferimento (lista vuota se è
    tutto a posto). Sempre: regret canonico <= regret grezzo, entro le tolleranze.
    Solo all'ottimo con gap nullo, dove le variabili ausiliarie non hanno più margine:
    l'obiettivo ricalcolato dai criteri deve coincidere con ObjVal e, se il regret è
    nell'obiettivo, la somma delle d_i deve coincidere con il regret canonico.
    """
    problemi = []
    if not r.ha_soluzione:
        return problemi

    # i B.X rispettano i vincoli solo entro FeasibilityTol (1e-6), quindi possono
    # stare di poco sotto lo schedule minimo: margine di 1e-5 per utente
    margine = 1e-5 * max(1, r.n)
    if r.canonici["regret"] > r.grezzi["regret"] + margine:
        problemi.append(f"regret canonico {r.canonici['regret']:.6f} maggiore del "
                        f"grezzo {r.grezzi['regret']:.6f}")

    if r.ottimo and r.gap is not None and r.gap <= 1e-8:
        ricalcolato = valore_obiettivo(r.coefficienti, r.grezzi)
        if not _vicini(ricalcolato, r.obj):
            problemi.append(f"obiettivo ricalcolato {ricalcolato:.6f} != {r.obj:.6f}")
        if r.somma_d is not None and r.coefficienti.get("regret", 0) > 0:
            if not _vicini(r.somma_d, r.canonici["regret"]):
                problemi.append(f"somma d_i {r.somma_d:.6f} != regret canonico "
                                f"{r.canonici['regret']:.6f}")
    return problemi


# ----------------------------------------------------------------------
# Stampa
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

    problemi = [f"{e.obiettivo}: {p}" for e in esiti for p in controlli(e)]
    print("\n".join(problemi) if problemi else "Controlli di coerenza superati.")