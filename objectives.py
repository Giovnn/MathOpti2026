"""
Le sette funzioni obiettivo del paper (Sez. 3.5), scritte come somme pesate di
quattro criteri: costo (f_c, eq. 10), richieste rifiutate (f_n), regret totale
(f_r, eq. 11 e 13) e regret massimo (f_rmax, eq. 12 e 14).

    obiettivo   costo  regret  regret_max  rifiuti
    fc            1
    fn                                        1
    fr                   1
    frmax                         1
    fcr           1    alpha
    fcrmax        1              beta
    frcr          1    alpha                gamma

Pesi del paper (Sez. 4.2): alpha = 1, beta = n/5, gamma = 20. Sono pensati per
costi in km e tempi in minuti; sui Cordeau, dove costo e tempo coincidono, lo
stesso peso ha un effetto diverso, per questo si possono cambiare (classe Pesi).
"""

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from graph import Grafo
from instances import Istanza
from model import costruisci_modello, etichetta

# Tolleranza sui tempi letti dal solver (Gurobi lavora con tolleranze di circa 1e-6),
# per questo è più larga dell'EPS = 1e-9 di model.py.
EPS = 1e-6

CRITERI = ("costo", "regret", "regret_max", "rifiuti")
OBIETTIVI = ("fc", "fn", "fr", "frmax", "fcr", "fcrmax", "frcr")


# ----------------------------------------------------------------------
# Pesi
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Pesi:
    """
    Pesi delle somme pesate (15)-(17): alpha per il regret totale (fcr, frcr),
                                       beta per il regret massimo (fcrmax),
                                       gamma per ogni richiesta rifiutata (frcr).
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 20.0


def pesi_paper(n: int) -> Pesi:
    """
    Pesi del paper (Sez. 4.2): alpha = 1, beta = n/5, gamma = 20.
    beta cresce con n perché il costo cresce con n e d_max no.
    """
    return Pesi(alpha=1.0, beta=n / 5, gamma=20.0)


# ----------------------------------------------------------------------
# Obiettivi del paper
# ----------------------------------------------------------------------

def coefficienti(nome: str, pesi: Pesi) -> dict[str, float]:
    """
    Pesi dei criteri per l'obiettivo `nome`; i criteri assenti hanno peso zero.
    """
    tabella = {
        "fc":     {"costo": 1.0},
        "fn":     {"rifiuti": 1.0},
        "fr":     {"regret": 1.0},
        "frmax":  {"regret_max": 1.0},
        "fcr":    {"costo": 1.0, "regret": pesi.alpha},
        "fcrmax": {"costo": 1.0, "regret_max": pesi.beta},
        "frcr":   {"costo": 1.0, "regret": pesi.alpha, "rifiuti": pesi.gamma},
    }
    if nome not in tabella:
        raise ValueError(f"obiettivo sconosciuto: {nome!r} (attesi {OBIETTIVI})")
    return tabella[nome]


def richiede_rifiuti(nome: str) -> bool:
    """
    True se l'obiettivo penalizza i rifiuti (fn, frcr): servono le p_i e il
    vincolo (3) al posto di (1c). I pesi qui non contano, conta quali criteri compaiono.
    """
    return "rifiuti" in coefficienti(nome, Pesi())


# ----------------------------------------------------------------------
# Variabili del regret, eq. (11)-(12)
# ----------------------------------------------------------------------

def _bound_regret(istanza: Istanza, i: int) -> float:
    """
    Bound superiore di d_i: l_i- - e_i-. Non taglia soluzioni, perché ogni B_v
    di drop-off di i ha già ub = l_i-.
    """
    drop = istanza.delivery(i)
    return drop.l - drop.e


def aggiungi_regret(m: gp.Model) -> dict[int, gp.Var]:
    """
    Variabili d_i >= 0 e vincoli (11): d_i >= B_v - e_i- per ogni nodo v di drop-off di i.

    Il vincolo si scrive su tutti i nodi di drop-off di i, senza big-M: i nodi non usati
    possono sempre scendere a B_v = e_i-, quindi quando d_i è minimizzata vale il regret
    del nodo usato (0 se la richiesta è rifiutata). Se le d_i esistono già, le restituisce.
    """
    if m._d is not None:
        return m._d

    istanza, grafo, B = m._istanza, m._grafo, m._B

    d: dict[int, gp.Var] = {}
    for i in istanza.utenti():
        e_meno = istanza.delivery(i).e
        d[i] = m.addVar(lb=0.0, ub=_bound_regret(istanza, i),
                        vtype=GRB.CONTINUOUS, name=f"d[{i}]")
        # stesso ordine di creazione di model.py
        for v in sorted(grafo.nodi_delivery(i)):
            m.addConstr(d[i] >= B[v] - e_meno, name=f"regret[{i},{etichetta(v)}]")

    m._d = d
    return d


def aggiungi_regret_max(m: gp.Model) -> gp.Var:
    """
    Variabile d_max >= 0 e vincoli (12): d_max >= d_i per ogni i (crea le d_i se mancano).
    Con frmax e fcrmax si minimizza solo il caso peggiore: le altre d_i possono stare
    ovunque fra il regret vero e d_max, quindi i loro valori non sono regret.
    """
    if m._dmax is not None:
        return m._dmax

    d = aggiungi_regret(m)
    istanza = m._istanza
    ub = max(_bound_regret(istanza, i) for i in istanza.utenti())

    dmax = m.addVar(lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name="dmax")
    for i in istanza.utenti():
        m.addConstr(dmax >= d[i], name=f"regret_max[{i}]")

    m._dmax = dmax
    return dmax


# ----------------------------------------------------------------------
# Espressioni dei criteri
# ----------------------------------------------------------------------

def espressione(m: gp.Model, criterio: str):
    """
    Espressione Gurobi di un criterio; crea d_i e d_max solo se servono.
    "rifiuti" include la costante n, come la colonna Obj.v. della Tabella 9.
    """
    if criterio == "costo":
        return gp.quicksum(m._costo[a] * m._x[a] for a in m._archi)
    if criterio == "regret":
        return gp.quicksum(aggiungi_regret(m).values())
    if criterio == "regret_max":
        return aggiungi_regret_max(m)
    if criterio == "rifiuti":
        if m._p is None:
            raise ValueError("il criterio 'rifiuti' richiede un modello costruito "
                             "con consenti_rifiuti=True")
        return m._istanza.n - gp.quicksum(m._p.values())
    raise ValueError(f"criterio sconosciuto: {criterio!r} (attesi {CRITERI})")


# ----------------------------------------------------------------------
# Vincoli di livello
# ----------------------------------------------------------------------

def _vincoli_livello(m: gp.Model) -> list:
    """
    Lista (criterio, vincolo) dei vincoli di livello presenti nel modello.
    """
    try:
        return m._livelli
    except AttributeError:
        m._livelli = []
        return m._livelli


def soglia_con_tolleranza(valore: float, rel: float = 1e-6) -> float:
    """
    Soglia per la seconda fase del lessicografico: il valore più un piccolo margine
    relativo, perché chiedere <= valore esatto può risultare infeasible per le tolleranze.
    """
    return valore + rel * max(1.0, abs(valore))


def aggiungi_vincolo_livello(m: gp.Model, criterio: str, soglia: float,
                             *, nome: str | None = None) -> gp.Constr:
    """
    Aggiunge il vincolo criterio <= soglia e lo restituisce (seconda fase del
    lessicografico, vedi results.risolvi_lessicografico).
    """
    vincolo = m.addConstr(espressione(m, criterio) <= soglia,
                          name=nome or f"livello[{criterio}]")
    _vincoli_livello(m).append((criterio, vincolo))
    m.update()
    return vincolo


def rimuovi_vincolo_livello(m: gp.Model, vincolo: gp.Constr) -> None:
    """Toglie un vincolo di livello dal modello e dal registro."""
    registro = _vincoli_livello(m)
    rimasti = [(c, v) for c, v in registro if v is not vincolo]
    if len(rimasti) == len(registro):
        raise ValueError("il vincolo non e' un vincolo di livello di questo modello")
    m.remove(vincolo)
    registro[:] = rimasti
    m.update()


# ----------------------------------------------------------------------
# Obiettivo
# ----------------------------------------------------------------------

def imposta_somma_pesata(m: gp.Model, *, costo: float = 0.0, regret: float = 0.0,
                         regret_max: float = 0.0, rifiuti: float = 0.0,
                         nome: str = "personalizzato") -> None:
    """
    Imposta come obiettivo (da minimizzare) la somma pesata dei quattro criteri.
    Rifiuta pesi negativi o tutti nulli, e un modello con rifiuti ammessi in cui nulla
    li penalizza (l'ottimo sarebbe non servire nessuno). È l'unico punto che abbassa
    il flag m._richiede_obiettivo messo da model.py.
    """
    pesi = {"costo": costo, "regret": regret, "regret_max": regret_max, "rifiuti": rifiuti}

    if any(w < 0 for w in pesi.values()):
        raise ValueError(f"{nome}: pesi negativi non ammessi: {pesi}")
    if all(w == 0 for w in pesi.values()):
        raise ValueError(f"{nome}: almeno un peso deve essere positivo")

    rifiuti_limitati = any(c == "rifiuti" for c, _ in _vincoli_livello(m))
    if m._p is not None and rifiuti == 0 and not rifiuti_limitati:
        raise ValueError(f"{nome}: il modello consente rifiuti ma l'obiettivo non li "
                         "penalizza e nessun vincolo di livello li limita: l'ottimo "
                         "sarebbe la soluzione vuota")
    if m._p is None and rifiuti > 0:
        raise ValueError(f"{nome}: il peso sui rifiuti richiede un modello costruito "
                         "con consenti_rifiuti=True")

    obiettivo = gp.LinExpr()
    for criterio, peso in pesi.items():
        if peso > 0:
            obiettivo += peso * espressione(m, criterio)

    m.setObjective(obiettivo, GRB.MINIMIZE)
    m._richiede_obiettivo = False
    m._obiettivo = nome
    m._coefficienti = pesi
    m.update()


def imposta_obiettivo(m: gp.Model, nome: str, pesi: Pesi | None = None) -> None:
    """
    Imposta uno dei sette obiettivi del paper; senza pesi espliciti usa pesi_paper(n).
    """
    if pesi is None:
        pesi = pesi_paper(m._istanza.n)
    imposta_somma_pesata(m, nome=nome, **coefficienti(nome, pesi))


def costruisci_con_obiettivo(grafo: Grafo, nome: str, *, pesi: Pesi | None = None,
                             **opzioni_modello) -> gp.Model:
    """
    Costruisce il modello con l'obiettivo `nome`. consenti_rifiuti lo decide
    l'obiettivo; le altre opzioni passano direttamente a costruisci_modello.
    """
    if "consenti_rifiuti" in opzioni_modello:
        raise TypeError("consenti_rifiuti e' deciso dall'obiettivo, non va passato")
    m = costruisci_modello(grafo, consenti_rifiuti=richiede_rifiuti(nome), **opzioni_modello)
    imposta_obiettivo(m, nome, pesi)
    return m


# ----------------------------------------------------------------------
# Valutazione dei criteri su una soluzione
# ----------------------------------------------------------------------

def regret_utente(istanza: Istanza, i: int, inizio_dropoff: float) -> float:
    """
    Regret dell'utente i: ritardo dell'arrivo al drop-off rispetto a e_i-.
    I negativi entro EPS sono arrotondamenti e diventano 0; oltre, è un errore.
    """
    r = inizio_dropoff - istanza.delivery(i).e
    if r < -EPS:
        raise ValueError(f"utente {i}: drop-off a {inizio_dropoff:.6f}, prima di "
                         f"e_i- = {istanza.delivery(i).e:.6f}: orari incoerenti")
    return max(0.0, r)


def valuta_criteri(istanza: Istanza, costo_rotte: float,
                   arrivi: dict[int, float]) -> dict[str, float]:
    """
    Valori dei quattro criteri di una soluzione, più ar (% di richieste servite) e il
    regret medio sugli utenti serviti.
    arrivi: {utente servito: inizio del servizio al drop-off}; chi manca è rifiutato.
    Gli orari possono venire dallo schedule minimo o dai B.X del solver (vedi results.py).
    """
    sconosciuti = set(arrivi) - set(istanza.utenti())
    if sconosciuti:
        raise ValueError(f"utenti inesistenti negli arrivi: {sorted(sconosciuti)}")

    regret = {i: regret_utente(istanza, i, b) for i, b in arrivi.items()}
    serviti = len(regret)
    totale = sum(regret.values())
    return {
        "costo": costo_rotte,
        "rifiuti": istanza.n - serviti,
        "regret": totale,
        "regret_max": max(regret.values(), default=0.0),
        "ar": 100.0 * serviti / istanza.n,
        "regret_medio_servito": totale / serviti if serviti else 0.0,
    }


def valore_obiettivo(coeff: dict[str, float], criteri: dict[str, float]) -> float:
    """
    Valore dell'obiettivo ricalcolato dai criteri (coeff: criterio -> peso, es. m._coefficienti).
    """
    return sum(peso * criteri[c] for c, peso in coeff.items())


# ----------------------------------------------------------------------
# Verifica rapida: python objectives.py [istanza]
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from instances import leggi_istanza
    from model import VARIANTI

    def criteri_grezzi(m: gp.Model) -> dict[str, float]:
        """
        Criteri calcolati dai valori B.X restituiti dal solver.
        """
        istanza = m._istanza
        costo_rotte = sum(m._costo[a] for a in m._archi if m._x[a].X > 0.5)
        arrivi = {}
        for i in istanza.utenti():
            attivi = [v for v in m._grafo.nodi_delivery(i)
                      if m._flusso_in[v].getValue() > 0.5]
            if attivi:
                arrivi[i] = m._B[attivi[0]].X
        return valuta_criteri(istanza, costo_rotte, arrivi)

    percorso = sys.argv[1] if len(sys.argv) > 1 else "dati_milp/a2-16.txt"
    istanza = leggi_istanza(percorso)
    grafo = Grafo.costruisci(istanza)
    print(f"{istanza.nome}: n = {istanza.n}, pesi del paper = {pesi_paper(istanza.n)}")
    print(f"{'obiettivo':<9} {'Model I':>11} {'Model II':>11} {'serviti':>8}  controlli")

    tutto_ok = True
    for nome in OBIETTIVI:
        valori, serviti, controlli = {}, "", []
        for variante in VARIANTI:
            m = costruisci_con_obiettivo(grafo, nome, variante=variante, log=False)
            m.optimize()
            if m.SolCount == 0:
                valori[variante] = None
                controlli.append(f"{variante}: nessuna soluzione (status {m.Status})")
                continue
            valori[variante] = m.ObjVal

            # l'obiettivo ricalcolato dai criteri deve coincidere con ObjVal
            criteri = criteri_grezzi(m)
            ricalcolato = valore_obiettivo(m._coefficienti, criteri)
            if abs(ricalcolato - m.ObjVal) > 1e-4 * max(1.0, abs(m.ObjVal)):
                controlli.append(f"{variante}: obbiettivo ricalcolato diverso ({ricalcolato:.4f})")
            if m._p is not None:
                serviti = f"{istanza.n - round(criteri['rifiuti'])}/{istanza.n}"

        # Model I e Model II sono equivalenti: devono trovare lo stesso ottimo.
        if None not in valori.values() and abs(valori["I"] - valori["II"]) > 1e-4:
            controlli.append("Model I e II DIVERSI")
        tutto_ok = tutto_ok and not controlli

        testo = [f"{valori[v]:11.4f}" if valori[v] is not None else f"{'-':>11}"
                 for v in VARIANTI]
        print(f"{nome:<9} {testo[0]} {testo[1]} {serviti:>8}  "
              f"{'; '.join(controlli) or 'OK'}")

    print("\nTutti i controlli superati." if tutto_ok else "\nCi sono controlli falliti.")