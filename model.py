"""
I due modelli MILP event-based per il DARP, Model I e Model II
(Gaul, Klamroth & Stiglmayr 2022, Sez. 3.2-3.4). L'obiettivo di default è il
costo f_c (eq. 10); le altre funzioni obiettivo sono in objectives.py.

Alcuni vincoli del DARP non compaiono come righe del modello:
  - capacità: solo le Q-tuple ammissibili diventano nodi;
  - pairing e precedenza: sono nella struttura degli archi A1-A6;
  - coppie incompatibili: eliminate da f1/f2 in graph.py;
  - sottotour: esclusi da (2b)/(9b), perché sommando B_w >= B_v + s + t lungo un
    ciclo si ottiene 0 >= somma(s + t), impossibile se un tempo è positivo.
    Per questo l'Esempio 1 del paper (s = t = 0) serve a verificare il grafo, non il MILP.

Il deposito è un solo nodo con una sola variabile B_0. Gli archi uscenti dal
deposito usano e_0 al posto di B_0, (2c)/(9c); quelli entranti restano in (2b)/(9b),
quindi B_0 è l'istante di rientro dell'ultimo veicolo e il suo bound superiore
impone la durata massima T del servizio.
"""

import gurobipy as gp
from gurobipy import GRB

from instances import Istanza
from graph import Grafo, Nodo, Arco, localita_id, EPS

VARIANTI = ("I", "II")


# ----------------------------------------------------------------------
# Etichette leggibili per variabili e vincoli
# ----------------------------------------------------------------------

def etichetta(nodo: Nodo) -> str:
    """
    Nome del nodo per variabili e vincoli, es. (-2, 1, 0) -> 'm2_1_0'
    (il segno meno non è ammesso nei nomi del formato LP).
    """
    return "_".join(f"m{-c}" if c < 0 else str(c) for c in nodo)


def etichetta_arco(arco: Arco) -> str:
    v, w = arco
    return f"{etichetta(v)}__{etichetta(w)}"


# ----------------------------------------------------------------------
# Precalcoli: localita' fisiche, big-M, finestre
# ----------------------------------------------------------------------

def mappa_localita(grafo: Grafo) -> tuple[dict[Nodo, int], dict[Nodo, int]]:
    """
    Id della località fisica di ogni nodo, in due mappe: coda (nodo come origine di
    un arco) e testa (nodo come destinazione). Differiscono solo sul deposito, che
    ha due id fisici (0 iniziale, 2n+1 finale).
    """
    istanza = grafo.istanza
    coda = {v: localita_id(v, istanza, is_partenza=True) for v in grafo.nodi}
    testa = {v: localita_id(v, istanza, is_partenza=False) for v in grafo.nodi}
    return coda, testa


def calcola_M_ride(istanza: Istanza) -> dict[int, float]:
    """
    M_i = l_i- - e_i+ - s_i+ - L_i (Sez. 3.3): big-M del ride time nel Model I (2e) e
    coefficiente delle finestre riformulate del Model II (9e)/(9f), dove il paper lo
    chiama TW. Si calcola dalle finestre effettive, così resta valido anche quando il
    preprocessing le ha strette. Se è negativo, la richiesta è infeasible in sé.
    """
    M: dict[int, float] = {}
    for i in istanza.utenti():
        p, d = istanza.pickup(i), istanza.delivery(i)
        grezzo = d.l - p.e - p.servizio - istanza.ride_max(i)
        if grezzo < -EPS:
            raise ValueError(
                f"{istanza.nome}: richiesta {i} infeasible: "
                f"l_i- - e_i+ - s_i+ - L_i = {grezzo:.6f} < 0. "
                "Controlla le finestre temporali o il preprocessing eq. (5)-(6)."
            )
        M[i] = max(0.0, grezzo)
    return M

def calcola_M_tilde(grafo: Grafo,
                        coda: dict[Nodo, int],
                        testa: dict[Nodo, int]) -> dict[Arco, float]:
    """
    Big-M dei vincoli (2b)/(9b), Sez. 3.3: M~ = l_v + s_v + t_(v,w) - e_w, il più piccolo
    valore che disattiva il vincolo ad arco spento (caso peggiore B_v = l_v, B_w = e_w).
    """
    istanza = grafo.istanza
    M: dict[Arco, float] = {}
    for arco in grafo.archi:
        v, w = arco
        nodo_v = istanza.nodo(coda[v])
        nodo_w = istanza.nodo(testa[w])
        # se è negativo il vincolo è già inattivo: un M negativo taglierebbe soluzioni ammissibili
        M[arco] = max(0.0, nodo_v.l + nodo_v.servizio + grafo.tempo(arco) - nodo_w.e)
    return M


def finestra_deposito(istanza: Istanza) -> tuple[float, float]:
    """
    Finestra di B_0: intersezione delle finestre dei due depositi del file. Nei Cordeau
    sono entrambe [0, T]; nelle istanze Trieste possono essere diverse.
    """
    iniziale, finale = istanza.deposito_iniziale(), istanza.deposito_finale()
    lb, ub = max(iniziale.e, finale.e), min(iniziale.l, finale.l)
    if lb > ub + EPS:
        raise ValueError(
            f"{istanza.nome}: le finestre dei due depositi non si intersecano: "
            f"[{iniziale.e}, {iniziale.l}] e [{finale.e}, {finale.l}]"
        )
    return lb, ub

def finestre_nodi(grafo: Grafo, coda: dict[Nodo, int]) -> dict[Nodo, tuple[float, float]]:
    """
    Bound [e_j, l_j] di ogni B_v, con j la località fisica di v: vincoli (2d) del Model I
    e parte di (9d)-(9f) del Model II, messi come lb/ub delle variabili e non come righe.
    """
    istanza = grafo.istanza
    deposito = grafo.deposito()
    lb_dep, ub_dep = finestra_deposito(istanza)

    finestre: dict[Nodo, tuple[float, float]] = {}
    for v in grafo.nodi:
        if v == deposito:
            finestre[v] = (lb_dep, ub_dep)
        else:
            nodo = istanza.nodo(coda[v])
            finestre[v] = (nodo.e, nodo.l)
    return finestre


# ----------------------------------------------------------------------
# Costruzione del modello
# ----------------------------------------------------------------------

def costruisci_modello(
        grafo: Grafo,
        *,
        variante: str = "II",
        consenti_rifiuti: bool = False,
        time_limit: float | None = None,
        mip_gap: float = 0.0,
        threads: int | None = None,
        log: bool = True,
) -> gp.Model:
    """
    Costruisce il MILP event-based con obiettivo f_c.

    variante "I": ride time con big-M, eq. (2e); ogni riga contiene due somme di archi.
    variante "II": ride time senza big-M, eq. (9g), reso innocuo sui nodi non usati
    dalle finestre (9e)/(9f). Le somme di archi restano solo in (9e)/(9f): matrice più
    rada e rilassamento lineare migliore.

    consenti_rifiuti=False usa (1c), ogni richiesta è servita (Tabelle 5 e 6).
    consenti_rifiuti=True usa (3) con le p_i e non imposta l'obiettivo, perché con f_c
    l'ottimo sarebbe non servire nessuno: il modello viene marcato con
    m._richiede_obiettivo e l'obiettivo va impostato con objectives.py.

    mip_gap è 0 di default (Gurobi usa 1e-4), per avere ottimi confrontabili con il paper.
    """
    if variante not in VARIANTI:
        raise ValueError(f"variante non valida: {variante!r} (attese {VARIANTI})")

    istanza = grafo.istanza

    # l'ordine di creazione delle variabili cambia il branch-and-bound: lo fissiamo,
    # così Model I e Model II si confrontano a parità di ordinamento
    nodi = sorted(grafo.nodi)
    archi = sorted(grafo.archi)
    deposito = grafo.deposito()

    coda, testa = mappa_localita(grafo)
    M_ride = calcola_M_ride(istanza)
    M_tilde = calcola_M_tilde(grafo, coda, testa)
    finestre = finestre_nodi(grafo, coda)
    costo = {a: grafo.costo(a) for a in archi}
    tempo = {a: grafo.tempo(a) for a in archi}

    m = gp.Model(f"DARP_event_{variante}_{istanza.nome}")
    m.Params.OutputFlag = 1 if log else 0
    m.Params.MIPGap = mip_gap
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if threads is not None:
        m.Params.Threads = threads

    # --- variabili ----------------------------------------------------
    # dizionari normali e non tupledict: le chiavi sono tuple di tuple, che gurobipy
    # interpreterebbe come indici multidimensionali
    x = {a: m.addVar(vtype=GRB.BINARY, name=f"x[{etichetta_arco(a)}]") for a in archi}

    B = {}
    for v in nodi:
        lb, ub = finestre[v]
        B[v] = m.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"B[{etichetta(v)}]")

    p = None
    if consenti_rifiuti:
        p = {i: m.addVar(vtype=GRB.BINARY, name=f"p[{i}]") for i in istanza.utenti()}

    m.update()

    # --- cache delle somme di flusso ----------------------------------
    # somma(x su delta_in(v)) serve in (1b), (1c), (2e), (9e), (9f): la calcoliamo una volta sola
    flusso_in = {v: gp.quicksum(x[a] for a in grafo.delta_in[v]) for v in nodi}
    flusso_out = {v: gp.quicksum(x[a] for a in grafo.delta_out[v]) for v in nodi}

    _vincoli_comuni(m, grafo, x, B, p, nodi, archi, deposito,
                    flusso_in, flusso_out, coda, tempo, M_tilde)

    if variante == "I":
        _vincoli_tempo_modello_I(m, grafo, B, flusso_in, M_ride)
    else:
        _vincoli_tempo_modello_II(m, grafo, B, flusso_in, M_ride)

    # --- obiettivo ----------------------------------------------------
    if consenti_rifiuti:
        # senza un obiettivo che penalizzi i rifiuti Gurobi minimizzerebbe 0 e darebbe
        # come ottimo la soluzione vuota: il flag impedisce di risolvere il modello così
        m._richiede_obiettivo = True
        m._obiettivo = None
        m._coefficienti = None
    else:
        m.setObjective(gp.quicksum(costo[a] * x[a] for a in archi), GRB.MINIMIZE)
        m._richiede_obiettivo = False
        # stesso formato di objectives.imposta_somma_pesata, così results.py tratta
        # allo stesso modo questo modello e quelli con obiettivo impostato
        m._obiettivo = "fc"
        m._coefficienti = {"costo": 1.0, "regret": 0.0,
                           "regret_max": 0.0, "rifiuti": 0.0}

    # --- stato appeso al modello, per objectives.py e results.py ------
    m._grafo = grafo
    m._istanza = istanza
    m._variante = variante
    m._x = x
    m._B = B
    m._p = p
    m._d = None
    m._dmax = None
    m._archi = archi
    m._deposito = deposito
    m._costo = costo
    m._tempo = tempo
    m._coda = coda
    m._testa = testa

    m.update()
    return m


def _vincoli_comuni(m, grafo, x, B, p, nodi, archi, deposito,
                    flusso_in, flusso_out, coda, tempo, M_tilde) -> None:
    """Vincoli (1b), (1c) o (3), (1d), (2b)/(9b), (2c)/(9c): identici nelle due varianti."""
    istanza = grafo.istanza

    # (1b) conservazione del flusso, deposito incluso
    for v in nodi:
        m.addConstr(flusso_in[v] - flusso_out[v] == 0, name=f"flusso[{etichetta(v)}]")

    # (1c) / (3) copertura: per ogni utente, esattamente uno dei suoi nodi di pickup
    # riceve un arco (nessuno, se p_i = 0)
    for i in istanza.utenti():
        servita = gp.quicksum(flusso_in[v] for v in grafo.nodi_pickup(i))
        if p is None:
            m.addConstr(servita == 1, name=f"copertura[{i}]")
        else:
            m.addConstr(servita == p[i], name=f"copertura[{i}]")

    # (1d) al piu' |K| dicicli passanti per il deposito
    m.addConstr(flusso_out[deposito] <= istanza.K, name="veicoli")

    # (2b)/(9b) propagazione dei tempi, archi non uscenti dal deposito.
    # (2c)/(9c) archi uscenti dal deposito: B_0 non compare, al suo posto e_0.
    e_0 = istanza.deposito_iniziale().e
    for a in archi:
        v, w = a
        nome = etichetta_arco(a)
        if v == deposito:
            m.addConstr(B[w] >= e_0 + tempo[a] * x[a], name=f"tempo_dep[{nome}]")
        else:
            s_v = istanza.nodo(coda[v]).servizio
            m.addConstr(
                B[w] >= B[v] + s_v + tempo[a] - M_tilde[a] * (1 - x[a]),
                name=f"tempo[{nome}]",
            )


def _vincoli_tempo_modello_I(m, grafo, B, flusso_in, M_ride) -> None:
    """
    (2e) ride time con big-M:  B_w - B_v - s_i+ <= L_i + M_i (2 - somma_in(v) - somma_in(w)).
    La parentesi vale 0 solo se entrambi i nodi sono usati; altrimenti il big-M
    disattiva il vincolo. Una riga per ogni coppia (pickup, drop-off) dello stesso utente.
    """
    istanza = grafo.istanza
    for i in istanza.utenti():
        s_i = istanza.pickup(i).servizio
        L_i = istanza.ride_max(i)
        M_i = M_ride[i]
        for v in grafo.nodi_pickup(i):
            for w in grafo.nodi_delivery(i):
                m.addConstr(
                    B[w] - B[v] - s_i <= L_i + M_i * (2 - flusso_in[v] - flusso_in[w]),
                    name=f"ride[{i},{etichetta(v)},{etichetta(w)}]",
                )


def _vincoli_tempo_modello_II(m, grafo, B, flusso_in, M_ride) -> None:
    """
    (9e), (9f), (9g): ride time senza big-M.

        (9g)  B_w - B_v - s_i+ <= L_i                       per ogni coppia (v, w)
        (9e)  B_v >= e_i+ + M_i (1 - somma_in(v))           nodi di pickup
        (9f)  B_w <= e_i+ + L_i + s_i+ + M_i somma_in(w)    nodi di drop-off

    (9g) vale anche sui nodi fantasma, quelli che nessuna rotta usa: (9e)/(9f) spingono
    i pickup fantasma in alto e i drop-off fantasma in basso, così (9g) è sempre
    soddisfatta:

        v usato,     w usato      -> è il vincolo di ride time vero
        v usato,     w fantasma   -> B_w <= e_i+ + L_i + s_i+, margine L_i
        v fantasma,  w usato      -> B_v >= l_i- - L_i - s_i+, margine L_i
        v fantasma,  w fantasma   -> vale se M_i >= 0 (garantito da calcola_M_ride)

    Su un nodo usato (9e)/(9f) danno i bound naturali e_i+ e l_i-, quindi non lo stringono.
    """
    istanza = grafo.istanza
    for i in istanza.utenti():
        pick = istanza.pickup(i)
        s_i, e_i_piu = pick.servizio, pick.e
        L_i = istanza.ride_max(i)
        M_i = M_ride[i]

        for v in grafo.nodi_pickup(i):
            m.addConstr(
                B[v] >= e_i_piu + M_i * (1 - flusso_in[v]),
                name=f"tw_pickup[{i},{etichetta(v)}]",
            )

        for w in grafo.nodi_delivery(i):
            m.addConstr(
                B[w] <= e_i_piu + L_i + s_i + M_i * flusso_in[w],
                name=f"tw_delivery[{i},{etichetta(w)}]",
            )

        for v in grafo.nodi_pickup(i):
            for w in grafo.nodi_delivery(i):
                m.addConstr(
                    B[w] - B[v] - s_i <= L_i,
                    name=f"ride[{i},{etichetta(v)},{etichetta(w)}]",
                )


# ----------------------------------------------------------------------
# Diagnostica
# ----------------------------------------------------------------------

def dimensioni(m: gp.Model) -> dict[str, int]:
    """Dimensioni del modello: variabili, binarie, vincoli, nonzeri."""
    m.update()
    return {
        "variabili": m.NumVars,
        "binarie": m.NumBinVars,
        "vincoli": m.NumConstrs,
        "nonzeri": m.NumNZs,
    }


if __name__ == "__main__":
    import sys
    import time

    from instances import leggi_istanza

    percorso = sys.argv[1] if len(sys.argv) > 1 else "dati_milp/a2-16.txt"

    istanza = leggi_istanza(percorso)
    grafo = Grafo.costruisci(istanza)
    print(f"{istanza.nome}: |V| = {len(grafo.nodi)}, |A| = {len(grafo.archi)}")

    M_ride = calcola_M_ride(istanza)
    print(f"M_i in [{min(M_ride.values()):.3f}, {max(M_ride.values()):.3f}] "
          f"(deve essere >= 0 su tutte le richieste)")

    risultati = {}
    for variante in VARIANTI:
        m = costruisci_modello(grafo, variante=variante, log=False)
        dim = dimensioni(m)
        print(f"\nModel {variante}: {dim['variabili']} variabili "
              f"({dim['binarie']} binarie), {dim['vincoli']} vincoli, "
              f"{dim['nonzeri']} nonzeri")

        t0 = time.time()
        m.optimize()
        wall = time.time() - t0

        if m.SolCount == 0:
            print(f"  nessuna soluzione intera (status {m.Status})")
            risultati[variante] = None
        else:
            print(f"  obiettivo = {m.ObjVal:.4f} | gap = {m.MIPGap:.2e} | "
                  f"Runtime = {m.Runtime:.2f}s | Work = {m.Work:.2f} | "
                  f"wall = {wall:.2f}s")
            risultati[variante] = m.ObjVal

    if risultati.get("I") is not None and risultati.get("II") is not None:
        scarto = abs(risultati["I"] - risultati["II"])
        print(f"\nScarto |Model I - Model II| = {scarto:.6f}")
        assert scarto < 1e-4, (
            "Model I e Model II danno ottimi diversi, ma sono equivalenti: "
            "c'è un errore nei vincoli."
        )
        print("OK: Model I e Model II concordano.")
