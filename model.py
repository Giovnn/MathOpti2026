"""
model.py — I due modelli MILP event-based per il DARP (Model I e Model II).

Riferimento: Gaul, Klamroth & Stiglmayr (2022), EJOR 301(3), Sez. 3.2-3.4.

Questo modulo costruisce la STRUTTURA del problema e nient'altro:
variabili x e B, conservazione del flusso, copertura delle richieste, numero di
veicoli, propagazione dei tempi, finestre temporali, ride time. L'obiettivo di
default e' il costo di routing f_c (eq. 10); le altre sei funzioni obiettivo
del paper vivono in objectives.py, la risoluzione e l'estrazione dei risultati
in results.py.

Cosa NON c'e' qui, e perche':
  - capacita' del veicolo      -> nella definizione dei nodi (Q-tuple ammissibili)
  - pairing e precedenza       -> nella struttura degli archi A1-A6
  - coppie incompatibili       -> preprocessing f1/f2 in graph.py
  - eliminazione dei sottotour -> implicita nei vincoli temporali (2b)/(9b):
        sommando B_w >= B_v + s + t lungo un ciclo si ottiene 0 >= somma(s+t),
        impossibile appena un tempo di servizio o di viaggio e' positivo.
        ATTENZIONE: nell'Esempio 1 del paper s = t = 0, quindi quell'istanza
        NON e' utilizzabile per validare il MILP (lo e' solo per il grafo).
  - durata massima T del servizio -> e' il bound superiore di B_0 (vedi sotto).

Il deposito e' un unico nodo (0,...,0), quindi esiste una sola variabile B_0.
Il paper esclude gli archi USCENTI dal deposito da (2b)/(9b) e li sostituisce
con (2c)/(9c), che usano la costante e_0 al posto di B_0. Gli archi ENTRANTI
restano in (2b)/(9b): di conseguenza B_0 e' l'istante di rientro dell'ultimo
veicolo (il makespan) e il suo bound superiore impone la durata massima del
servizio. Non serve un vincolo separato per T.
"""

import gurobipy as gp
from gurobipy import GRB

from instances import Istanza
from graph import Grafo, Nodo, Arco, localita_id

# Tolleranza sui confronti float. I dati Cordeau cadono spesso esattamente sui
# bordi delle finestre, quindi i test di segno vanno fatti con un margine.
EPS = 1e-9

VARIANTI = ("I", "II")


# ----------------------------------------------------------------------
# Etichette leggibili per variabili e vincoli
# ----------------------------------------------------------------------

def etichetta(nodo: Nodo) -> str:
    """
    Nome ASCII di un nodo, utilizzabile dentro i nomi delle variabili Gurobi.

    Il segno meno non e' un carattere sicuro nel formato LP, quindi le
    componenti negative diventano 'm': (-2, 1, 0) -> 'm2_1_0'.
    il formato LP è un formato testuale per la rappresentazione dei modelli di programmazione lineare,
    e alcuni caratteri speciali possono causare problemi durante la lettura del file da parte di Gurobi o altri solver.
    Per evitare questi problemi, le componenti negative dei nodi vengono sostituite con 'm' (per "minus") nel nome della variabile.
    Ad esempio, un nodo con coordinate (-2, 1, 0) sarà rappresentato come 'm2_1_0',
    garantendo che il nome sia sicuro e leggibile all'interno del modello.
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
    Per ogni nodo evento, l'id della localita' fisica associata, in due versioni.

    Il deposito e' l'unico nodo ambiguo: nel formato Cordeau ha due id fisici
    (0 iniziale, 2n+1 finale). `coda` va usata quando il nodo e' l'origine di un
    arco, `testa` quando ne e' la destinazione. Per tutti gli altri nodi le due
    mappe coincidono.
    """
    istanza = grafo.istanza
    coda = {v: localita_id(v, istanza, is_partenza=True) for v in grafo.nodi}
    testa = {v: localita_id(v, istanza, is_partenza=False) for v in grafo.nodi}
    return coda, testa


def calcola_M_ride(istanza: Istanza) -> dict[int, float]:
    """
    M_i = l_i- - e_i+ - s_i+ - L_i, troncato a zero.  (paper, Sez. 3.3)

    E' il big-M dei vincoli di ride time del Model I (2e) ED e' il coefficiente
    delle finestre riformulate del Model II (9e)/(9f): il paper lo chiama TW nel
    secondo caso, ma le due quantita' coincidono quando le finestre mancanti
    sono generate dalle eq. (5)-(6). Calcolarlo dalle finestre effettive invece
    di leggere TW rende la formula valida anche quando le finestre sono state
    strette da un preprocessing (sulle istanze b4-40 e b8-80 il taglio
    sull'orizzonte del deposito rende M_i < TW: e' corretto e piu' stretto).

    M_i < 0 significa che la richiesta i e' infeasible in se': la finestra di
    drop-off si chiude prima di quanto il ride time massimo consenta.
    """
    M: dict[int, float] = {}
    for i in istanza.utenti():
        p, d = istanza.pickup(i), istanza.delivery(i)
        grezzo = d.l - p.e - p.servizio - istanza.L
        if grezzo < -EPS:
            raise ValueError(
                f"{istanza.nome}: richiesta {i} infeasible: "
                f"l_i- - e_i+ - s_i+ - L_i = {grezzo:.6f} < 0. "
                "Controlla le finestre temporali o il preprocessing eq. (5)-(6)."
            )
        M[i] = max(0.0, grezzo)
    return M


def calcola_M_tilde(grafo: Grafo,
                    coda: dict[Nodo, int] | None = None,
                    testa: dict[Nodo, int] | None = None) -> dict[Arco, float]:
    """
    M~_(v,w) = l_v1 + s_v1 + t_(v,w) - e_w1, troncato a zero.  (paper, Sez. 3.3)

    E' il piu' piccolo valore che rende (2b) ridondante quando l'arco non e'
    usato: caso peggiore B_v = l_v1 (massimo) e B_w = e_w1 (minimo). Il piu'
    piccolo M valido da' il rilassamento lineare piu' stretto.

    Il troncamento non e' cosmetico: se l_v1 + s + t <= e_w1 il vincolo e'
    sempre soddisfatto e un M~ negativo lo renderebbe attivo anche ad arco
    spento, tagliando soluzioni ammissibili.
    """
    istanza = grafo.istanza
    if coda is None or testa is None:
        coda, testa = mappa_localita(grafo)

    M: dict[Arco, float] = {}
    for arco in grafo.archi:
        v, w = arco
        nodo_v = istanza.nodo(coda[v])
        nodo_w = istanza.nodo(testa[w])
        M[arco] = max(0.0, nodo_v.l + nodo_v.servizio + grafo.tempo(arco) - nodo_w.e)
    return M


def finestra_deposito(istanza: Istanza) -> tuple[float, float]:
    """
    Finestra della singola variabile B_0.

    B_0 e' contemporaneamente il nodo sorgente e il nodo pozzo del grafo, mentre
    nel file Cordeau i due depositi sono nodi distinti. Il bound corretto e'
    l'intersezione delle due finestre: nei benchmark sono entrambe [0, T] e
    l'intersezione non cambia nulla, ma per le istanze Trieste (deposito con
    orario di apertura) non e' detto.
    """
    iniziale, finale = istanza.deposito_iniziale(), istanza.deposito_finale()
    lb, ub = max(iniziale.e, finale.e), min(iniziale.l, finale.l)
    if lb > ub + EPS:
        raise ValueError(
            f"{istanza.nome}: le finestre dei due depositi non si intersecano: "
            f"[{iniziale.e}, {iniziale.l}] e [{finale.e}, {finale.l}]"
        )
    return lb, ub


def finestre_nodi(grafo: Grafo, coda: dict[Nodo, int] | None = None) -> dict[Nodo, tuple[float, float]]:
    """
    Bound [e_j, l_j] di ogni variabile B_v, con j la localita' fisica di v.
    "Il servizio deve iniziare tra e_j e l_j" (paper, Sez. 3.4).
    Corrisponde a (2d) per il Model I e alla parte "bound" di (9d)-(9f) per il
    Model II: le finestre non sono righe della matrice, sono lb/ub delle
    variabili. Le due varianti condividono questi bound; il Model II aggiunge
    poi le righe (9e)/(9f) che stringono il lato libero sui nodi inattivi.
    """

    istanza = grafo.istanza

    if coda is None:
        coda, _ = mappa_localita(grafo)  # se la mappa non e' stata passata, la calcoliamo qui

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
        grado_entrante_unitario: bool = False,
        time_limit: float | None = None,
        mip_gap: float = 0.0,
        threads: int | None = None,
        log: bool = True,
        nome: str | None = None,
) -> gp.Model:
    """
    Costruisce il MILP event-based, variante "I" o "II", con obiettivo f_c.

    variante:
        "I"  -> ride time linearizzati con big-M, eq. (2e). Famiglia O(n^(2Q-1))
                di vincoli, ciascuno contenente DUE somme di archi.
        "II" -> ride time in forma pulita, eq. (9g), resi innocui sui nodi
                inattivi dalle finestre riformulate (9e)/(9f). Stesso numero di
                righe, ma le somme di archi restano solo nella famiglia O(n^Q)
                delle finestre: matrice molto piu' rada e meno big-M, quindi
                rilassamento lineare migliore.

    consenti_rifiuti:
        False -> (1c): ogni richiesta e' servita. E' il caso delle Tabelle 5 e 6
                 del paper. Il modello riceve l'obiettivo f_c ed e' risolvibile
                 cosi' com'e'.
        True  -> (3): introduce p_i binaria, la richiesta i e' servita se
                 p_i = 1. In questo caso il modello NON riceve alcun obiettivo e
                 viene marcato con m._richiede_obiettivo = True: con f_c
                 l'ottimo sarebbe la soluzione vuota a costo zero. Sta a
                 objectives.py impostare un obiettivo che penalizzi i rifiuti
                 (f_rcr) e azzerare il flag; results.py rifiuta di risolvere un
                 modello con il flag ancora alzato.

    grado_entrante_unitario:
        Aggiunge somma(x su delta_in(v)) <= 1 per ogni nodo diverso dal
        deposito. E' una disuguaglianza valida: non cambia l'insieme delle
        soluzioni intere (quindi i valori ottimi restano confrontabili con le
        tabelle pubblicate), ma stringe il rilassamento lineare. Disattivata di
        default, cosi' il modello di riferimento resta quello del paper.

    mip_gap:
        0.0 di default, NON il default di Gurobi (1e-4 relativo). Con il gap di
        Gurobi il solver puo' fermarsi sopra l'ottimo vero e dichiarare
        "risolto": i valori non sarebbero confrontabili con le Tabelle 5 e 6.
    """
    if variante not in VARIANTI:
        raise ValueError(f"variante non valida: {variante!r} (attese {VARIANTI})")

    istanza = grafo.istanza

    # Ordinamento deterministico: l'ordine di creazione delle variabili cambia
    # il percorso di branch & bound, e quindi i tempi. Senza questo, confrontare
    # Model I e Model II sarebbe confrontare anche due ordinamenti diversi.
    nodi = sorted(grafo.nodi)
    archi = sorted(grafo.archi)
    deposito = grafo.deposito()

    coda, testa = mappa_localita(grafo)
    M_ride = calcola_M_ride(istanza)
    M_tilde = calcola_M_tilde(grafo, coda, testa)
    finestre = finestre_nodi(grafo, coda)
    costo = {a: grafo.costo(a) for a in archi}
    tempo = {a: grafo.tempo(a) for a in archi}

    m = gp.Model(nome or f"DARP_event_{variante}_{istanza.nome}")
    m.Params.OutputFlag = 1 if log else 0
    m.Params.MIPGap = mip_gap
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if threads is not None:
        m.Params.Threads = threads

    # --- variabili ----------------------------------------------------
    # Dizionari semplici, non tupledict: le chiavi sono tuple annidate
    # (un arco e' una coppia di Q-tuple) e l'indicizzazione multidimensionale
    # di gurobipy su chiavi del genere e' ambigua.
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
    # somma(x su delta_in(v)) compare in (1b), (1c), (2e), (9e), (9f).
    # Ricostruirla dentro i cicli e' il collo di bottiglia della costruzione.
    flusso_in = {v: gp.quicksum(x[a] for a in grafo.delta_in[v]) for v in nodi}
    flusso_out = {v: gp.quicksum(x[a] for a in grafo.delta_out[v]) for v in nodi}

    _vincoli_comuni(m, grafo, x, B, p, nodi, archi, deposito,
                    flusso_in, flusso_out, coda, tempo, M_tilde)

    if variante == "I":
        _vincoli_tempo_modello_I(m, grafo, B, flusso_in, M_ride)
    else:
        _vincoli_tempo_modello_II(m, grafo, B, flusso_in, M_ride)

    if grado_entrante_unitario:
        for v in nodi:
            if v != deposito:
                m.addConstr(flusso_in[v] <= 1, name=f"grado_in[{etichetta(v)}]")

    # --- obiettivo ----------------------------------------------------
    if consenti_rifiuti:
        # Senza penalizzazione dei rifiuti l'ottimo e' la soluzione vuota.
        # Gurobi non protesta se manca l'obiettivo: assume "minimize 0" e
        # restituisce OPTIMAL con valore 0. Il flag serve esattamente a
        # impedire che quel risultato passi per buono.
        m._richiede_obiettivo = True
        m._obiettivo = None
        m._coefficienti = None
    else:
        m.setObjective(gp.quicksum(costo[a] * x[a] for a in archi), GRB.MINIMIZE)
        m._richiede_obiettivo = False
        # Stesso formato di objectives.imposta_somma_pesata: tutti e quattro i
        # criteri, quelli assenti con peso zero. Cosi' results.py tratta allo
        # stesso modo un modello di default e uno passato da objectives.py.
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
    m._livelli = []       # vincoli di livello (criterio, vincolo), vedi objectives.py
    m._nodi = nodi
    m._archi = archi
    m._deposito = deposito
    m._flusso_in = flusso_in
    m._flusso_out = flusso_out
    m._costo = costo
    m._tempo = tempo
    m._coda = coda
    m._testa = testa
    m._M_ride = M_ride
    m._M_tilde = M_tilde

    m.update()
    return m


def _vincoli_comuni(m, grafo, x, B, p, nodi, archi, deposito,
                    flusso_in, flusso_out, coda, tempo, M_tilde) -> None:
    """Vincoli (1b), (1c) o (3), (1d), (2b)/(9b), (2c)/(9c): identici nelle due varianti."""
    istanza = grafo.istanza

    # (1b) conservazione del flusso, deposito incluso
    for v in nodi:
        m.addConstr(flusso_in[v] - flusso_out[v] == 0, name=f"flusso[{etichetta(v)}]")

    # (1c) / (3) copertura delle richieste: esattamente un nodo di pickup di i
    # e' raggiunto, e lo e' una volta sola. La somma e' doppia: su tutti i nodi
    # di pickup di i, e per ciascuno su tutti i suoi archi entranti.
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
    (2e) ride time linearizzato con big-M.

        B_w - B_v - s_i+ <= L_i + M_i (2 - somma_in(v) - somma_in(w))

    La parentesi vale 0 se entrambi i nodi sono attivi (vincolo vero), 1 se lo e'
    uno solo, 2 se nessuno: in quei casi il big-M lo disattiva. Una riga per ogni
    coppia (nodo di pickup, nodo di drop-off) dello stesso utente, cioe'
    O(n^(2Q-1)) righe, ciascuna con due somme di archi dentro.
    """
    istanza = grafo.istanza
    L = istanza.L
    for i in istanza.utenti():
        s_i = istanza.pickup(i).servizio
        M_i = M_ride[i]
        for v in grafo.nodi_pickup(i):
            for w in grafo.nodi_delivery(i):
                m.addConstr(
                    B[w] - B[v] - s_i <= L + M_i * (2 - flusso_in[v] - flusso_in[w]),
                    name=f"ride[{i},{etichetta(v)},{etichetta(w)}]",
                )


def _vincoli_tempo_modello_II(m, grafo, B, flusso_in, M_ride) -> None:
    """
    (9e), (9f), (9g): ride time senza meccanismo di attivazione.

        (9g)  B_w - B_v - s_i+ <= L_i              su TUTTE le coppie
        (9e)  B_v >= e_i+ + M_i (1 - somma_in(v))  sui nodi di pickup
        (9f)  B_v <= e_i+ + L_i + s_i+ + M_i somma_in(v)   sui nodi di drop-off

    (9g) e' imposta anche sui "nodi fantasma", quelli che nessuna rotta usa.
    Sono le (9e)/(9f) a renderla innocua: spingono i pickup inattivi in alto e i
    drop-off inattivi in basso, quel tanto che basta perche' (9g) risulti
    automaticamente soddisfatta. I quattro casi:

        v attivo,   w attivo   -> vincolo vero, e' quello che vogliamo
        v attivo,   w inattivo -> B_w <= e_i+ + L_i + s_i+  => margine esatto L_i
        v inattivo, w attivo   -> B_v >= l_i- - L_i - s_i+  => margine esatto L_i
        v inattivo, w inattivo -> vale sse l_i- >= e_i+ + s_i+ + L_i, cioe' M_i >= 0

    Tutta la correttezza poggia su M_i >= 0, garantito da calcola_M_ride.

    Nota sui coefficienti: quando il nodo di drop-off e' attivo, (9f) da'
    e_i+ + L_i + s_i+ + M_i = l_i-, esattamente il bound naturale. Le righe
    (9e)/(9f) non stringono mai un nodo attivo: agiscono solo sui fantasmi.
    """
    istanza = grafo.istanza
    L = istanza.L
    for i in istanza.utenti():
        pick = istanza.pickup(i)
        s_i, e_i_piu = pick.servizio, pick.e
        M_i = M_ride[i]

        for v in grafo.nodi_pickup(i):
            m.addConstr(
                B[v] >= e_i_piu + M_i * (1 - flusso_in[v]),
                name=f"tw_pickup[{i},{etichetta(v)}]",
            )

        for w in grafo.nodi_delivery(i):
            m.addConstr(
                B[w] <= e_i_piu + L + s_i + M_i * flusso_in[w],
                name=f"tw_delivery[{i},{etichetta(w)}]",
            )

        for v in grafo.nodi_pickup(i):
            for w in grafo.nodi_delivery(i):
                m.addConstr(
                    B[w] - B[v] - s_i <= L,
                    name=f"ride[{i},{etichetta(v)},{etichetta(w)}]",
                )


# ----------------------------------------------------------------------
# Diagnostica
# ----------------------------------------------------------------------

def dimensioni(m: gp.Model) -> dict[str, int]:
    """Dimensioni della matrice, per le tabelle di confronto della relazione."""
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

    percorso = sys.argv[1] if len(sys.argv) > 1 else "a2-16.txt"

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
            "I due modelli danno ottimi diversi: sono equivalenti per "
            "costruzione, quindi c'e' un errore. Sospetti nell'ordine: "
            "M_tilde troppo piccolo, M_i troppo piccolo, finestre dei nodi "
            "fantasma nel Model II, is_partenza invertito nei precalcoli."
        )
        print("OK: Model I e Model II concordano.")
