"""
objectives.py — Le sette funzioni obiettivo del paper come somme pesate di quattro criteri.

Riferimento: Gaul, Klamroth & Stiglmayr (2022), EJOR 301(3), Sez. 3.5 (pagg. 1055-1056)
e Sez. 4.2 (pagg. 1058-1061).

L'idea del modulo
-----------------
Il paper definisce quattro criteri elementari e li combina in sette funzioni obiettivo.
Ogni funzione obiettivo e' quindi un VETTORE DI QUATTRO PESI applicato agli stessi
quattro criteri:

    criterio     simbolo  formula              da dove vengono le variabili
    costo        f_c      somma_a c_a x_a       eq. (10), x gia' in model.py
    rifiuti      f_n      n - somma_i p_i       Sez. 3.5, p gia' in model.py (se rifiuti ammessi)
    regret       f_r      somma_i d_i           eq. (13), d_i e vincoli (11) creati QUI
    regret_max   f_rmax   d_max                 eq. (14), d_max e vincoli (12) creati QUI

    obiettivo   costo  regret  regret_max  rifiuti   eq.    tabelle del paper
    fc            1      .         .          .      (10)   5, 6, 8
    fn            .      .         .          1       -     nessuna
    fr            .      1         .          .      (13)   8
    frmax         .      .         1          .      (14)   10
    fcr           1    alpha       .          .      (15)   9
    fcrmax        1      .       beta         .      (16)   10
    frcr          1    alpha       .        gamma    (17)   9

Una sola funzione generica (imposta_somma_pesata) costruisce qualunque combinazione; i
sette obiettivi del paper sono "preset" (vedi coefficienti). Un solo percorso di codice
da testare, e gli sweep sui pesi o l'epsilon-constraint non richiedono codice nuovo.

Pesi del paper: alpha = 1, beta = n/5, gamma = 20 (pag. 1060). Sono tarati su istanze in
cui il costo e' in km e il tempo in minuti (t = 4c, 15 km/h): sui benchmark Cordeau, dove
costo = tempo, lo stesso peso ha un significato diverso. Per questo sono configurabili.

Cosa c'e' qui e cosa no
-----------------------
  qui:        variabili e vincoli dei criteri (11)-(12), l'obiettivo, i vincoli di
              livello, e le DEFINIZIONI dei criteri valutati a posteriori (funzioni pure:
              ricevono numeri, restituiscono numeri, non toccano Gurobi).
  results.py: optimize(), lettura dei .X, ricostruzione delle rotte, orchestrazione del
              lessicografico e dell'epsilon-constraint.
  graph.py:   schedule minimo (Bellman-Ford), dopo il refactor di _esiste_schedule.

Due avvertenze per chi legge i risultati
----------------------------------------
  1. Le d_i sono il regret vero solo se l'obiettivo le minimizza TUTTE (fr, fcr, frcr).
     Con fc e fn non esistono nemmeno; con frmax e fcrmax viene spinto in basso solo il
     massimo. In quei casi i d_i.X e i B.X non sono unici (dipendono dal solver): i
     criteri si valutano sullo schedule minimo delle rotte trovate (colonna "canonica");
     i valori letti dal solver formano la colonna "grezza".
  2. Gli obiettivi "puri" (fr, frmax, fn) hanno in generale molti ottimi: il costo della
     soluzione restituita e' arbitrario. Per un costo riproducibile si usa il
     lessicografico in due fasi (vedi aggiungi_vincolo_livello).
"""

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from graph import Grafo
from instances import Istanza
from model import costruisci_modello, etichetta

# Tolleranza per i controlli su tempi gia' passati per il solver. E' piu' larga dell'EPS
# di model.py (1e-9), che lavora sui dati di input: qui i numeri arrivano da Gurobi e
# portano con se' le sue tolleranze (dell'ordine di 1e-6).
EPS = 1e-6

CRITERI = ("costo", "regret", "regret_max", "rifiuti")
OBIETTIVI = ("fc", "fn", "fr", "frmax", "fcr", "fcrmax", "frcr")


# ----------------------------------------------------------------------
# Blocco 1 — Pesi
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Pesi:
    """
    I tre parametri delle somme pesate (15)-(17).

        alpha  peso del regret totale in fcr e frcr
        beta   peso del regret massimo in fcrmax
        gamma  prezzo di una richiesta rifiutata in frcr, in unita' di costo

    frozen=True rende l'oggetto immutabile: nessuna funzione puo' cambiare per sbaglio i
    pesi di un'altra. Per una variante si crea una copia con un campo diverso:
        from dataclasses import replace
        replace(pesi_paper(16), gamma=40.0)   ->  alpha=1, beta=3.2, gamma=40
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 20.0


def pesi_paper(n: int) -> Pesi:
    """
    alpha = 1, beta = n/5, gamma = 20 (Sez. 4.2, pag. 1060).

    E' una funzione e non una costante perche' beta dipende dal numero di richieste: il
    costo cresce circa linearmente con n, d_max no (e' un massimo), e beta = n/5 tiene
    stabile il rapporto fra i due termini di fcrmax.
    """
    return Pesi(alpha=1.0, beta=n / 5, gamma=20.0)


# ----------------------------------------------------------------------
# Blocco 2 — Catalogo degli obiettivi del paper
# ----------------------------------------------------------------------

def coefficienti(nome: str, pesi: Pesi) -> dict[str, float]:
    """
    Il preset di un obiettivo del paper: dizionario criterio -> peso.
    I criteri assenti dal dizionario hanno peso zero.
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
    L'obiettivo ha bisogno delle variabili p_i e del vincolo (3) al posto di (1c)?
    Nel paper solo frcr (e fn, che il paper definisce ma non usa in nessuna tabella).
    I pesi passati sono irrilevanti: conta solo quali criteri compaiono nel preset.
    """
    return "rifiuti" in coefficienti(nome, Pesi())


# ----------------------------------------------------------------------
# Blocco 3 — Variabili ausiliarie dei criteri di regret (idempotenti)
# ----------------------------------------------------------------------

def _bound_regret(istanza: Istanza, i: int) -> float:
    """
    Bound superiore valido di d_i: l_i- - e_i-.

    Ogni B_v con v nodo di drop-off di i ha bound superiore l_i- (in entrambi i modelli),
    quindi B_v - e_i- <= l_i- - e_i-. Il bound non taglia nessuna soluzione e aiuta il
    presolve. Interpretazione:
        richiesta inbound:  TW + L_i - t_i   (a2-16, utente 16: 15 + 30 - 19.84 = 25.16)
        richiesta outbound: TW               (a2-16, utenti 1-8: 15)
    """
    drop = istanza.delivery(i)
    return drop.l - drop.e


def aggiungi_regret(m: gp.Model) -> dict[int, gp.Var]:
    """
    Variabili d_i >= 0 e vincoli (11):   d_i >= B_v - e_i-   per ogni i e ogni v in V_i-.

    Perche' su TUTTI i nodi di drop-off di i e senza big-M: non sappiamo quale nodo sara'
    attivo (lo decidono le x), quindi si scrive un vincolo per ciascuno. Insieme dicono
    d_i >= max_v (B_v - e_i-). Un drop-off fantasma puo' sempre scendere a B_v = e_i-
    (nessun vincolo lo spinge in alto, analisi_funzioni_obiettivo.md, par. 1.3 passo 2),
    e allora contribuisce 0: quando d_i e' minimizzata, all'ottimo d_i e' il regret del
    nodo attivo.

    Un utente rifiutato (p_i = 0) ha tutti i nodi fantasma, quindi d_i = 0.

    Idempotente: se le d_i esistono gia', le restituisce senza aggiungere nulla.
    """
    if m._d is not None:
        return m._d

    istanza, grafo, B = m._istanza, m._grafo, m._B

    d: dict[int, gp.Var] = {}
    for i in istanza.utenti():
        e_meno = istanza.delivery(i).e
        d[i] = m.addVar(lb=0.0, ub=_bound_regret(istanza, i),
                        vtype=GRB.CONTINUOUS, name=f"d[{i}]")
        # sorted(...): ordine di creazione deterministico, come in model.py
        for v in sorted(grafo.nodi_delivery(i)):
            m.addConstr(d[i] >= B[v] - e_meno, name=f"regret[{i},{etichetta(v)}]")

    m._d = d
    return d


def aggiungi_regret_max(m: gp.Model) -> gp.Var:
    """
    Variabile d_max >= 0 e vincoli (12):   d_max >= d_i   per ogni i.

    Richiede le d_i, e le crea se mancano: (12) lega d_max alle d_i, e sono le d_i a
    essere legate ai tempi B tramite (11).

    Attenzione nell'interpretare i risultati: minimizzando d_max viene spinto in basso
    solo l'utente peggiore; per tutti gli altri d_i e' libera fra il regret vero e d_max.
    Con frmax e fcrmax i valori d_i.X NON sono regret.

    Idempotente come aggiungi_regret.
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
# Blocco 4 — Le espressioni dei quattro criteri
# ----------------------------------------------------------------------

def espressione(m: gp.Model, criterio: str):
    """
    L'espressione Gurobi di un criterio. Le variabili ausiliarie nascono qui, e solo se
    servono: chiedere "regret" crea le d_i, chiedere "costo" no.

    Il criterio "rifiuti" contiene la costante n: il valore dell'obiettivo di frcr la
    include, come la colonna Obj.v. della Tabella 9.
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
# Blocco 5 — Vincoli di livello (lessicografico, epsilon-constraint)
# ----------------------------------------------------------------------

def _vincoli_livello(m: gp.Model) -> list:
    """
    Registro dei vincoli di livello presenti nel modello: lista di coppie
    (criterio, vincolo). Creato al primo uso.
    """
    try:
        return m._livelli
    except AttributeError:
        m._livelli = []
        return m._livelli


def soglia_con_tolleranza(valore: float, rel: float = 1e-6) -> float:
    """
    Soglia per la seconda fase del lessicografico.

    Chiedere criterio <= valore ESATTO puo' rendere il modello numericamente infeasible:
    valore e' un float restituito dal solver, affetto dalle sue tolleranze. Si concede un
    margine relativo, mai sotto rel in assoluto (conta per valori vicini a zero, per
    esempio f_r* = 0 su a5-50).
        soglia_con_tolleranza(14.1864) -> 14.1864 + 1e-6 * 14.1864 = 14.18641...
        soglia_con_tolleranza(0.0)     -> 0.0 + 1e-6 * 1 = 1e-6
    """
    return valore + rel * max(1.0, abs(valore))


def aggiungi_vincolo_livello(m: gp.Model, criterio: str, soglia: float,
                             *, nome: str | None = None) -> gp.Constr:
    """
    Aggiunge il vincolo  criterio <= soglia  e lo restituisce.

    Due usi, entrambi orchestrati da results.py:
      - lessicografico, seconda fase: risolto fr con valore f_r*, si aggiunge
            aggiungi_vincolo_livello(m, "regret", soglia_con_tolleranza(f_r*))
        e si reimposta l'obiettivo "fc": si ottiene la soluzione piu' economica fra
        quelle migliori per gli utenti (su a2-16: costo 317.946);
      - epsilon-constraint: regret <= epsilon con obiettivo "fc", epsilon via via piu'
        basso; trova anche i punti di Pareto che nessuna somma pesata raggiunge.

    Il vincolo resta nel modello finche' non lo si toglie con rimuovi_vincolo_livello.
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
# Blocco 6 — Impostare l'obiettivo
# ----------------------------------------------------------------------

def imposta_somma_pesata(m: gp.Model, *, costo: float = 0.0, regret: float = 0.0,
                         regret_max: float = 0.0, rifiuti: float = 0.0,
                         nome: str = "personalizzato") -> None:
    """
    Imposta come obiettivo la somma pesata dei quattro criteri (minimizzazione).

    E' l'unico punto del progetto che imposta obiettivi diversi da fc, ed e' l'unico che
    abbassa il flag m._richiede_obiettivo alzato da model.py: qui abbiamo verificato che,
    se il modello ha le p_i, qualcosa impedisce la soluzione vuota.

    Validazioni:
      - pesi negativi: vietati (un peso negativo premierebbe il regret o i rifiuti);
      - tutti i pesi nulli: vietato (l'obiettivo sarebbe "minimizza 0");
      - modello con p_i, peso "rifiuti" nullo e nessun vincolo di livello sui rifiuti:
        vietato, perche' l'ottimo sarebbe p = 0, x = 0 (nessuno servito, costo zero);
      - modello senza p_i e peso "rifiuti" positivo: vietato, il criterio non esiste.
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
    Imposta uno dei sette obiettivi del paper. Senza pesi espliciti usa quelli del
    paper, calcolati per l'istanza del modello (beta = n/5).
    """
    if pesi is None:
        pesi = pesi_paper(m._istanza.n)
    imposta_somma_pesata(m, nome=nome, **coefficienti(nome, pesi))


def costruisci_con_obiettivo(grafo: Grafo, nome: str, *, pesi: Pesi | None = None,
                             **opzioni_modello) -> gp.Model:
    """
    Costruisce il modello di model.py e gli imposta l'obiettivo `nome`.

    consenti_rifiuti non si passa: lo decide l'obiettivo (richiede_rifiuti). Tutte le
    altre opzioni di costruisci_modello (variante, time_limit, mip_gap, threads, log,
    ...) vengono inoltrate tali e quali.
    """
    if "consenti_rifiuti" in opzioni_modello:
        raise TypeError("consenti_rifiuti e' deciso dall'obiettivo, non va passato")
    m = costruisci_modello(grafo, consenti_rifiuti=richiede_rifiuti(nome), **opzioni_modello)
    imposta_obiettivo(m, nome, pesi)
    return m


# ----------------------------------------------------------------------
# Blocco 7 — Valutazione a posteriori dei criteri (funzioni pure, senza Gurobi)
# ----------------------------------------------------------------------

def regret_utente(istanza: Istanza, i: int, inizio_dropoff: float) -> float:
    """
    Regret dell'utente i: di quanto l'inizio del servizio al suo drop-off supera il primo
    istante ammesso e_i-.

    Un valore negativo oltre la tolleranza vuol dire orari incoerenti con le finestre:
    meglio fermarsi che riportare un numero sbagliato. I negativi minuscoli (-1e-12,
    arrotondamenti del solver) vengono riportati a zero.
    """
    r = inizio_dropoff - istanza.delivery(i).e
    if r < -EPS:
        raise ValueError(f"utente {i}: drop-off a {inizio_dropoff:.6f}, prima di "
                         f"e_i- = {istanza.delivery(i).e:.6f}: orari incoerenti")
    return max(0.0, r)


def valuta_criteri(istanza: Istanza, costo_rotte: float,
                   arrivi: dict[int, float]) -> dict[str, float]:
    """
    Valori dei quattro criteri (piu' due indicatori per le tabelle) di una soluzione.

    arrivi: {utente servito: inizio del servizio al suo drop-off}. Gli utenti assenti
    dal dizionario sono rifiutati.

    La funzione non sa da dove vengono gli orari: dallo schedule minimo delle rotte
    (colonna "canonica") o dai B.X restituiti dal solver (colonna "grezza"). E'
    results.py a scegliere la fonte e a etichettare il risultato.

    Chiavi restituite:
        costo, rifiuti, regret, regret_max  -> gli stessi nomi di CRITERI
        ar                                  -> % di richieste servite (colonna a.r.)
        regret_medio_servito                -> regret / serviti; con frcr il regret
                                               totale cala anche solo perche' i
                                               rifiutati non contano
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
    Ricalcola il valore di un obiettivo dai criteri misurati. coeff e' un dizionario
    criterio -> peso (per esempio m._coefficienti). Se coincide con m.ObjVal, l'obiettivo
    nel modello e' esattamente quello dichiarato.
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
        Criteri calcolati dai valori restituiti dal solver (colonna "grezza").
        Solo per questa verifica rapida: nel progetto la lettura dei risultati e' in
        results.py.
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

            # T5: l'obiettivo nel modello e' quello dichiarato. Per ciascuno dei sette
            # obiettivi il criterio che compare con peso positivo e' ottimizzato,
            # quindi i valori grezzi bastano (il costo e i rifiuti non dipendono dagli
            # orari; regret e regret_max, quando sono nell'obiettivo, sono spinti al
            # minimo anche sul nodo attivo).
            criteri = criteri_grezzi(m)
            ricalcolato = valore_obiettivo(m._coefficienti, criteri)
            if abs(ricalcolato - m.ObjVal) > 1e-4 * max(1.0, abs(m.ObjVal)):
                controlli.append(f"{variante}: T5 FALLITO ({ricalcolato:.4f})")
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