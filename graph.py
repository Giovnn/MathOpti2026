"""
Costruzione del grafo event-based G = (V, A) per il DARP
(Gaul, Klamroth & Stiglmayr 2022, Sez. 3.1).

Invece di un grafo geografico, ogni nodo qui è uno stato di occupazione del
veicolo (una Q-tupla), con la prima componente che indica l'evento più recente.

Codifica di un nodo (v1, v2, ..., vQ):
    v1 = +i   -> utente i appena caricato (pickup, "i+")
    v1 = -i   -> utente i appena scaricato (delivery, "i-")
    vj = 0    -> posto libero (j >= 2, zero-padding in coda)
    vj = k>0  -> utente k seduto a bordo (j >= 2)
    (0,0,...,0) -> deposito

Forma canonica (par. 2.2), per evitare che permutazioni della stessa
allocazione diano nodi duplicati: v1 è l'ultimo evento (con segno), v2..vQ
sono i compagni di viaggio in ordine decrescente, zeri in coda.

Il modulo dipende solo dall'interfaccia pubblica di Istanza: non conosce il
formato dei file, riceve un'Istanza già pronta.
"""

from dataclasses import dataclass, field
from itertools import combinations

from instances import Istanza

# Alias di tipo: un nodo e' semplicemente una tupla di interi con segno.
Nodo = tuple[int, ...]
Arco = tuple[Nodo, Nodo]


# ----------------------------------------------------------------------
# Indicatori di fattibilita' f1, f2 (Sez. 3.2)
# ----------------------------------------------------------------------
#
# f1(i,j): fattibile visitare j+ -> i+ -> j- -> i- (il giro di i dentro quello di j)
# f2(i,j): fattibile visitare j+ -> i+ -> i- -> j- (i sale e scende dentro il giro di j)
#
# Convenzione del paper: f1(i,0) = f1(0,i) = f2(i,0) = f2(0,i) = 1, il deposito
# come utente fittizio non impone mai vincoli.

EPS = 1e-9  # tolleranza sui confronti float (i ride time cadono spesso esattamente su L)



def schedule_minimo(sequenza_id: list[int],
                    vincoli_ride: list[tuple[int, int]],
                    istanza: Istanza,
                    tempi: list[float] | None = None) -> list[float] | None:
    """
    Schedule minimo lungo la sequenza di località data: per ogni posizione k il più
    piccolo istante di inizio servizio B_k compatibile con finestre, tempi di viaggio
    e ride time. None se nessuno schedule è ammissibile.

    Tutti i vincoli hanno la forma B_u - B_v <= c:
        B_k >= e_k                      ->  B_rif - B_k <= -e_k
        B_k <= l_k                      ->  B_k - B_rif <=  l_k
        B_{k+1} >= B_k + s_k + t_k      ->  B_k - B_{k+1} <= -(s_k + t_k)
        B_d - B_p <= L + s_p            ->  vincolo di ride time
    Un sistema di vincoli di differenza è ammissibile se e solo se il suo grafo dei
    vincoli non ha cicli negativi (Bellman-Ford). Al punto fisso, B = -dist è la
    soluzione più piccola: partendo da dist = 0, ogni distanza scende solo quanto serve.

    vincoli_ride: coppie (posizione del pickup, posizione della delivery) nella
    sequenza, una per ogni utente il cui ride time va limitato.
    tempi: tempi di percorrenza fra posizioni consecutive (lunghezza n-1). None usa
    istanza.tempo; per le rotte estratte dal modello si passano i tempi degli archi.
    """
    n = len(sequenza_id)
    if tempi is None:
        tempi = [istanza.tempo(sequenza_id[k], sequenza_id[k + 1])
                 for k in range(n - 1)]
    elif len(tempi) != n - 1:
        raise ValueError(f"servono {n - 1} tempi di percorrenza, ricevuti {len(tempi)}")

    RIF = n                                   # nodo di riferimento, B_RIF = 0
    archi: list[tuple[int, int, float]] = []   # (u, v, peso) <=> B_u - B_v <= peso

    for k, id_nodo in enumerate(sequenza_id):
        nodo = istanza.nodo(id_nodo)
        archi.append((k, RIF, nodo.l))
        archi.append((RIF, k, -nodo.e))

    for k in range(n - 1):
        nodo = istanza.nodo(sequenza_id[k])
        archi.append((k, k + 1, -(nodo.servizio + tempi[k])))

    for pos_p, pos_d in vincoli_ride:
        s_p = istanza.nodo(sequenza_id[pos_p]).servizio
        L_i = istanza.ride_max(istanza.utente(sequenza_id[pos_p]))
        archi.append((pos_d, pos_p, L_i + s_p))

    dist = [0.0] * (n + 1)
    for _ in range(n + 1):
        aggiornato = False
        for u, v, peso in archi:
            if dist[u] + peso < dist[v] - EPS:
                dist[v] = dist[u] + peso
                aggiornato = True
        if not aggiornato:
            # punto fisso raggiunto: nessun ciclo negativo
            return [dist[RIF] - dist[k] for k in range(n)]
    return None                  # ancora in aggiornamento dopo n+1 passate


def _esiste_schedule(sequenza_id: list[int],
                     vincoli_ride: list[tuple[int, int]],
                     istanza: Istanza) -> bool:
    """Esiste uno schedule ammissibile? È la domanda che f1/f2 pongono."""
    return schedule_minimo(sequenza_id, vincoli_ride, istanza) is not None


def f1(i: int, j: int, istanza: Istanza) -> bool:
    """Fattibilità di j+ -> i+ -> j- -> i- (ride time e finestre)."""
    if i == 0 or j == 0:
        return True
    sequenza = [istanza.pickup(j).id, istanza.pickup(i).id,
                istanza.delivery(j).id, istanza.delivery(i).id]
    return _esiste_schedule(sequenza, [(1, 3), (0, 2)], istanza)


def f2(i: int, j: int, istanza: Istanza) -> bool:
    """Fattibilità di j+ -> i+ -> i- -> j- (ride time e finestre)."""
    if i == 0 or j == 0:
        return True
    sequenza = [istanza.pickup(j).id, istanza.pickup(i).id,
                istanza.delivery(i).id, istanza.delivery(j).id]
    return _esiste_schedule(sequenza, [(1, 2), (0, 3)], istanza)


# ----------------------------------------------------------------------
# Generazione dei nodi (Sez. 3.1)
# ----------------------------------------------------------------------

def genera_companion_validi(utente: int, tipo: str, istanza: Istanza) -> list[tuple[int, ...]]:
    """
    Sottoinsiemi di compagni di viaggio ammissibili per un evento (pickup o delivery)
    dell'utente `utente`: rispettano la capacità (somma q <= Q) e la compatibilità
    individuale f1/f2 di ciascun compagno con `utente`. Il paper non richiede
    compatibilità a coppie fra i compagni stessi, solo con l'utente dell'evento.

    La condizione è asimmetrica fra pickup e delivery (definizioni Vi+/Vi-, pag. 4):
        pickup:   f1(utente, compagno) OR f2(utente, compagno)
        delivery: f1(compagno, utente) OR f2(utente, compagno)
    """
    if tipo not in ("pickup", "delivery"):
        raise ValueError(f"tipo non valido: {tipo!r} (atteso 'pickup' o 'delivery')")

    Q = istanza.Q
    capacita_residua = Q - istanza.carico(utente)

    candidati = []
    for altro in istanza.utenti():
        if altro == utente:
            continue
        if tipo == "pickup":
            ammissibile = f1(utente, altro, istanza) or f2(utente, altro, istanza)
        else:
            ammissibile = f1(altro, utente, istanza) or f2(utente, altro, istanza)
        if ammissibile:
            candidati.append(altro)

    compagni_validi: list[tuple[int, ...]] = [()]  # nessun compagno e' sempre valido
    for dim in range(1, Q):  # al massimo Q-1 compagni (un posto e' di `utente`)
        for gruppo in combinations(candidati, dim):
            carico_gruppo = sum(istanza.carico(u) for u in gruppo)
            if carico_gruppo <= capacita_residua:
                compagni_validi.append(gruppo)
    return compagni_validi


def crea_nodo(utente: int, tipo: str, compagni: tuple[int, ...], Q: int) -> Nodo:
    """
    Rappresentazione canonica (Q-tupla) di un evento.
    tipo determina il segno di v1; compagni sono gli utenti a bordo oltre a `utente`,
    già un insieme valido.
    """
    if tipo not in ("pickup", "delivery"):
        raise ValueError(f"tipo non valido: {tipo!r} (atteso 'pickup' o 'delivery')")

    v1 = utente if tipo == "pickup" else -utente
    resto = tuple(sorted(compagni, reverse=True))
    resto = resto + (0,) * (Q - 1 - len(resto))
    return (v1,) + resto


def genera_nodi(istanza: Istanza) -> set[Nodo]:
    """Assembla V = V0 U (unione Vi+) U (unione Vi-), per i in 1..n."""
    Q = istanza.Q
    nodi: set[Nodo] = {(0,) * Q}  # deposito

    for utente in istanza.utenti():
        for tipo in ("pickup", "delivery"):
            for compagni in genera_companion_validi(utente, tipo, istanza):
                nodi.add(crea_nodo(utente, tipo, compagni, Q))

    return nodi


# ----------------------------------------------------------------------
# Generazione degli archi A = A1 U ... U A6 (Sez. 3.3)
# ----------------------------------------------------------------------
#
# Ogni Ak è definito (pag. 4-5) come uguaglianza fra insiemi di occupanti
# prima/dopo la transizione. I nodi sono già in forma canonica, quindi confrontare
# insiemi di interi basta, tranne in A3 dove il paper confronta le tuple v2..vQ direttamente.

def genera_archi(grafo: "Grafo") -> set[Arco]:
    istanza = grafo.istanza
    deposito = grafo.deposito()
    zero_pad = (0,) * (istanza.Q - 1)

    pickup_di = {i: grafo.nodi_pickup(i) for i in istanza.utenti()}
    delivery_di = {i: grafo.nodi_delivery(i) for i in istanza.utenti()}

    archi: set[Arco] = set()

    # --- A1: pickup i -> delivery j (j = i possibile: nessuno sale/scende tra i due) ---
    for i in istanza.utenti():
        for u in pickup_di[i]:
            occupanti_u = set(u)  # u[0] = i, quindi set(u) = {i} U compagni
            for j in istanza.utenti():
                for w in delivery_di[j]:
                    occupanti_w = {j} | set(w[1:])
                    if occupanti_u == occupanti_w:
                        archi.add((u, w))

    # --- A2: pickup i (con un posto libero, vQ=0) -> pickup j, j != i ---
    for i in istanza.utenti():
        for u in pickup_di[i]:
            if u[-1] != 0:
                continue
            occupanti_dopo_carico_j = {i} | set(u[1:-1])
            for j in istanza.utenti():
                if j == i:
                    continue
                for w in pickup_di[j]:
                    if set(w[1:]) == occupanti_dopo_carico_j:
                        archi.add((u, w))

    # --- A3: delivery i -> pickup j, j != i, stessi compagni (v2..vQ identici) ---
    for i in istanza.utenti():
        for u in delivery_di[i]:
            for j in istanza.utenti():
                if j == i:
                    continue
                for w in pickup_di[j]:
                    if u[1:] == w[1:]:
                        archi.add((u, w))

    # --- A4: delivery i -> delivery j (con un posto libero finale, wQ=0), j != i ---
    for i in istanza.utenti():
        for u in delivery_di[i]:
            occupanti_u = set(u[1:])
            for j in istanza.utenti():
                if j == i:
                    continue
                for w in delivery_di[j]:
                    if w[-1] != 0:
                        continue
                    occupanti_prima_scarico_j = {j} | set(w[1:-1])
                    if occupanti_u == occupanti_prima_scarico_j:
                        archi.add((u, w))

    # --- A5: delivery i con veicolo ora vuoto -> deposito ---
    for i in istanza.utenti():
        for u in delivery_di[i]:
            if u[1:] == zero_pad:
                archi.add((u, deposito))

    # --- A6: deposito -> pickup i con veicolo prima vuoto ---
    for i in istanza.utenti():
        for u in pickup_di[i]:
            if u[1:] == zero_pad:
                archi.add((deposito, u))

    return archi


# ----------------------------------------------------------------------
# Costi e tempi degli archi (Sez. 3.4)
# ----------------------------------------------------------------------

def localita_id(nodo: Nodo, istanza: Istanza, *, is_partenza: bool) -> int:
    """
    Id della località fisica (in Istanza) di un nodo evento.
    Il deposito è un caso speciale: ha due id fisici nel file (0 iniziale, 2n+1
    finale) ma un solo nodo nel grafo event-based. `is_partenza` sceglie quale dei
    due usare, a seconda che il nodo sia origine o destinazione dell'arco.
    """
    if all(v == 0 for v in nodo):
        return istanza.deposito_iniziale().id if is_partenza else istanza.deposito_finale().id
    utente = abs(nodo[0])
    return istanza.pickup(utente).id if nodo[0] > 0 else istanza.delivery(utente).id


def _localita_arco(arco: Arco, istanza: Istanza) -> tuple[int, int]:
    """Id delle due localita' fisiche collegate dall'arco (partenza, arrivo)."""
    v, w = arco
    return (localita_id(v, istanza, is_partenza=True),
            localita_id(w, istanza, is_partenza=False))

# ----------------------------------------------------------------------
# Contenitore del grafo
# ----------------------------------------------------------------------

@dataclass
class Grafo:
    """Il grafo event-based G = (V, A) associato a un'Istanza, con adiacenze pronte per model.py."""
    istanza: Istanza
    nodi: set[Nodo] = field(default_factory=set)
    archi: set[Arco] = field(default_factory=set)
    delta_out: dict[Nodo, list[Arco]] = field(default_factory=dict, repr=False)
    delta_in: dict[Nodo, list[Arco]] = field(default_factory=dict, repr=False)

    @classmethod
    def costruisci(cls, istanza: Istanza) -> "Grafo":
        g = cls(istanza=istanza, nodi=genera_nodi(istanza))
        g.archi = genera_archi(g)
        g._costruisci_adiacenze()
        return g

    def _costruisci_adiacenze(self) -> None:
        self.delta_out = {v: [] for v in self.nodi}
        self.delta_in = {v: [] for v in self.nodi}
        for arco in self.archi:
            v, w = arco
            self.delta_out[v].append(arco)
            self.delta_in[w].append(arco)

    # ------------------------------------------------------------------
    # accesso ai nodi e agli archi, usato da model.py e objectives.py
    # ------------------------------------------------------------------
    def deposito(self) -> Nodo:
        return (0,) * self.istanza.Q

    def nodi_pickup(self, utente: int) -> list[Nodo]:
        return [v for v in self.nodi if v[0] == utente]

    def nodi_delivery(self, utente: int) -> list[Nodo]:
        return [v for v in self.nodi if v[0] == -utente]

    def costo(self, arco: Arco) -> float:
        """c_a: costo di routing dell'arco (Cordeau: distanza euclidea; OSM: km su strada)."""
        return self.istanza.costo(*_localita_arco(arco, self.istanza))

    def tempo(self, arco: Arco) -> float:
        """t_a: tempo di viaggio dell'arco in minuti (Cordeau: = costo; OSM: 4 * costo a 15 km/h)."""
        return self.istanza.tempo(*_localita_arco(arco, self.istanza))


if __name__ == "__main__":
    # verifica sull'Esempio 1 del paper (Sez. 3.1, pag. 5): 3 utenti, Q=3,
    # q1=q2=1, q3=3, nessuna finestra temporale
    from instances import Nodo as NodoIstanza

    istanza_esempio = Istanza(
        nome="esempio1_paper",
        K=1, n=3, T=999.0, Q=3, L=999.0,
        nodi=[
            NodoIstanza(id=0, x=0, y=0, servizio=0, domanda=0, e=0, l=999),
            NodoIstanza(id=1, x=0, y=0, servizio=0, domanda=1, e=0, l=999),  # pickup 1
            NodoIstanza(id=2, x=0, y=0, servizio=0, domanda=1, e=0, l=999),  # pickup 2
            NodoIstanza(id=3, x=0, y=0, servizio=0, domanda=3, e=0, l=999),  # pickup 3
            NodoIstanza(id=4, x=0, y=0, servizio=0, domanda=-1, e=0, l=999),  # delivery 1
            NodoIstanza(id=5, x=0, y=0, servizio=0, domanda=-1, e=0, l=999),  # delivery 2
            NodoIstanza(id=6, x=0, y=0, servizio=0, domanda=-3, e=0, l=999),  # delivery 3
            NodoIstanza(id=7, x=0, y=0, servizio=0, domanda=0, e=0, l=999),
        ],
    )

    g = Grafo.costruisci(istanza_esempio)

    print(f"Nodi generati: {len(g.nodi)}")
    print(f"Archi generati: {len(g.archi)}")

    # Controllo 1: nessun nodo deve contenere utente 1 e utente 3 insieme
    # (q1+q3 = 1+3 = 4 > Q=3).
    violazioni = [v for v in g.nodi if 1 in (abs(v[0]), *v[1:]) and 3 in (abs(v[0]), *v[1:])]
    assert not violazioni, f"Trovati nodi con utenti 1 e 3 insieme: {violazioni}"
    print("OK: nessun nodo contiene utenti 1 e 3 insieme.")

    # Controllo 2: i due dicycle d'esempio del paper (C1, C2) devono essere
    # percorribili: ogni arco che li compone deve esistere in g.archi.
    C1 = [(0, 0, 0), (1, 0, 0), (2, 1, 0), (-2, 1, 0), (-1, 0, 0), (0, 0, 0)]
    C2 = [(0, 0, 0), (3, 0, 0), (-3, 0, 0), (0, 0, 0)]

    for ciclo, nome in [(C1, "C1"), (C2, "C2")]:
        for k in range(len(ciclo) - 1):
            arco = (ciclo[k], ciclo[k + 1])
            assert arco in g.archi, f"Arco mancante nel dicycle {nome}: {arco}"
    print("OK: tutti gli archi dei dicycle C1 e C2 del paper esistono nel grafo generato.")