"""
graph.py — Costruzione del grafo event-based G = (V, A) per il DARP.

Riferimento: Gaul, Klamroth & Stiglmayr (2022), EJOR 301(3), Sez. 3.1.

Idea chiave del paper: invece di un grafo geografico (nodi = luoghi fisici),
ogni nodo qui rappresenta uno STATO DI OCCUPAZIONE del veicolo (una "Q-tupla"),
con la prima componente che indica l'evento piu' recente (pickup o delivery).

Convenzione di codifica di un nodo (v1, v2, ..., vQ):
    v1 = +i   -> utente i appena caricato (evento pickup, "i+")
    v1 = -i   -> utente i appena scaricato (evento delivery, "i-")
    vj = 0    -> posto libero (per j >= 2, zero-padding in coda)
    vj = k>0  -> utente k seduto a bordo (per j >= 2)
    (0,0,...,0) -> deposito

Ordinamento canonico (par. 2.2 del paper), per evitare che permutazioni della
stessa allocazione generino nodi duplicati:
    - v1 = ultimo evento (con il segno)
    - v2..vQ = compagni di viaggio, ordine DECRESCENTE, zeri in coda

Questo modulo dipende SOLO dall'interfaccia pubblica di Istanza (instances.py):
non conosce il formato Cordeau, non legge file. Riceve un'Istanza gia' pronta.
"""

from dataclasses import dataclass, field
from itertools import combinations

from instances import Istanza

# Alias di tipo: un nodo e' semplicemente una tupla di interi con segno.
Nodo = tuple[int, ...]
Arco = tuple[Nodo, Nodo]





# ----------------------------------------------------------------------
# FASE 3.2 — Indicatori di fattibilita' f1, f2
# ----------------------------------------------------------------------
#
# f1(i,j): fattibile visitare j+ -> i+ -> j- -> i-   (il giro di i "annidato" in quello di j)
# f2(i,j): fattibile visitare j+ -> i+ -> i- -> j-   (i sale e scende, tutto dentro il giro di j)
#
# Convenzione del paper: f1(i,0) = f1(0,i) = f2(i,0) = f2(0,i) = 1 
# il "deposito" come utente fittizio non impone mai vincoli.

def _orari_fattibili(sequenza_id: list[int], istanza: Istanza) -> list[float] | None:
    """
    Simula in avanti i tempi di inizio servizio B lungo una sequenza fissa di
    localita' (dati i loro id in Istanza). Ritorna la lista dei B se la
    sequenza rispetta tutte le finestre temporali, altrimenti None.

    Regola di propagazione: si arriva alla fermata successiva non prima di
    aver finito il servizio alla fermata precedente e aver percorso la
    distanza tra le due; se si arriva in anticipo rispetto alla finestra, si
    aspetta (B = e). Se si arriva dopo la chiusura della finestra, la
    sequenza e' infattibile.
    """
    B: list[float] = []

    for idx, id_nodo in enumerate(sequenza_id): # per ogni nodo della sequenza, calcolo il tempo di inizio servizio B
        """
        enumerate(sequenza_id) restituisce una coppia (indice, valore) per ogni elemento della lista 
        (es.) sequenza_id=[5,21,42,7] => idx=[0,1,2,3], id_nodo=[5,21,42,7]
        """    

        nodo = istanza.nodo(id_nodo)
        if idx == 0: # idx sarà sempre 0 per il primo nodo della sequenza, quindi il tempo di arrivo sarà sempre la finestra di apertura del nodo
            arrivo = nodo.e 
        else: # altrimenti calcolo il tempo di arrivo in base al nodo precedente e alla distanza
            id_prec = sequenza_id[idx - 1]
            nodo_prec = istanza.nodo(id_prec)
            arrivo = B[idx - 1] + nodo_prec.servizio + istanza.distanza(id_prec, id_nodo) # tempo di arrivo = tempo di inizio servizio precedente + tempo di servizio precedente + distanza tra i due nodi
        inizio = max(arrivo, nodo.e) # l'inizio del servizio e' il massimo tra il tempo di arrivo e la finestra di apertura del nodo
        if inizio > nodo.l:
            return None
        B.append(inizio) # aggiungo il tempo di inizio servizio alla lista B, per ogni nodo della sequenza
    return B




def f1(i: int, j: int, istanza: Istanza) -> bool: #i e j vengono passati dalla funzione genera_companion_validi()
    """Fattibilita' della sequenza j+ -> i+ -> j- -> i- (ride time e finestre)."""
    if i == 0 or j == 0:
        return True
    p_j, p_i = istanza.pickup(j).id, istanza.pickup(i).id # sull'istanza viene chiamato il metodo pickup(j) che restituisce il nodo di pickup dell'utente j, e poi viene preso l'id del nodo
    d_j, d_i = istanza.delivery(j).id, istanza.delivery(i).id
    B = _orari_fattibili([p_j, p_i, d_j, d_i], istanza) #B ora contiene i tempi di inizio servizio per ogni nodo della sequenza, se la sequenza e' fattibile, altrimenti None
    if B is None:
        return False
    s_pi = istanza.pickup(i).servizio #viene recuperato il tempo di servizio del nodo di pickup dell'utente i
    s_pj = istanza.pickup(j).servizio
    ride_i = B[3] - (B[1] + s_pi)  # il ride time di i è il tempo di fine servizio di i (B[3]) meno il tempo di inizio servizio di i (B[1]) più il tempo di servizio di i (s_pi) al momento del pickup
    ride_j = B[2] - (B[0] + s_pj)
    return ride_i <= istanza.L and ride_j <= istanza.L #viene ritornato True se il ride time di i e j e' minore o uguale a L, altrimenti False


#f2 verifica se la sequenza j+ -> i+ -> i- -> j- e' fattibile, cioe' se il ride time di i e j e' minore o uguale a L
def f2(i: int, j: int, istanza: Istanza) -> bool:
    """Fattibilita' della sequenza j+ -> i+ -> i- -> j- (ride time e finestre)."""
    if i == 0 or j == 0:
        return True
    p_j, p_i = istanza.pickup(j).id, istanza.pickup(i).id
    d_i, d_j = istanza.delivery(i).id, istanza.delivery(j).id
    B = _orari_fattibili([p_j, p_i, d_i, d_j], istanza) #qua viene creata la famosa sequenza_id che viene poi passata alla funzione _orari_fattibili
    if B is None:
        return False
    s_pi = istanza.pickup(i).servizio
    s_pj = istanza.pickup(j).servizio
    ride_i = B[2] - (B[1] + s_pi)
    ride_j = B[3] - (B[0] + s_pj)
    return ride_i <= istanza.L and ride_j <= istanza.L











# ----------------------------------------------------------------------
# FASE 3.1 — Generazione dei nodi V (ora con filtro f1/f2 oltre alla capacita')
# ----------------------------------------------------------------------

def genera_companion_validi(utente: int, tipo: str, istanza: Istanza) -> list[tuple[int, ...]]:
    """
    Tutti i sottoinsiemi di "compagni di viaggio" ammissibili per un evento
    (pickup o delivery) dell'utente `utente`, rispettando:
      1) il vincolo di capacita' (Sigma q <= Q);
      2) la compatibilita' individuale f1/f2 di ciascun compagno con `utente`
         (Fase 3.2). Nota: il paper NON richiede compatibilita' a coppie tra i
         compagni stessi, solo con l'utente "focale" dell'evento.

    La condizione di ammissibilita' individuale e' asimmetrica tra pickup e
    delivery (vedi definizioni Vi+/Vi- del paper, pag. 4):
      - pickup:   f1(utente, compagno) OR f2(utente, compagno)
      - delivery: f1(compagno, utente) OR f2(utente, compagno)
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
    Costruisce la rappresentazione canonica (Q-tupla) di un evento.

    tipo: "pickup" o "delivery" -> determina il segno di v1.
    compagni: utenti aggiuntivi a bordo (senza `utente`), gia' un insieme valido.
    """
    if tipo not in ("pickup", "delivery"):
        raise ValueError(f"tipo non valido: {tipo!r} (atteso 'pickup' o 'delivery')")

    v1 = utente if tipo == "pickup" else -utente
    resto = tuple(sorted(compagni, reverse=True))
    resto = resto + (0,) * (Q - 1 - len(resto))
    return (v1,) + resto






def genera_nodi(istanza: Istanza) -> set[Nodo]:

    """Assembla V = V0 U (unione Vi+) U (unione Vi-), per i in 1..n.
    l'obbiettivo e' generare tutti i nodi possibili, quindi per ogni utente i 
    genera tutti i nodi pickup e delivery possibili, 
    rispettando le condizioni di capacita' e compatibilita' individuale f1/f2."""

    Q = istanza.Q
    nodi: set[Nodo] = {(0,) * Q}  # creo un insieme di nodi e aggiungo il deposito (0, 0, ..., 0) come primo nodo

    for utente in istanza.utenti(): 
        for tipo in ("pickup", "delivery"):
            for compagni in genera_companion_validi(utente, tipo, istanza): # genera_companion_validi() ritorna una lista di tuple, dove ogni tupla rappresenta un insieme di compagni validi per l'utente e il tipo di evento (pickup o delivery)
                nodi.add(crea_nodo(utente, tipo, compagni, Q))

    return nodi









# ----------------------------------------------------------------------
# FASE 3.3 — Generazione degli archi A = A1 U ... U A6
# ----------------------------------------------------------------------
#
# Ogni Ak e' definito (pag. 4-5 del paper) come uguaglianza tra insiemi di
# occupanti prima/dopo la transizione. Qui sfruttiamo il fatto che i nodi
# sono gia' in forma canonica: confrontare insiemi di interi e' sufficiente,
# non serve un ordinamento esplicito (tranne in A3, dove il paper stesso usa
# le stesse variabili v2..vQ su entrambi i lati => confronto diretto di tupla).

def _pickup_index(grafo: "Grafo") -> dict[int, list[Nodo]]:
    return {i: grafo.nodi_pickup(i) for i in grafo.istanza.utenti()}


def _delivery_index(grafo: "Grafo") -> dict[int, list[Nodo]]:
    return {i: grafo.nodi_delivery(i) for i in grafo.istanza.utenti()}


def genera_archi(grafo: "Grafo") -> set[Arco]:
    istanza = grafo.istanza
    deposito = grafo.deposito()
    zero_pad = (0,) * (istanza.Q - 1)

    pickup_di = _pickup_index(grafo)
    delivery_di = _delivery_index(grafo)

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
# FASE 3.4 — Costi e tempi degli archi
# ----------------------------------------------------------------------

def localita_id(nodo: Nodo, istanza: Istanza, *, is_partenza: bool) -> int:
    """
    Id della localita' fisica (in Istanza) associata a un nodo evento.

    Il deposito e' un caso speciale: nel formato Cordeau esistono DUE id fisici
    per il deposito (0 = iniziale, 2n+1 = finale), ma nel grafo event-based e'
    rappresentato da un UNICO nodo (0,...,0). `is_partenza` disambigua quale
    dei due usare quando il nodo deposito e' l'origine o la destinazione
    dell'arco che si sta valutando.
    """
    if all(v == 0 for v in nodo):
        return istanza.deposito_iniziale().id if is_partenza else istanza.deposito_finale().id
    utente = abs(nodo[0])
    return istanza.pickup(utente).id if nodo[0] > 0 else istanza.delivery(utente).id


def costo_arco(arco: Arco, istanza: Istanza) -> float:
    """ca: costo di routing dell'arco = distanza tra le due localita' fisiche."""
    v, w = arco
    id_v = localita_id(v, istanza, is_partenza=True)
    id_w = localita_id(w, istanza, is_partenza=False)
    return istanza.distanza(id_v, id_w)


# Nei benchmark Cordeau (e nel paper) costo = tempo = distanza euclidea.
tempo_arco = costo_arco









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
        return g #g sarebbe un oggetto della classe grafo

    def _costruisci_adiacenze(self) -> None: #serve a costruire le liste di adiacenza delta_out e delta_in per ogni nodo del grafo
        self.delta_out = {v: [] for v in self.nodi} # 
        self.delta_in = {v: [] for v in self.nodi}
        for arco in self.archi: # per ogni arco del grafo
            v, w = arco # v = nodo di partenza, w = nodo di arrivo
            self.delta_out[v].append(arco) # v viene aggiunto alla lista delta_out del nodo v, quindi delta_out[v] contiene tutti gli archi in uscita da v
            self.delta_in[w].append(arco)

    # ------------------------------------------------------------------
    # accesso semantico, comodo per i test e per model.py
    # ------------------------------------------------------------------
    def deposito(self) -> Nodo:
        return (0,) * self.istanza.Q

    def nodi_pickup(self, utente: int) -> list[Nodo]:
        return [v for v in self.nodi if v[0] == utente]

    def nodi_delivery(self, utente: int) -> list[Nodo]:
        return [v for v in self.nodi if v[0] == -utente]

    def costo(self, arco: Arco) -> float:
        return costo_arco(arco, self.istanza)

    def tempo(self, arco: Arco) -> float:
        return tempo_arco(arco, self.istanza)





if __name__ == "__main__":
    # --- Sanity check sull'Esempio 1 del paper (Sez. 3.1, pag. 5) ---
    # 3 utenti, Q=3, q1=q2=1, q3=3. Niente finestre temporali in questo esempio.
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
    # percorribili, cioe' ogni arco che li compone deve esistere in g.archi.
    C1 = [(0, 0, 0), (1, 0, 0), (2, 1, 0), (-2, 1, 0), (-1, 0, 0), (0, 0, 0)]
    C2 = [(0, 0, 0), (3, 0, 0), (-3, 0, 0), (0, 0, 0)]

    for ciclo, nome in [(C1, "C1"), (C2, "C2")]:
        for k in range(len(ciclo) - 1):
            arco = (ciclo[k], ciclo[k + 1])
            assert arco in g.archi, f"Arco mancante nel dicycle {nome}: {arco}"
    print("OK: tutti gli archi dei dicycle C1 e C2 del paper esistono nel grafo generato.")
