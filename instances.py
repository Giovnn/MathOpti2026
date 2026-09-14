"""
instances.py — Lettura e rappresentazione delle istanze DARP.

Formato Cordeau (benchmark branch-and-cut):
    riga 1:  K n T Q L
    righe successive:  id  x  y  servizio  domanda  e  l

Convenzione di numerazione dei nodi:
    0          deposito iniziale
    1 .. n     pickup   delle richieste 1..n
    n+1 .. 2n  delivery delle richieste 1..n  (delivery di i ha id i+n)
    2n+1       deposito finale

Questa convenzione e' incapsulata nei metodi pickup()/delivery() di Istanza:
il resto del progetto non deve mai ricalcolarla a mano.
"""

from dataclasses import dataclass, field
from pathlib import Path
import math






@dataclass
class Nodo:
    """Una localita' fisica: pickup, drop-off o deposito."""
    id: int
    x: float
    y: float
    servizio: float   # s_i : durata dell'operazione di carico/scarico
    domanda: int      # q_i : >0 pickup, <0 delivery, 0 deposito
    e: float          # e_i : inizio finestra temporale
    l: float          # l_i : fine finestra temporale

    @property
    def is_pickup(self) -> bool:
        return self.domanda > 0

    @property
    def is_delivery(self) -> bool:
        return self.domanda < 0

    @property
    def is_deposito(self) -> bool:
        return self.domanda == 0






@dataclass
class Istanza:
    """Un'istanza completa del problema: parametri globali + nodi."""
    nome: str
    K: int      # numero di veicoli disponibili
    n: int      # numero di richieste
    T: float    # durata massima di un percorso
    Q: int      # capacita' del veicolo
    L: float    # ride time massimo per richiesta

    nodi: list[Nodo] = field(default_factory=list)
    _per_id: dict[int, Nodo] = field(default_factory=dict, repr=False)

    # indice interno id -> Nodo, costruito automaticamente
    # privato perchè non vogliamo erroneamente modificarlo dall'esterno, altrimenti l'indice diverrebbe incoerente



    def __post_init__(self):
        self._per_id = {nodo.id: nodo for nodo in self.nodi}


    # ------------------------------------------------------------------
    # accesso generico
    # ------------------------------------------------------------------
    def nodo(self, id_nodo: int) -> Nodo:
        """Nodo con l'id dato. Solleva KeyError se l'id non esiste."""
        return self._per_id[id_nodo]


    # ------------------------------------------------------------------
    # accesso semantico: la convenzione di numerazione vive SOLO qui
    # ------------------------------------------------------------------
    def pickup(self, utente: int) -> Nodo:
        """Nodo di pickup (i+) dell'utente i, con i in 1..n."""
        return self.nodo(utente)
    

    def delivery(self, utente: int) -> Nodo:
        """Nodo di drop-off (i-) dell'utente i, con i in 1..n."""
        return self.nodo(utente + self.n)
    # per la convenzione di numerazione, il nodo di delivery dell'utente i ha id = i+n, qui return self.nodo(utente + self.n) significa che se utente=1, allora delivery(1) restituira' il nodo con id=1+n, cioe' il nodo di drop-off dell'utente 1

    def deposito_iniziale(self) -> Nodo:
        return self.nodo(0)

    def deposito_finale(self) -> Nodo:
        return self.nodo(2 * self.n + 1)

    def carico(self, utente: int) -> int:
        """q_i : numero di posti occupati dall'utente i."""
        return self.pickup(utente).domanda

    def utenti(self) -> range:
        """Iteratore sugli indici utente: 1, 2, ..., n."""
        return range(1, self.n + 1) # ritorna un oggetto range che rappresenta gli indici degli utenti da 1 a n; per esempio se n=5, ritorna range(1, 6) che rappresenta gli indici degli utenti 1, 2, 3, 4, 5



    def stampa(self) -> str:
        """Stringa di riepilogo leggibile, utile per il debug e per i log."""
        return (
            f"Istanza {self.nome}: n={self.n} richieste, K={self.K} veicoli, "
            f"Q={self.Q}, L={self.L} min, T={self.T} min | "
            f"{len(self.nodi)} nodi"
        )



    # ------------------------------------------------------------------
    # geometria: distanze euclidee (Cordeau usa costo = tempo = distanza)
    # ------------------------------------------------------------------
    def distanza(self, id_a: int, id_b: int) -> float:
        a, b = self.nodo(id_a), self.nodo(id_b)
        return math.hypot(a.x - b.x, a.y - b.y)

    def matrice_distanze(self) -> dict[tuple[int, int], float]:
        """Dizionario {(id_a, id_b): distanza} su tutte le coppie di nodi."""
        ids = [nodo.id for nodo in self.nodi]
        return {(a, b): self.distanza(a, b) for a in ids for b in ids}







def leggi_istanza(path: str | Path) -> Istanza:
    """Legge un file in formato Cordeau e restituisce un oggetto Istanza."""
    path = Path(path)
    righe = path.read_text().splitlines()

    # --- riga 1: parametri globali: K, n, T, Q, L ---
    campi = righe[0].split()
    if len(campi) < 5:
        raise ValueError(f"Header malformato in {path.name}: {righe[0]!r}")
    K, n, T, Q, L = (
        int(campi[0]), int(campi[1]), float(campi[2]),
        int(campi[3]), float(campi[4]),
    )


    # --- righe successive: un nodo ciascuna ---
    nodi: list[Nodo] = []

    for riga in righe[1:]:
        campi = riga.split()
        if not campi:               # salta righe vuote
            continue
        if len(campi) < 7:
            raise ValueError(f"Riga nodo malformata in {path.name}: {riga!r}")
        nodi.append(Nodo(
            id=int(campi[0]),
            x=float(campi[1]),
            y=float(campi[2]),
            servizio=float(campi[3]),
            domanda=int(campi[4]),
            e=float(campi[5]),
            l=float(campi[6]),
        ))

    istanza = Istanza(nome=path.stem, K=K, n=n, T=T, Q=Q, L=L, nodi=nodi) #path.stem = nome del file senza estensione

    costruisci_finestre_mancanti(istanza) # ricostruisce le finestre temporali mancanti per ogni richiesta, se necessario

    verifica_istanza(istanza)
    return istanza


def verifica_istanza(istanza: Istanza) -> None:
    """Controlli di coerenza sui dati letti. Fallisce rumorosamente se qualcosa non torna."""
    attesi = 2 * istanza.n + 2
    if len(istanza.nodi) != attesi:
        raise ValueError(
            f"{istanza.nome}: attesi {attesi} nodi (2n+2 con n={istanza.n}), "
            f"trovati {len(istanza.nodi)}"
        )

    for i in istanza.utenti():
        p, d = istanza.pickup(i), istanza.delivery(i)
        if p.domanda <= 0:
            raise ValueError(f"{istanza.nome}: il nodo {p.id} dovrebbe essere un pickup")
        if d.domanda >= 0:
            raise ValueError(f"{istanza.nome}: il nodo {d.id} dovrebbe essere una delivery")
        if p.domanda != -d.domanda:
            raise ValueError(
                f"{istanza.nome}: richiesta {i}: domanda pickup {p.domanda} "
                f"non bilancia delivery {d.domanda}"
            )
        if p.domanda > istanza.Q:
            raise ValueError(
                f"{istanza.nome}: richiesta {i} chiede {p.domanda} posti > capacita' Q={istanza.Q}"
            )

    for nodo in istanza.nodi:
        if nodo.e > nodo.l:
            raise ValueError(f"{istanza.nome}: nodo {nodo.id} ha finestra vuota [{nodo.e}, {nodo.l}]")





def _larghezza(nodo: Nodo) -> float:
    return nodo.l - nodo.e


def costruisci_finestre_mancanti(istanza: Istanza) -> None:
    """Eq. (5)-(6) del paper: ricostruisce la finestra non specificata di ogni richiesta.

    Convenzione Cordeau: per ogni richiesta e' data UNA sola finestra, di lunghezza
    fissa TW; l'altra e' un segnaposto ampio. La finestra stretta e' quella reale.
        inbound  = data la finestra di pickup   -> si deriva quella di drop-off (eq. 5)
        outbound = data la finestra di drop-off -> si deriva quella di pickup   (eq. 6)
    """
    dep = istanza.deposito_iniziale()
    for i in istanza.utenti():
        p, d = istanza.pickup(i), istanza.delivery(i)
        t_i = istanza.distanza(p.id, d.id)          # tempo di viaggio diretto
        w_p, w_d = _larghezza(p), _larghezza(d)

        if w_p == w_d:
            raise ValueError(
                f"{istanza.nome}: richiesta {i} ha finestre di pari ampiezza "
                f"({w_p}): impossibile stabilire se sia inbound o outbound"
            )

        if w_d > w_p:                               # inbound
            d.e = max(dep.e, p.e + p.servizio + t_i)
            d.l = min(dep.l, p.l + p.servizio + istanza.L)
        else:                                       # outbound
            p.e = max(dep.e, d.e - istanza.L - p.servizio)
            p.l = min(dep.l, d.l - t_i - p.servizio)






def riassumi(istanza: Istanza) -> str:
    """Stringa di riepilogo leggibile, utile per il debug e per i log. Confronta le finestre
    di pickup e delivery per contare quante richieste hanno la finestra di pickup piu' stretta
    di quella di delivery (outbound) e quante hanno la finestra di delivery piu' stretta di quella di pickup (inbound)."""
    outbound = sum(
        1 for i in istanza.utenti()
        if istanza.pickup(i).l - istanza.pickup(i).e
        < istanza.delivery(i).l - istanza.delivery(i).e
    )
    return (
        f"Istanza {istanza.nome}: n={istanza.n} richieste, K={istanza.K} veicoli, "
        f"Q={istanza.Q}, L={istanza.L} min, T={istanza.T} min | "
        f"{len(istanza.nodi)} nodi | outbound={outbound}, inbound={istanza.n - outbound}"
    )


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:] or ["a2-16.txt"]:
        ist = leggi_istanza(arg)
        print(riassumi(ist))