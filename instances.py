"""
Lettura e rappresentazione delle istanze DARP: due formati in ingresso, una
sola rappresentazione (Istanza).

Cordeau (.txt, benchmark branch-and-cut):
    riga 1: K n T Q L
    righe successive: id x y servizio domanda e l
    costo = tempo = distanza euclidea; stesso L per tutti gli utenti; per ogni
    richiesta è data una sola finestra, l'altra si ricostruisce (eq. 5-6).

OSM (.json, generato da osm_city.py):
    matrice dei costi in km su strada (asimmetrica); tempo = costo * 60 / v;
    L_i diverso per utente; entrambe le finestre già calcolate; tipo
    inbound/outbound scritto esplicitamente per ogni richiesta.

Il resto del progetto legge costi, tempi e ride time solo con costo(), tempo()
e ride_max(): non deve sapere da quale formato viene l'istanza.

Numerazione dei nodi (incapsulata in pickup()/delivery()/utente(), il resto
del progetto non la ricalcola a mano):
    0          deposito iniziale
    1 .. n     pickup delle richieste 1..n
    n+1 .. 2n  delivery delle richieste 1..n (delivery di i ha id i+n)
    2n+1       deposito finale
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import math


TIPI_RICHIESTA = ("inbound", "outbound")
FORMATI_JSON = ("darp-osm-v1",)
TOLLERANZA_FINESTRE = 1e-6


@dataclass
class Nodo:
    """Una localita' fisica: pickup, drop-off o deposito."""
    id: int
    x: float          # Cordeau: coordinata x;  OSM: longitudine (solo per i grafici)
    y: float          # Cordeau: coordinata y;  OSM: latitudine  (solo per i grafici)
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
    K: int              # numero di veicoli disponibili
    n: int              # numero di richieste
    T: float            # durata massima di un percorso
    Q: int              # capacita' del veicolo
    L: float | None     # ride time massimo COMUNE (Cordeau); None se e' per utente (OSM)

    nodi: list[Nodo] = field(default_factory=list)
    L_utente: dict[int, float] = field(default_factory=dict)   # L_i per utente (OSM)
    tipo: dict[int, str] = field(default_factory=dict)         # utente -> "inbound"/"outbound"

    # matrici su rete stradale (OSM); None -> geometria euclidea (Cordeau)
    _costo: dict[tuple[int, int], float] | None = field(default=None, repr=False)
    _tempo: dict[tuple[int, int], float] | None = field(default=None, repr=False)

    # indice interno id -> Nodo, costruito automaticamente
    # privato: modificarlo da fuori renderebbe l'indice incoerente
    _per_id: dict[int, Nodo] = field(default_factory=dict, repr=False)

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
        """Nodo di drop-off (i-) dell'utente i, con i in 1..n (ha id i+n)."""
        return self.nodo(utente + self.n)

    def utente(self, id_nodo: int) -> int:
        """Operazione inversa di pickup()/delivery(): l'utente a cui appartiene un nodo."""
        if 1 <= id_nodo <= self.n:
            return id_nodo
        if self.n + 1 <= id_nodo <= 2 * self.n:
            return id_nodo - self.n
        raise ValueError(f"{self.nome}: il nodo {id_nodo} è un deposito, non appartiene a nessun utente")

    def deposito_iniziale(self) -> Nodo:
        return self.nodo(0)

    def deposito_finale(self) -> Nodo:
        return self.nodo(2 * self.n + 1)

    def carico(self, utente: int) -> int:
        """q_i : numero di posti occupati dall'utente i."""
        return self.pickup(utente).domanda

    def utenti(self) -> range:
        """Iteratore sugli indici utente: 1, 2, ..., n."""
        return range(1, self.n + 1)

    def ride_max(self, utente: int) -> float:
        """L_i : ride time massimo dell'utente i (per utente se definito, altrimenti il comune L)."""
        if utente in self.L_utente:
            return self.L_utente[utente]
        if self.L is None:
            raise ValueError(f"{self.nome}: nessun ride time massimo per l'utente {utente}")
        return self.L

    def tipo_richiesta(self, utente: int) -> str:
        """'inbound' (data la finestra di pickup) oppure 'outbound' (data quella di drop-off)."""
        return self.tipo[utente]

    def su_rete_stradale(self) -> bool:
        """True per le istanze OSM: costi e tempi vengono da una matrice, non dalle coordinate."""
        return self._costo is not None

    def stampa(self) -> str:
        """Stringa di riepilogo leggibile, utile per il debug e per i log."""
        return (
            f"Istanza {self.nome}: n={self.n} richieste, K={self.K} veicoli, "
            f"Q={self.Q}, {self._testo_L()}, T={self.T} min | {len(self.nodi)} nodi"
        )

    def _testo_L(self) -> str:
        return f"L={self.L} min" if self.L is not None else "L_i per utente"

    # ------------------------------------------------------------------
    # costi e tempi di viaggio
    # ------------------------------------------------------------------
    def costo(self, id_a: int, id_b: int) -> float:
        """c_ab : costo di routing (Cordeau: distanza euclidea; OSM: km su strada)."""
        if self._costo is not None:
            return self._costo[(id_a, id_b)]
        return self._euclidea(id_a, id_b)

    def tempo(self, id_a: int, id_b: int) -> float:
        """t_ab : tempo di viaggio in minuti (Cordeau: distanza euclidea; OSM: km * 60 / v)."""
        if self._tempo is not None:
            return self._tempo[(id_a, id_b)]
        return self._euclidea(id_a, id_b)

    def distanza(self, id_a: int, id_b: int) -> float:
        """Distanza euclidea fra le coordinate. Solo Cordeau: su rete stradale e' ambigua
        (costo o tempo?) e solleva un errore, cosi' un vecchio uso dimenticato non passa inosservato."""
        if self.su_rete_stradale():
            raise ValueError(f"{self.nome}: distanza() non vale su rete stradale: usare costo() o tempo()")
        return self._euclidea(id_a, id_b)

    def _euclidea(self, id_a: int, id_b: int) -> float:
        a, b = self.nodo(id_a), self.nodo(id_b)
        return math.hypot(a.x - b.x, a.y - b.y)

    def matrice_distanze(self) -> dict[tuple[int, int], float]:
        """Dizionario {(id_a, id_b): costo} su tutte le coppie di nodi."""
        ids = [nodo.id for nodo in self.nodi]
        return {(a, b): self.costo(a, b) for a in ids for b in ids}


# ======================================================================
# lettura
# ======================================================================
def leggi_istanza(path: str | Path) -> Istanza:
    """Legge un'istanza scegliendo il formato dall'estensione: .json -> OSM, altrimenti Cordeau."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return leggi_istanza_json(path)
    return leggi_istanza_cordeau(path)


def leggi_istanza_cordeau(path: str | Path) -> Istanza:
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

    istanza = Istanza(nome=path.stem, K=K, n=n, T=T, Q=Q, L=L, nodi=nodi)
    costruisci_finestre_mancanti(istanza)   # riconosce inbound/outbound e completa le finestre
    verifica_istanza(istanza)
    return istanza


def leggi_istanza_json(path: str | Path) -> Istanza:
    """Legge un'istanza su rete stradale generata da osm_city.py."""
    path = Path(path)
    dati = json.loads(path.read_text(encoding="utf-8"))
    if dati.get("formato") not in FORMATI_JSON:
        raise ValueError(f"{path.name}: formato {dati.get('formato')!r} non riconosciuto "
                         f"(attesi: {', '.join(FORMATI_JSON)})")

    nodi = [Nodo(id=int(v["id"]), x=float(v["lon"]), y=float(v["lat"]),
                 servizio=float(v["servizio"]), domanda=int(v["domanda"]),
                 e=float(v["e"]), l=float(v["l"]))
            for v in dati["nodi"]]

    # la matrice è indicizzata per posizione nella lista dei nodi: qui la traduco in id
    ids = [nodo.id for nodo in nodi]
    matrice = dati["costo_km"]
    if len(matrice) != len(ids) or any(len(riga) != len(ids) for riga in matrice):
        raise ValueError(f"{path.name}: la matrice dei costi non e' {len(ids)} x {len(ids)}")
    minuti_per_km = 60.0 / float(dati["velocita_kmh"])
    costo = {(a, b): float(matrice[ra][cb])
             for ra, a in enumerate(ids) for cb, b in enumerate(ids)}
    tempo = {coppia: km * minuti_per_km for coppia, km in costo.items()}

    # L_i e tipo stanno sul nodo di pickup, che ha id = indice dell'utente
    L_utente = {int(v["id"]): float(v["L"]) for v in dati["nodi"] if "L" in v}
    tipo = {int(v["id"]): v["tipo"] for v in dati["nodi"] if "tipo" in v}

    istanza = Istanza(nome=path.stem, K=int(dati["K"]), n=int(dati["n"]), T=float(dati["T"]),
                      Q=int(dati["Q"]), L=None, nodi=nodi, L_utente=L_utente, tipo=tipo,
                      _costo=costo, _tempo=tempo)
    verifica_istanza(istanza)
    verifica_finestre_derivate(istanza)     # le finestre sono gia' complete: qui solo controllo
    return istanza


# ======================================================================
# verifiche
# ======================================================================
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
        if istanza.tipo.get(i) not in TIPI_RICHIESTA:
            raise ValueError(f"{istanza.nome}: richiesta {i} senza tipo inbound/outbound valido")
        if istanza.ride_max(i) < istanza.tempo(p.id, d.id):
            raise ValueError(
                f"{istanza.nome}: richiesta {i}: L_i = {istanza.ride_max(i):.4f} minore del "
                f"tempo di viaggio diretto t_i = {istanza.tempo(p.id, d.id):.4f}"
            )

    for nodo in istanza.nodi:
        if nodo.e > nodo.l:
            raise ValueError(f"{istanza.nome}: nodo {nodo.id} ha finestra vuota [{nodo.e}, {nodo.l}]")

    if istanza.su_rete_stradale():
        ids = [nodo.id for nodo in istanza.nodi]
        for a in ids:
            if istanza.costo(a, a) != 0:
                raise ValueError(f"{istanza.nome}: costo non nullo da {a} a se stesso")
            for b in ids:
                if istanza.costo(a, b) < 0 or not math.isfinite(istanza.costo(a, b)):
                    raise ValueError(f"{istanza.nome}: costo non valido da {a} a {b}")


def _larghezza(nodo: Nodo) -> float:
    return nodo.l - nodo.e


def finestra_derivata(istanza: Istanza, utente: int, tipo: str) -> tuple[float, float]:
    """
    Eq. (5)-(6): la finestra non data della richiesta, calcolata da quella data.
        inbound  (data la finestra di pickup)   -> [e_i-, l_i-]  (eq. 5)
        outbound (data la finestra di drop-off) -> [e_i+, l_i+]  (eq. 6)
    """
    dep = istanza.deposito_iniziale()
    p, d = istanza.pickup(utente), istanza.delivery(utente)
    t_i = istanza.tempo(p.id, d.id)              # tempo di viaggio diretto
    L_i = istanza.ride_max(utente)
    if tipo == "inbound":
        return max(dep.e, p.e + p.servizio + t_i), min(dep.l, p.l + p.servizio + L_i)
    if tipo == "outbound":
        return max(dep.e, d.e - L_i - p.servizio), min(dep.l, d.l - t_i - p.servizio)
    raise ValueError(f"tipo non valido: {tipo!r} (atteso uno fra {TIPI_RICHIESTA})")


def costruisci_finestre_mancanti(istanza: Istanza) -> None:
    """
    Formato Cordeau: riconosce il tipo di ogni richiesta e ricostruisce la finestra
    mancante. Per ogni richiesta è data una sola finestra, di lunghezza fissa TW;
    l'altra è un segnaposto ampio. La finestra stretta è quella reale:
        pickup più stretta   -> inbound  -> si deriva la finestra di drop-off (eq. 5)
        drop-off più stretta -> outbound -> si deriva la finestra di pickup   (eq. 6)
    Il tipo va registrato in istanza.tipo prima di derivare la finestra, perché dopo
    il confronto fra ampiezze non è più affidabile (lo tronca l'orizzonte del deposito).
    """
    for i in istanza.utenti():
        p, d = istanza.pickup(i), istanza.delivery(i)
        w_p, w_d = _larghezza(p), _larghezza(d)
        if w_p == w_d:
            raise ValueError(
                f"{istanza.nome}: richiesta {i} ha finestre di pari ampiezza "
                f"({w_p}): impossibile stabilire se sia inbound o outbound"
            )
        tipo = "inbound" if w_d > w_p else "outbound"
        istanza.tipo[i] = tipo
        derivato = d if tipo == "inbound" else p
        derivato.e, derivato.l = finestra_derivata(istanza, i, tipo)


def verifica_finestre_derivate(istanza: Istanza, tolleranza: float = TOLLERANZA_FINESTRE) -> None:
    """
    Formato OSM: le finestre sono già complete. Controlla che la finestra derivata
    coincida con l'eq. (5)/(6) ricalcolata con i tempi e gli L_i del file.
    """
    for i in istanza.utenti():
        tipo = istanza.tipo_richiesta(i)
        e_atteso, l_atteso = finestra_derivata(istanza, i, tipo)
        derivato = istanza.delivery(i) if tipo == "inbound" else istanza.pickup(i)
        if abs(derivato.e - e_atteso) > tolleranza or abs(derivato.l - l_atteso) > tolleranza:
            raise ValueError(
                f"{istanza.nome}: richiesta {i} ({tipo}): finestra [{derivato.e}, {derivato.l}] "
                f"diversa dall'eq. (5)-(6), che da' [{e_atteso}, {l_atteso}]"
            )


def riassumi(istanza: Istanza) -> str:
    """Stringa di riepilogo leggibile, con il conteggio delle richieste inbound e outbound."""
    inbound = sum(1 for i in istanza.utenti() if istanza.tipo_richiesta(i) == "inbound")
    return (
        f"Istanza {istanza.nome}: n={istanza.n} richieste, K={istanza.K} veicoli, "
        f"Q={istanza.Q}, {istanza._testo_L()}, T={istanza.T} min | "
        f"{len(istanza.nodi)} nodi | inbound={inbound}, outbound={istanza.n - inbound}"
    )


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:] or ["a2-16.txt"]:
        ist = leggi_istanza(arg)
        print(riassumi(ist))