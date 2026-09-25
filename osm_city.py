"""
Generatore di istanze DARP da una rete stradale OpenStreetMap, secondo la ricetta
del caso Wuppertal (Sez. 4.2 del paper). Vale per qualunque città: cambiano solo
la rete (il GraphML prodotto da rete_osm.py) e le coordinate del deposito.

Ogni istanza è un file JSON con:
  - i 2n+2 nodi nella convenzione Cordeau (0 deposito, 1..n pickup, n+1..2n delivery,
    2n+1 deposito finale), con finestre, servizio, domanda, L_i e tipo "inbound"
  - la matrice dei costi c[a][b] in km fra i nodi (asimmetrica); il tempo è c * 60/v

Scelte nostre, non fissate dal paper:
  - pickup e drop-off solo su strade con nome, dei tipi in HIGHWAY_FERMATA
  - pickup e drop-off distanti almeno DISTANZA_MIN_KM su strada
  - le richieste non servibili da sole (deposito -> pickup -> drop-off -> deposito)
    vengono ricampionate per intero; gli scarti sono salvati nel file
  - K = n provvisorio: il numero di veicoli definitivo lo scrive imposta_k.py,
    dalla Tabella 7 del paper

Uso:
    python osm_city.py trieste_osm_highway_drive.graphml istanze_trieste prova
    python osm_city.py trieste_osm_highway_drive.graphml istanze_trieste
Con "prova" genera, stampa e salva una sola istanza (Q3, n = 20); senza, tutte le 60.
Entrambi sovrascrivono i JSON della cartella con K = n: i seed sono fissi e le istanze
escono identiche, ma poi va rilanciato imposta_k.py.
"""

import json
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox

# ---- citta' -------------------------------------------------------------------------------
CITTA = "Trieste"
DEPOSITO_LAT, DEPOSITO_LON = 45.657, 13.772      # Trieste Centrale (approssimato)

# ---- ricetta del paper (Sez. 4.2) ------------------------------------------------------------
T = 240.0
VELOCITA_KMH = 15.0
MINUTI_PER_KM = 60.0 / VELOCITA_KMH              # = 4: t_a = 4 c_a
AMPIEZZA_FINESTRA = 15.0
ISTANTI_PICKUP = list(range(5, 206, 5))          # e_i+ in {5, 10, ..., 205}
FATTORE_RIDE = 1.5                               # L_i = 1.5 t_i
POSTI = {3: (1, 1), 6: (1, 6)}                   # Q -> intervallo di q_i; s_i = q_i
VALORI_N = [20, 30, 40, 60, 80, 100]
ISTANZE_PER_N = 5

# ---- scelte nostre -----------------------------------------------------------------------------
DISTANZA_MIN_KM = 0.5
HIGHWAY_FERMATA = {"primary", "secondary", "tertiary",
                   "unclassified", "residential", "living_street"}
DECIMALI_KM = 4                                  # 0.1 m: piu' che sufficiente
TOLLERANZA = 1e-6


# =============================================================================================
# campionamento
# =============================================================================================
def nomi_arco(dati: dict) -> list[str]:
    nome = dati.get("name")
    if nome is None:
        return []
    return nome if isinstance(nome, list) else [nome]


def tipi_arco(dati: dict) -> list[str]:
    tipo = dati.get("highway")
    return tipo if isinstance(tipo, list) else [tipo]


def fermate_per_nome(G: nx.MultiDiGraph) -> dict[str, list[int]]:
    """Per ogni nome di strada, i nodi (ordinati) dove ci si puo' fermare lungo quella strada."""
    per_nome: dict[str, set[int]] = {}
    for u, v, dati in G.edges(data=True):
        if not all(t in HIGHWAY_FERMATA for t in tipi_arco(dati)):
            continue
        for nome in nomi_arco(dati):
            per_nome.setdefault(nome, set()).update((u, v))
    return {nome: sorted(nodi) for nome, nodi in sorted(per_nome.items())}


def campiona_nodo(rng: random.Random, fermate: dict[str, list[int]], nomi: list[str]) -> int:
    """Fermata casuale: prima una strada a caso, poi un nodo a caso lungo quella strada."""
    nome = rng.choice(nomi)
    return rng.choice(fermate[nome])


def esito_richiesta(e: float, s: float, t_dep_p: float, t_i: float, t_d_dep: float) -> str:
    """'ok' se la richiesta, servita da sola, rispetta finestra di pickup e orizzonte T."""
    if t_dep_p > e + AMPIEZZA_FINESTRA:
        return "pickup non raggiungibile entro l_i+"
    rientro = max(e, t_dep_p) + s + t_i + s + t_d_dep
    if rientro > T + TOLLERANZA:
        return "rientro al deposito oltre T"
    return "ok"


@dataclass
class Richiesta:
    pickup: int      # nodo OSM
    dropoff: int     # nodo OSM
    q: int           # posti richiesti = tempo di servizio
    e: float         # e_i+


class Rete:
    """La rete stradale con tutto cio' che serve al campionamento, calcolato una volta sola."""

    def __init__(self, percorso: Path):
        self.nome_file = percorso.name
        self.G = ox.io.load_graphml(percorso)
        self.deposito = ox.distance.nearest_nodes(self.G, X=DEPOSITO_LON, Y=DEPOSITO_LAT)
        self.min_da_dep = self._minuti(nx.single_source_dijkstra_path_length(
            self.G, self.deposito, weight="length"))
        self.min_verso_dep = self._minuti(nx.single_source_dijkstra_path_length(
            self.G.reverse(copy=False), self.deposito, weight="length"))
        self.fermate = fermate_per_nome(self.G)
        self.nomi = list(self.fermate)

    @staticmethod
    def _minuti(metri: dict) -> dict:
        return {v: d / 1000 * MINUTI_PER_KM for v, d in metri.items()}

    def km(self, a: int, b: int) -> float:
        return 0.0 if a == b else nx.bidirectional_dijkstra(self.G, a, b, weight="length")[0] / 1000


def genera_richieste(rete: Rete, rng: random.Random, n: int, Q: int) -> tuple[list[Richiesta], Counter]:
    richieste: list[Richiesta] = []
    scarti: Counter = Counter()
    while len(richieste) < n:
        p = campiona_nodo(rng, rete.fermate, rete.nomi)
        d = campiona_nodo(rng, rete.fermate, rete.nomi)
        km_pd = rete.km(p, d)
        if km_pd < DISTANZA_MIN_KM:
            scarti["pickup e drop-off troppo vicini"] += 1
            continue
        q = rng.randint(*POSTI[Q])
        e = float(rng.choice(ISTANTI_PICKUP))
        esito = esito_richiesta(e, q, rete.min_da_dep[p], km_pd * MINUTI_PER_KM,
                                rete.min_verso_dep[d])
        if esito != "ok":
            scarti[esito] += 1          # si ricampiona l'intera richiesta
            continue
        richieste.append(Richiesta(p, d, q, e))
    return richieste, scarti


# =============================================================================================
# costruzione dell'istanza
# =============================================================================================
def matrice_costi(G: nx.MultiDiGraph, luoghi: list[int]) -> list[list[float]]:
    """c[a][b] in km fra i nodi dell'istanza; luoghi[k] = nodo OSM del nodo-istanza k."""
    distinti = sorted(set(luoghi))
    da: dict[int, dict[int, float]] = {}
    for sorgente in distinti:
        dist = nx.single_source_dijkstra_path_length(G, sorgente, weight="length")
        mancanti = [b for b in distinti if b not in dist]
        if mancanti:
            raise ValueError(f"da {sorgente} non si raggiungono {mancanti}: rete non fortemente connessa")
        da[sorgente] = {b: dist[b] for b in distinti}
    return [[round(da[a][b] / 1000, DECIMALI_KM) for b in luoghi] for a in luoghi]


def costruisci_istanza(nome: str, seed: str, Q: int, richieste: list[Richiesta],
                       scarti: Counter, rete: Rete) -> dict:
    n = len(richieste)
    dep = rete.deposito
    luoghi = [dep] + [r.pickup for r in richieste] + [r.dropoff for r in richieste] + [dep]
    c = matrice_costi(rete.G, luoghi)

    def nodo(k: int, osm: int, servizio: float, domanda: int, e: float, l: float, **altro) -> dict:
        return {"id": k, "osm": int(osm), "lat": rete.G.nodes[osm]["y"], "lon": rete.G.nodes[osm]["x"],
                "servizio": servizio, "domanda": domanda, "e": e, "l": l, **altro}

    pickup, delivery = [], []
    for i, r in enumerate(richieste, start=1):
        t_i = c[i][i + n] * MINUTI_PER_KM
        L_i = FATTORE_RIDE * t_i
        l_p = r.e + AMPIEZZA_FINESTRA
        e_d = r.e + r.q + t_i                          # eq. (5)
        l_d = min(T, l_p + r.q + L_i)                  # eq. (5)
        pickup.append(nodo(i, r.pickup, r.q, r.q, r.e, l_p, L=L_i, tipo="inbound"))
        delivery.append(nodo(i + n, r.dropoff, r.q, -r.q, e_d, l_d))

    nodi = ([nodo(0, dep, 0, 0, 0.0, T)] + pickup + delivery + [nodo(2 * n + 1, dep, 0, 0, 0.0, T)])
    return {
        "formato": "darp-osm-v1",
        "nome": nome, "citta": CITTA, "rete": rete.nome_file, "seed": seed,
        "K": n, "n": n, "T": T, "Q": Q, "velocita_kmh": VELOCITA_KMH,
        "scelte": {"distanza_min_km": DISTANZA_MIN_KM, "highway_fermata": sorted(HIGHWAY_FERMATA),
                   "fattibilita_singola": "richiesta ricampionata per intero", "K": "n"},
        "scarti": dict(scarti),
        "nodi": nodi,
        "costo_km": c,
    }


# =============================================================================================
# verifica dell'istanza generata: ValueError al primo controllo che non torna
# =============================================================================================
def verifica(ist: dict) -> None:
    n, nome = ist["n"], ist["nome"]
    nodi, c = ist["nodi"], np.array(ist["costo_km"])
    mpk = 60.0 / ist["velocita_kmh"]

    if c.shape != (2 * n + 2, 2 * n + 2) or len(nodi) != 2 * n + 2:
        raise ValueError(f"{nome}: dimensioni errate")
    if [v["id"] for v in nodi] != list(range(2 * n + 2)):
        raise ValueError(f"{nome}: id dei nodi non consecutivi")
    if not np.all(np.isfinite(c)) or c.min() < 0 or np.any(np.diag(c) != 0):
        raise ValueError(f"{nome}: matrice con valori non validi")

    # disuguaglianza triangolare: c[a][b] <= c[a][k] + c[k][b] per ogni k (tolleranza arrotondamenti)
    for k in range(c.shape[0]):
        if np.any(c > c[:, [k]] + c[[k], :] + 3 * 10 ** -DECIMALI_KM):
            raise ValueError(f"{nome}: disuguaglianza triangolare violata passando per il nodo {k}")

    for i in range(1, n + 1):
        p, d = nodi[i], nodi[i + n]
        s, t_i = p["servizio"], c[i][i + n] * mpk
        if p["tipo"] != "inbound" or p["domanda"] != -d["domanda"]:
            raise ValueError(f"{nome}: richiesta {i} malformata")
        if abs(d["e"] - (p["e"] + s + t_i)) > TOLLERANZA or \
           abs(d["l"] - min(ist["T"], p["l"] + s + p["L"])) > TOLLERANZA:
            raise ValueError(f"{nome}: richiesta {i}: finestra di delivery diversa dall'eq. (5)")
        esito = esito_richiesta(p["e"], s, c[0][i] * mpk, t_i, c[i + n][2 * n + 1] * mpk)
        if esito != "ok":
            raise ValueError(f"{nome}: richiesta {i} non servibile da sola ({esito})")


# =============================================================================================
# generazione e salvataggio
# =============================================================================================
def genera(rete: Rete, Q: int, n: int, m: int) -> dict:
    seed = f"{CITTA}-Q{Q}-n{n}-m{m}"                 # seed leggibile e riproducibile
    rng = random.Random(seed)
    richieste, scarti = genera_richieste(rete, rng, n, Q)
    ist = costruisci_istanza(f"{CITTA}_Q{Q}.{n}.{m}", seed, Q, richieste, scarti, rete)
    verifica(ist)
    return ist


def salva(ist: dict, cartella: Path) -> Path:
    percorso = cartella / f"{ist['nome']}.json"
    percorso.write_text(json.dumps(ist, ensure_ascii=False), encoding="utf-8")
    return percorso


def riassunto(ist: dict) -> str:
    n, c = ist["n"], ist["costo_km"]
    t = [c[i][i + n] * 60.0 / ist["velocita_kmh"] for i in range(1, n + 1)]
    scartate = sum(ist["scarti"].values())
    return (f"{ist['nome']:<22} t_i mediana {statistics.median(t):5.1f} min, max {max(t):5.1f}"
            f" | scartate {scartate:3d} {ist['scarti']}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Uso: python osm_city.py <rete.graphml> <cartella_uscita> [prova]")
        sys.exit(1)
    rete = Rete(Path(sys.argv[1]))
    cartella = Path(sys.argv[2])
    cartella.mkdir(parents=True, exist_ok=True)
    print(f"Rete {rete.nome_file}: deposito nodo {rete.deposito}, {len(rete.nomi)} strade con fermate\n")

    if len(sys.argv) == 4 and sys.argv[3] == "prova":
        ist = genera(rete, Q=3, n=20, m=1)
        print(riassunto(ist))
        n = ist["n"]
        print("\n  i   posti   pickup [e, l]      delivery [e, l]      t_i    L_i")
        for i in range(1, 4):
            p, d = ist["nodi"][i], ist["nodi"][i + n]
            t_i = ist["costo_km"][i][i + n] * MINUTI_PER_KM
            print(f"  {i:<3} {p['domanda']:^5}  [{p['e']:5.1f},{p['l']:6.1f}]   "
                  f"[{d['e']:6.1f},{d['l']:6.1f}]   {t_i:5.1f}  {p['L']:5.1f}")
        print(f"\nSalvata in {salva(ist, cartella)}")
    else:
        for Q in POSTI:
            for n in VALORI_N:
                for m in range(1, ISTANZE_PER_N + 1):
                    ist = genera(rete, Q, n, m)
                    salva(ist, cartella)
                    print(riassunto(ist))
        print(f"\nIstanze salvate in {cartella}")