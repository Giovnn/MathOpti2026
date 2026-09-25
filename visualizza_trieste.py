"""
Mappa interattiva delle istanze Trieste (rotte dei veicoli, pooling, pickup e
drop-off) in un unico file HTML da aprire nel browser.

Per ogni istanza:
1. risolve Model II con gli obiettivi scelti (default f_c) e ricostruisce sulla rete
   stradale la rotta di ogni veicolo, deposito -> fermate -> deposito. Ogni tratto fra
   due fermate consecutive è un arco del modello, cioè il percorso più breve sulla
   rete: la somma dei tratti deve dare il costo della rotta nel modello, e lo script
   lo controlla;
2. per ogni tratto annota chi è a bordo: lo spessore della linea cresce con i
   passeggeri e i tratti a veicolo vuoto sono grigi tratteggiati;
3. per ogni richiesta calcola il percorso diretto pickup -> drop-off, come in
   osm_city.py, e controlla che coincida con costo_km[i][n+i] del JSON (sulla mappa
   i percorsi diretti si accendono con un interruttore);
4. scrive un solo file HTML, con un menu per scegliere istanza e obiettivo.
Senza Gurobi la mappa si fa lo stesso, ma con i soli percorsi diretti.

Colori: un gruppo di pooling è un tratto della rotta fra due momenti in cui il
veicolo è vuoto. La rotta +16 +10 -16 +20 -20 -10 forma un solo gruppo {16, 10, 20}:
16 e 20 non sono mai a bordo insieme, ma entrambi lo sono con 10. Ogni gruppo ha un
colore, usato anche per i tratti percorsi con quel gruppo a bordo; le richieste
rifiutate sono grigie.

Uso (dalla cartella del progetto):
    python visualizza_trieste.py istanze_trieste/Trieste_Q3.20.*.json --rete trieste_osm_highway_drive.graphml
    ... --obiettivi fc fr frcr     menu con tre obiettivi (default: solo fc)
    ... --senza-soluzione          niente Gurobi: solo percorsi diretti, un colore per richiesta
    ... --out mappa.html           nome del file prodotto (default: mappa_trieste.html)
L'asterisco funziona anche dal terminale di Windows: i nomi li espande lo script.
Per aprire l'HTML serve internet: Leaflet, i font e lo sfondo della mappa vengono
scaricati dal browser.
"""

import argparse
import ast
import glob
import json
import math
from functools import lru_cache
from pathlib import Path

import networkx as nx

# Etichette leggibili degli obiettivi (le chiavi sono quelle di objectives.py).
ETICHETTE = {
    "fc": "f_c (costo)",
    "fr": "f_r (regret)",
    "fcr": "f_cr (costo + regret)",
    "frcr": "f_rcr (costo + regret + rifiuti)",
    "frmax": "f_rmax (regret massimo)",
    "fcrmax": "f_crmax (costo + regret massimo)",
    "fn": "f_n (rifiuti)",
}

# 20 colori ben distinguibili fra loro (serie di Kelly, senza bianco, nero e grigio, piu'
# un verde acqua). I grigi sono riservati alle richieste rifiutate e ai tratti a vuoto.
PALETTE = [
    "#F3C300", "#875692", "#F38400", "#A1CAF1", "#BE0032", "#C2B280", "#008856",
    "#E68FAC", "#0067A5", "#F99379", "#604E97", "#F6A600", "#B3446C", "#DCD300",
    "#882D17", "#8DB600", "#654522", "#E25822", "#2B3D26", "#00897B",
]
GRIGIO = "#9AA1A6"  # richieste rifiutate
GRIGIO_VUOTO = "#7D868C"  # tratti percorsi a veicolo vuoto

TOLLERANZA_KM = 1e-3  # costo_km e' arrotondato a 4 decimali: 1 metro di margine basta
PASSO_FRECCE_M = 700  # una freccia del verso di marcia ogni 700 metri di rotta


# ----------------------------------------------------------------------
# Rete stradale: percorsi e nomi delle vie
# ----------------------------------------------------------------------

def carica_rete(percorso: Path) -> nx.MultiDiGraph:
    """
    Legge il GraphML salvato da OSMnx, dove tutti gli attributi sono testo: converte in
    numeri coordinate e lunghezze, e gli id dei nodi in interi come il campo "osm" dei JSON.
    """
    G = nx.read_graphml(percorso, node_type=int)
    for _, dati in G.nodes(data=True):
        dati["lat"] = float(dati["y"])
        dati["lon"] = float(dati["x"])
    for _, _, dati in G.edges(data=True):
        dati["length"] = float(dati["length"])
    return G


def _vicino(p: list[float], q: list[float]) -> float:
    """Distanza al quadrato fra due punti [lat, lon]: basta per confrontare."""
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def punti_arco(G: nx.MultiDiGraph, u: int, v: int, arco: dict) -> list[list[float]]:
    """
    I punti [lat, lon] che disegnano l'arco u -> v, estremi compresi.

    Un arco della rete semplificata puo' seguire una strada curva: la forma sta
    nell'attributo "geometry", un testo del tipo "LINESTRING (lon lat, lon lat, ...)".
    Gli archi rettilinei non ce l'hanno e bastano i due estremi.
    """
    inizio = [G.nodes[u]["lat"], G.nodes[u]["lon"]]
    fine = [G.nodes[v]["lat"], G.nodes[v]["lon"]]
    if "geometry" not in arco:
        return [inizio, fine]

    testo = arco["geometry"]
    dentro = testo[testo.index("(") + 1: testo.rindex(")")]  # "lon lat, lon lat, ..."
    punti = []
    for coppia in dentro.split(","):
        lon, lat = coppia.split()
        punti.append([float(lat), float(lon)])
    # per sicurezza: il disegno deve partire da u, non da v
    if _vicino(punti[0], fine) < _vicino(punti[0], inizio):
        punti.reverse()
    return punti


@lru_cache(maxsize=None)
def percorso_piu_breve(G: nx.MultiDiGraph, da: int, a: int) -> tuple[float, list[list[float]]]:
    """
    Percorso più breve da -> a sulla rete orientata, come in osm_city.py: restituisce
    (km, punti [lat, lon] da disegnare). È in cache perché lo stesso tratto torna in più
    obiettivi.
    """
    nodi = nx.shortest_path(G, da, a, weight="length")
    metri = 0.0
    punti = [[G.nodes[da]["lat"], G.nodes[da]["lon"]]]
    for u, v in zip(nodi[:-1], nodi[1:]):  # coppie consecutive (u, v) del percorso
        # fra u e v possono esserci piu' archi paralleli: Dijkstra ha usato il piu' corto
        arco = min(G[u][v].values(), key=lambda dati: dati["length"])
        metri += arco["length"]
        punti += punti_arco(G, u, v, arco)[1:]  # [1:]: il primo punto c'e' gia'
    punti = [[round(lat, 6), round(lon, 6)] for lat, lon in punti]
    return metri / 1000, punti


def frecce(punti: list[list[float]]) -> list[list[float]]:
    """
    Dove mettere le frecce del verso di marcia lungo una linea: una ogni PASSO_FRECCE_M
    metri, oppure una sola a meta' se la linea e' corta.
    Restituisce [[lat, lon, angolo], ...], con l'angolo in gradi da nord in senso orario.
    """
    # lunghezza (metri) e direzione di ogni segmento; su pochi km la Terra si puo'
    # trattare come piatta: un grado di latitudine vale circa 110,5 km
    segmenti = []
    for (la1, lo1), (la2, lo2) in zip(punti[:-1], punti[1:]):
        dx = (lo2 - lo1) * 111_320 * math.cos(math.radians(la1))
        dy = (la2 - la1) * 110_540
        segmenti.append((math.hypot(dx, dy), math.degrees(math.atan2(dx, dy))))
    totale = sum(lunghezza for lunghezza, _ in segmenti)
    if totale < 80:
        return []
    if totale < PASSO_FRECCE_M:
        traguardi = [totale / 2]
    else:
        quante = int((totale - PASSO_FRECCE_M / 2) // PASSO_FRECCE_M) + 1
        traguardi = [PASSO_FRECCE_M / 2 + j * PASSO_FRECCE_M for j in range(quante)]

    risultato, percorsi_m, j = [], 0.0, 0
    for (lunghezza, angolo), (la1, lo1), (la2, lo2) in zip(segmenti, punti[:-1], punti[1:]):
        while j < len(traguardi) and traguardi[j] <= percorsi_m + lunghezza:
            f = (traguardi[j] - percorsi_m) / lunghezza if lunghezza > 0 else 0.0
            risultato.append([round(la1 + f * (la2 - la1), 6), round(lo1 + f * (lo2 - lo1), 6),
                              round(angolo)])
            j += 1
        percorsi_m += lunghezza
    return risultato


def _come_lista(valore: str | None) -> list[str]:
    """OSMnx salva le liste come testo: "['Via A', 'Via B']" -> ['Via A', 'Via B']."""
    if not valore:
        return []
    if valore.startswith("["):
        return list(ast.literal_eval(valore))
    return [valore]


def vie_del_nodo(G: nx.MultiDiGraph, osm: int) -> str:
    """
    Nomi delle strade che toccano il nodo, es. "Via Paduina / Via Scipio Slataper".
    Un arco con più nomi è un tratto semplificato che attraversa più strade, anche
    lontane dal nodo: i suoi nomi si usano solo se non c'è niente di meglio.
    """
    archi = list(G.in_edges(osm, data=True)) + list(G.out_edges(osm, data=True))
    singoli, multipli, tipi = [], [], []
    for _, _, dati in archi:
        nomi = _come_lista(dati.get("name"))
        destinazione = singoli if len(nomi) == 1 else multipli
        for nome in nomi:
            if nome not in destinazione:
                destinazione.append(nome)
        for tipo in _come_lista(dati.get("highway")):
            if tipo not in tipi:
                tipi.append(tipo)
    nomi = singoli or multipli
    if nomi:
        return " / ".join(nomi)
    return f"strada senza nome ({', '.join(tipi)})"


# ----------------------------------------------------------------------
# Rotte dei veicoli e pooling
# ----------------------------------------------------------------------
# Gli eventi di una rotta si scrivono come li stampa results.descrivi():
# +i = sale l'utente i, -i = scende l'utente i. Es. [16, 10, -16, 20, -20, -10].

def gruppi_di_pooling(eventi: list[int]) -> list[list[int]]:
    """
    Divide la rotta nei tratti fra due momenti in cui il veicolo e' vuoto.
    [16, 10, -16, 20, -20, -10, 3, -3]  ->  [[16, 10, 20], [3]]
    """
    gruppi, corrente, a_bordo = [], [], 0
    for e in eventi:
        if e > 0:  # sale qualcuno
            corrente.append(e)
            a_bordo += 1
        else:  # scende qualcuno
            a_bordo -= 1
            if a_bordo == 0:  # veicolo vuoto: il gruppo si chiude
                gruppi.append(corrente)
                corrente = []
    return gruppi


def compagni_di_viaggio(eventi: list[int]) -> dict[int, list[int]]:
    """
    Per ogni utente, chi e' stato a bordo con lui almeno per un tratto.
    [16, 10, -16, 20, -20, -10]  ->  {16: [10], 10: [16, 20], 20: [10]}
    """
    compagni: dict[int, list[int]] = {}
    a_bordo: list[int] = []
    for e in eventi:
        if e > 0:
            compagni[e] = list(a_bordo)  # chi c'e' gia' quando sale
            for altro in a_bordo:
                compagni[altro].append(e)  # e lui diventa compagno di loro
            a_bordo.append(e)
        else:
            a_bordo.remove(-e)
    return compagni


def risolvi_rotte(grafo, obiettivo: str, time_limit: float) -> tuple[str, list[dict]]:
    """
    Risolve Model II con l'obiettivo dato (stessi moduli di esperimenti.py, MIPGap = 0)
    e restituisce (stato, rotte). Di ogni rotta tiene, posizione per posizione:
        sequenza  evento (+i sale, -i scende, 0 deposito)
        localita  id fisico del nodo (0 e 2n+1 per il deposito)
        schedule  orario di inizio servizio (schedule minimo, results.py)
    e il costo della rotta secondo il modello.
    """
    # import qui dentro: senza Gurobi il resto dello script funziona lo stesso
    from objectives import costruisci_con_obiettivo
    from results import risolvi

    m = costruisci_con_obiettivo(grafo, obiettivo, variante="II", log=False,
                                 time_limit=time_limit, mip_gap=0.0)
    r = risolvi(m)
    rotte = [{"sequenza": [v[0] for v in rotta.nodi], "localita": list(rotta.localita),
              "schedule": list(rotta.schedule), "costo": rotta.costo}
             for rotta in r.rotte]
    return r.stato, rotte


def tratti_della_rotta(G: nx.MultiDiGraph, per_id: dict, rotta: dict,
                       colore: dict[int, str]) -> list[dict]:
    """
    Divide la rotta nei tratti fra due fermate consecutive. Ogni tratto e' un arco del
    modello, cioe' il percorso piu' breve sulla rete, e porta con se' chi e' a bordo
    mentre lo si percorre (e quindi il suo colore).
    """
    sequenza, localita = rotta["sequenza"], rotta["localita"]
    tratti, a_bordo = [], []
    for pos in range(len(localita) - 1):
        evento = sequenza[pos]  # cosa succede alla fermata da cui parte il tratto
        if evento > 0:
            a_bordo.append(evento)
        elif evento < 0:
            a_bordo.remove(-evento)
        da, a = localita[pos], localita[pos + 1]
        km, punti = percorso_piu_breve(G, int(per_id[da]["osm"]), int(per_id[a]["osm"]))
        tratti.append({
            "da": da, "a": a, "km": round(km, 4), "a_bordo": list(a_bordo),
            "colore": colore[a_bordo[0]] if a_bordo else GRIGIO_VUOTO,
            "percorso": punti, "frecce": frecce(punti),
        })
    return tratti


def viaggi_della_rotta(k: int, rotta: dict, tratti: list[dict], per_id: dict) -> dict[int, dict]:
    """
    Il viaggio di ogni passeggero della rotta, dalla fermata di salita a quella di discesa.
    Il tempo a bordo si misura come nel vincolo del paper: B(discesa) - (B(salita) + s),
    quindi comprende guida, soste alle fermate intermedie ed eventuali attese.
    """
    sequenza, localita, orari = rotta["sequenza"], rotta["localita"], rotta["schedule"]
    salita = {e: pos for pos, e in enumerate(sequenza) if e > 0}
    discesa = {-e: pos for pos, e in enumerate(sequenza) if e < 0}
    viaggi = {}
    for i, p in salita.items():
        d = discesa[i]
        sosta = float(per_id[localita[p]]["servizio"])
        limite = per_id[i].get("L")  # L_i sta sul nodo di pickup (id = i)
        viaggi[i] = {
            "veicolo": k,
            "km": round(sum(t["km"] for t in tratti[p:d]), 4),  # i tratti da p a d-1
            "minuti": round(orari[d] - orari[p] - sosta, 2),
            "salita": round(orari[p], 2), "discesa": round(orari[d], 2),
            "limite": None if limite is None else float(limite),
        }
    return viaggi


def soluzione_da_rotte(chiave: str, stato: str, rotte: list[dict], G: nx.MultiDiGraph,
                       per_id: dict, n: int) -> dict:
    """Traduce le rotte nei dati per la mappa: tratti, gruppi, colori, viaggi, rifiutate."""
    veicoli, colore, compagni, viaggi = [], {}, {}, {}
    numero_gruppo = 0
    for k, rotta in enumerate(rotte, start=1):
        eventi = [e for e in rotta["sequenza"] if e != 0]
        gruppi = gruppi_di_pooling(eventi)
        for gruppo in gruppi:  # prima i colori, poi i tratti
            tinta = PALETTE[numero_gruppo % len(PALETTE)]  # oltre 20 gruppi si ripetono
            numero_gruppo += 1
            for i in gruppo:
                colore[i] = tinta
        compagni.update(compagni_di_viaggio(eventi))
        tratti = tratti_della_rotta(G, per_id, rotta, colore)
        viaggi.update(viaggi_della_rotta(k, rotta, tratti, per_id))
        km = sum(t["km"] for t in tratti)
        veicoli.append({
            "k": k, "eventi": eventi, "gruppi": gruppi, "tratti": tratti,
            "km": round(km, 4), "costo_modello": round(rotta["costo"], 4),
            # ogni tratto e' arrotondato a 4 decimali: 1 metro di margine per tratto
            "coincide": abs(km - rotta["costo"]) <= TOLLERANZA_KM * len(tratti),
        })
    rifiutate = [i for i in range(1, n + 1) if i not in colore]
    for i in rifiutate:
        colore[i] = GRIGIO
    return {"chiave": chiave, "etichetta": ETICHETTE.get(chiave, chiave), "calcolata": True,
            "stato": stato, "messaggio": "", "veicoli": veicoli, "rifiutate": rifiutate,
            "colore": colore, "compagni": compagni, "viaggi": viaggi}


def soluzione_assente(chiave: str, etichetta: str, messaggio: str, n: int) -> dict:
    """Nessuna rotta disponibile: ogni richiesta ha il suo colore, nessun veicolo."""
    return {"chiave": chiave, "etichetta": etichetta, "calcolata": False, "stato": "",
            "messaggio": messaggio,
            "veicoli": [{"k": None, "eventi": [], "gruppi": [[i] for i in range(1, n + 1)],
                         "tratti": [], "km": 0.0, "costo_modello": None, "coincide": True}],
            "rifiutate": [],
            "colore": {i: PALETTE[(i - 1) % len(PALETTE)] for i in range(1, n + 1)},
            "compagni": {}, "viaggi": {}}


# ----------------------------------------------------------------------
# Dati di un'istanza per la mappa
# ----------------------------------------------------------------------

def dati_istanza(percorso_json: Path, G: nx.MultiDiGraph, obiettivi: list[str],
                 risolvere: bool, time_limit: float) -> dict:
    """Dati di un'istanza per la mappa: deposito, richieste con percorso diretto, una soluzione per obiettivo."""
    dati = json.loads(percorso_json.read_text(encoding="utf-8"))
    n = int(dati["n"])
    velocita = float(dati["velocita_kmh"])
    matrice = dati["costo_km"]
    posizione = {int(v["id"]): p for p, v in enumerate(dati["nodi"])}  # la matrice e' per posizione
    per_id = {int(v["id"]): v for v in dati["nodi"]}

    mancanti = [v["osm"] for v in dati["nodi"] if int(v["osm"]) not in G]
    if mancanti:
        raise SystemExit(f"{percorso_json.name}: i nodi OSM {mancanti} non sono nella rete. "
                         f"L'istanza e' stata generata da {dati.get('rete')}?")

    # nodi fisici che cadono nello stesso punto (stesso nodo OSM): i marker si sovrappongono
    stesso_osm: dict[int, list[int]] = {}
    for v in dati["nodi"]:
        stesso_osm.setdefault(int(v["osm"]), []).append(int(v["id"]))

    def scheda_nodo(id_nodo: int) -> dict:
        v = per_id[id_nodo]
        osm = int(v["osm"])
        return {"id": id_nodo, "osm": osm, "lat": v["lat"], "lon": v["lon"],
                "vie": vie_del_nodo(G, osm),
                "stesso_punto": [j for j in stesso_osm[osm] if j != id_nodo]}

    # --- richieste: percorso diretto e verifica con la matrice ---
    richieste = []
    for i in range(1, n + 1):
        pickup, dropoff = scheda_nodo(i), scheda_nodo(n + i)
        km, punti = percorso_piu_breve(G, pickup["osm"], dropoff["osm"])
        km_matrice = float(matrice[posizione[i]][posizione[n + i]])
        richieste.append({
            "i": i, "pickup": pickup, "dropoff": dropoff,
            "km": round(km, 4), "km_matrice": km_matrice,
            "coincide": abs(km - km_matrice) <= TOLLERANZA_KM,
            "minuti": round(km * 60 / velocita, 2),
            "percorso": punti,
        })

    # il deposito compare due volte (id 0 e 2n+1) ma e' lo stesso luogo
    deposito = scheda_nodo(0)
    deposito["stesso_punto"] = []
    deposito["id_finale"] = 2 * n + 1

    # --- soluzioni (una per obiettivo) ---
    soluzioni = []
    if not risolvere:
        soluzioni.append(soluzione_assente("nessuna", "nessuna soluzione",
                                           "Soluzione non calcolata: solo percorsi diretti, "
                                           "un colore per richiesta", n))
    else:
        from instances import leggi_istanza
        from graph import Grafo
        grafo = Grafo.costruisci(leggi_istanza(percorso_json))  # non dipende dall'obiettivo
        for ob in obiettivi:
            try:
                stato, rotte = risolvi_rotte(grafo, ob, time_limit)
            except Exception as errore:  # licenza, memoria, ...: la mappa si fa lo stesso
                soluzioni.append(soluzione_assente(ob, ETICHETTE.get(ob, ob),
                                                   f"Errore del solver: {errore}", n))
                continue
            if not rotte:
                soluzioni.append(soluzione_assente(ob, ETICHETTE.get(ob, ob),
                                                   f"Nessuna soluzione trovata ({stato})", n))
            else:
                soluzioni.append(soluzione_da_rotte(ob, stato, rotte, G, per_id, n))

    return {"nome": dati["nome"], "n": n, "K": int(dati["K"]), "Q": int(dati["Q"]),
            "velocita": velocita, "deposito": deposito, "richieste": richieste,
            "soluzioni": soluzioni}


def riassunto(ist: dict) -> str:
    """Il resoconto di un'istanza per il terminale."""
    uguali = sum(r["coincide"] for r in ist["richieste"])
    righe = [f"{ist['nome']}: {uguali}/{ist['n']} percorsi diretti coincidono con costo_km"]
    for s in ist["soluzioni"]:
        if not s["calcolata"]:
            righe.append(f"    {s['chiave']}: {s['messaggio']}")
            continue
        gruppi = [g for v in s["veicoli"] for g in v["gruppi"]]
        in_pooling = sum(len(g) for g in gruppi if len(g) > 1)
        rotte_ok = sum(v["coincide"] for v in s["veicoli"])
        righe.append(f"    {s['chiave']}: {s['stato']}, {len(s['veicoli'])} veicoli "
                     f"({rotte_ok} rotte = costo del modello), {in_pooling} richieste in "
                     f"pooling, {len(s['rifiutate'])} rifiutate")
        for v in s["veicoli"]:
            if not v["coincide"]:
                righe.append(f"    ATTENZIONE veicolo {v['k']}: rotta {v['km']:.4f} km, "
                             f"modello {v['costo_modello']:.4f} km")
    for r in ist["richieste"]:
        if not r["coincide"]:
            righe.append(f"    ATTENZIONE richiesta {r['i']}: percorso {r['km']:.4f} km, "
                         f"matrice {r['km_matrice']:.4f} km")
    return "\n".join(righe)


# ----------------------------------------------------------------------
# Programma principale
# ----------------------------------------------------------------------

def espandi(nomi: list[str]) -> list[Path]:
    """Espande gli asterischi (il terminale di Windows non lo fa) e ordina i file."""
    percorsi = []
    for nome in nomi:
        trovati = sorted(glob.glob(nome)) if any(c in nome for c in "*?") else [nome]
        if not trovati:
            raise SystemExit(f"nessun file corrisponde a {nome}")
        percorsi += [Path(p) for p in trovati]
    for p in percorsi:
        if not p.is_file():
            raise SystemExit(f"file inesistente: {p}")
    return percorsi


def gurobi_disponibile() -> bool:
    try:
        import gurobipy  # serve solo sapere se è installato
        return True
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Mappa interattiva delle istanze Trieste.")
    parser.add_argument("istanze", nargs="+", help="file JSON delle istanze (si puo' usare *)")
    parser.add_argument("--rete", type=Path, required=True, help="file .graphml della rete")
    parser.add_argument("--obiettivi", nargs="+", default=["fc"], choices=list(ETICHETTE),
                        help="obiettivi da risolvere, uno per voce del menu (default: fc)")
    parser.add_argument("--senza-soluzione", action="store_true",
                        help="non risolvere: solo percorsi diretti, niente rotte")
    parser.add_argument("--time-limit", type=float, default=60.0,
                        help="secondi per ogni risoluzione (default 60)")
    parser.add_argument("--out", type=Path, default=Path("mappa_trieste.html"))
    args = parser.parse_args()

    percorsi = espandi(args.istanze)
    risolvere = not args.senza_soluzione
    if risolvere and not gurobi_disponibile():
        print("gurobipy non trovato: mappa senza rotte (solo percorsi diretti)")
        risolvere = False

    print(f"lettura della rete {args.rete} ...")
    G = carica_rete(args.rete)
    print(f"  {G.number_of_nodes()} nodi, {G.number_of_edges()} archi\n")

    istanze = []
    for p in percorsi:
        ist = dati_istanza(p, G, args.obiettivi, risolvere, args.time_limit)
        print(riassunto(ist))
        istanze.append(ist)

    html = MODELLO_HTML.replace("__DATI__", json.dumps({"istanze": istanze},
                                                       ensure_ascii=False,
                                                       separators=(",", ":")))
    args.out.write_text(html, encoding="utf-8")
    print(f"\nfatto: {args.out.resolve()}")


# ----------------------------------------------------------------------
# Pagina HTML (Leaflet). __DATI__ viene sostituito dai dati calcolati sopra.
# ----------------------------------------------------------------------

MODELLO_HTML = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Istanze di Trieste: rotte e pooling</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Semi+Condensed:wght@600;700&display=swap">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root {
  --carta: #FAFBFB;
  --inchiostro: #1B2A33;
  --tenue: #5B6975;
  --riga: #DCE2E6;
  --adriatico: #0D5C75;
  --adriatico-chiaro: #E3EFF3;
  --ok: #1E7A4C;
  --ko: #B3261E;
  --testo: "Barlow", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --stretto: "Barlow Semi Condensed", "Barlow", "Arial Narrow", sans-serif;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  display: flex; font-family: var(--testo); font-size: 15px; line-height: 1.45;
  color: var(--inchiostro); background: var(--carta);
}
#pannello {
  width: 380px; flex: none; height: 100%; display: flex; flex-direction: column;
  border-right: 1px solid var(--riga); background: var(--carta);
}
#mappa { flex: 1; height: 100%; background: #E8ECEE; }

.testa { padding: 18px 20px 14px; border-bottom: 1px solid var(--riga); }
h1 { font-family: var(--stretto); font-weight: 700; font-size: 26px; line-height: 1.1; margin: 0; }
.sottotitolo { margin: 4px 0 14px; color: var(--tenue); font-size: 14px; }

.scelta-istanza { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.scelta-istanza button {
  font: 600 14px/1 var(--stretto); padding: 7px 10px; border-radius: 6px; cursor: pointer;
  color: var(--inchiostro); background: #fff; border: 1px solid var(--riga);
  font-variant-numeric: tabular-nums;
}
.scelta-istanza button[aria-pressed="true"] {
  background: var(--adriatico); border-color: var(--adriatico); color: #fff;
}
label.obiettivo { display: block; font-size: 13px; color: var(--tenue); margin-bottom: 10px; }
label.obiettivo select {
  display: block; width: 100%; margin-top: 4px; font: 500 15px var(--testo);
  padding: 6px 8px; border: 1px solid var(--riga); border-radius: 6px; background: #fff;
  color: var(--inchiostro);
}
.fatti { margin: 0; font-size: 14px; }
.fatti p { margin: 2px 0; }
.verifica { font-weight: 600; }
.verifica.ok { color: var(--ok); }
.verifica.ko { color: var(--ko); }
.avviso { color: var(--ko); }

.legenda { padding: 10px 20px; border-bottom: 1px solid var(--riga); font-size: 13px; color: var(--tenue); }
.legenda .fila { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.legenda .fila + .fila { margin-top: 6px; }
.legenda span { display: inline-flex; align-items: center; gap: 6px; }
.legenda i { display: inline-block; width: 16px; height: 12px; background: var(--tenue); }
.legenda .l-pickup i { border-radius: 6px; }
.legenda .l-dropoff i { border-radius: 2px; }
.legenda .l-deposito i { background: var(--inchiostro); width: 13px; height: 13px; border-radius: 3px; }
.legenda svg { display: block; }
.legenda [hidden] { display: none; }
.interruttore { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; color: var(--inchiostro); }
.interruttore input { accent-color: var(--adriatico); width: 16px; height: 16px; margin: 0; }
#mostra-tutto {
  margin-left: auto; font: 500 13px var(--testo); color: var(--adriatico); background: none;
  border: 0; padding: 2px 0; cursor: pointer; text-decoration: underline;
}
#mostra-tutto[hidden] { display: none; }

#elenco { flex: 1; overflow-y: auto; padding: 4px 0 16px; }
.veicolo { padding: 10px 20px 4px; }
.veicolo h2 {
  font: 700 16px/1.2 var(--stretto); margin: 0; display: flex; align-items: baseline;
  justify-content: space-between; gap: 10px;
}
.veicolo h2 small { font: 500 13px var(--testo); color: var(--tenue); font-variant-numeric: tabular-nums; }
.nome-veicolo {
  font: inherit; color: inherit; background: none; border: 0; padding: 2px 6px; margin-left: -6px;
  border-radius: 5px; cursor: pointer;
}
.nome-veicolo:hover { background: #EEF2F4; }
.nome-veicolo[aria-pressed="true"] { background: var(--adriatico); color: #fff; }
.eventi {
  margin: 2px 0 8px; font-size: 13px; color: var(--tenue); word-spacing: 2px;
  font-variant-numeric: tabular-nums;
}
.gruppo { border-left: 4px solid var(--c); margin: 0 0 8px; padding-left: 8px; }
.gruppo .etichetta { font-size: 12px; color: var(--tenue); margin: 0 0 2px 6px; }
.riga {
  display: grid; grid-template-columns: 30px minmax(0, 1fr) auto; gap: 10px; align-items: center;
  width: 100%; text-align: left; padding: 6px; border: 0; border-radius: 6px;
  background: none; font: inherit; color: inherit; cursor: pointer;
}
.riga:hover { background: #EEF2F4; }
.riga[aria-current="true"] { background: var(--adriatico-chiaro); }
.segno {
  display: inline-flex; align-items: center; justify-content: center; height: 22px;
  border-radius: 11px; font: 700 13px/1 var(--stretto); background: var(--c); color: var(--t);
}
.nome { font-weight: 600; display: block; }
.vie { display: block; font-size: 13px; color: var(--tenue); overflow: hidden;
       text-overflow: ellipsis; white-space: nowrap; }
.misure { text-align: right; font-size: 13px; color: var(--tenue); font-variant-numeric: tabular-nums;
          white-space: nowrap; }
.misure b { display: block; font-weight: 600; font-size: 14px; color: var(--inchiostro); }
.aiuto { padding: 10px 20px; border-top: 1px solid var(--riga); font-size: 13px; color: var(--tenue); margin: 0; }

button:focus-visible, select:focus-visible, a:focus-visible, input:focus-visible {
  outline: 2px solid var(--adriatico); outline-offset: 2px;
}

/* marker sulla mappa: pickup = pillola, dropoff = quadrato, deposito = scuro */
.icona-nodo { background: none; border: 0; }
.nodo {
  position: absolute; transform: translate(-50%, -50%);
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 32px; height: 22px; padding: 0 6px; white-space: nowrap; cursor: pointer;
  font: 700 13px/1 var(--stretto); background: var(--c); color: var(--t);
  border: 2px solid #fff; box-shadow: 0 1px 4px rgba(27, 42, 51, .45);
}
.nodo.pickup { border-radius: 11px; }
.nodo.dropoff { border-radius: 3px; }
.nodo.deposito { width: 30px; min-width: 0; height: 30px; padding: 0; border-radius: 5px;
                 background: var(--inchiostro); color: #fff; }
.nodo.deposito svg { width: 18px; height: 18px; }
/* frecce del verso di marcia */
.freccia { position: absolute; width: 13px; height: 13px; display: block; pointer-events: none; }
.freccia svg { width: 13px; height: 13px; display: block; }
.freccia path { fill: var(--c); stroke: #fff; stroke-width: 1.4; stroke-linejoin: round; }

/* schede che si aprono cliccando un nodo o un tratto */
.leaflet-popup-content-wrapper { border-radius: 8px; }
.leaflet-popup-content { margin: 14px 16px; font: 14px/1.45 var(--testo); color: var(--inchiostro); min-width: 270px; }
.scheda h3 { font: 700 18px/1.2 var(--stretto); margin: 0; display: flex; align-items: center; gap: 8px; }
.scheda h3 .segno { min-width: 34px; padding: 0 6px; }
.scheda .ruolo { margin: 2px 0 10px; color: var(--tenue); }
.scheda .ruolo a { color: var(--adriatico); }
.scheda dl { display: grid; grid-template-columns: auto 1fr; gap: 3px 12px; margin: 0 0 10px; }
.scheda dt { color: var(--tenue); }
.scheda dd { margin: 0; }
.scheda dd b { font-variant-numeric: tabular-nums; }
.scheda p { margin: 4px 0 0; }
.scheda .pooling { padding-top: 8px; border-top: 1px solid var(--riga); margin-top: 8px; }

@media (max-width: 760px) {
  body { flex-direction: column-reverse; }
  #pannello { width: 100%; height: 48%; border-right: 0; border-top: 1px solid var(--riga); }
  #mappa { flex: none; height: 52%; }
}
</style>
</head>
<body>
<aside id="pannello">
  <div class="testa">
    <h1>Istanze di Trieste</h1>
    <p class="sottotitolo">Rotte e percorsi sulla rete stradale OSM</p>
    <div class="scelta-istanza" id="scelta-istanza" role="group" aria-label="Istanza"></div>
    <div id="scelta-obiettivo"></div>
    <div class="fatti" id="fatti"></div>
  </div>
  <div class="legenda">
    <div class="fila">
      <span class="l-pickup"><i></i>pickup</span>
      <span class="l-dropoff"><i></i>dropoff</span>
      <span class="l-deposito"><i></i>deposito</span>
      <button id="mostra-tutto" type="button" hidden>Mostra tutto</button>
    </div>
    <div class="fila" id="legenda-rotte" hidden>
      <span>a bordo:</span>
      <span><svg width="26" height="12" aria-hidden="true"><line x1="2" y1="6" x2="24" y2="6" stroke="#7D868C" stroke-width="3" stroke-dasharray="5 4"/></svg>0</span>
      <span><svg width="26" height="12" aria-hidden="true"><line x1="2" y1="6" x2="24" y2="6" stroke="#5B6975" stroke-width="4" stroke-linecap="round"/></svg>1</span>
      <span><svg width="26" height="12" aria-hidden="true"><line x1="3" y1="6" x2="23" y2="6" stroke="#5B6975" stroke-width="7" stroke-linecap="round"/></svg>2</span>
      <span><svg width="26" height="12" aria-hidden="true"><line x1="5" y1="6" x2="21" y2="6" stroke="#5B6975" stroke-width="10" stroke-linecap="round"/></svg>3</span>
    </div>
    <div class="fila" id="fila-diretti" hidden>
      <label class="interruttore"><input type="checkbox" id="diretti">
        <svg width="26" height="12" aria-hidden="true"><line x1="2" y1="6" x2="24" y2="6" stroke="#5B6975" stroke-width="3" stroke-dasharray="0.5 5" stroke-linecap="round"/></svg>
        percorsi diretti pickup → dropoff</label>
    </div>
  </div>
  <div id="elenco"></div>
  <p class="aiuto" id="aiuto"></p>
</aside>
<div id="mappa" role="region" aria-label="Mappa di Trieste"></div>

<script>
const DATI = __DATI__;
const riduciMovimento = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const CASA = '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 3 2 11h3v9h5v-6h4v6h5v-9h3z"/></svg>';
const FRECCIA = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 1 11 10.5 6 8 1 10.5Z"/></svg>';

// ---------- piccoli aiuti ----------
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}
function num(x, cifre) {
  return x.toLocaleString("it-IT", {minimumFractionDigits: cifre, maximumFractionDigits: cifre});
}
function elenca(v) {            // [7] -> "7",  [7, 12] -> "7 e 12",  [3, 7, 12] -> "3, 7 e 12"
  return v.length < 2 ? v.join("") : v.slice(0, -1).join(", ") + " e " + v[v.length - 1];
}
function richieste(v) {        // [7] -> "richiesta 7",  [7, 12] -> "richieste 7 e 12"
  return (v.length === 1 ? "richiesta " : "richieste ") + elenca(v);
}
function testoSu(hex) {         // testo scuro sui colori chiari, bianco su quelli scuri
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return (0.299 * r + 0.587 * g + 0.114 * b) > 150 ? "#1B2A33" : "#FFFFFF";
}
function breve(vie) { return vie.split(" / ")[0]; }
function nomeCorto(nome) { return nome.replace(/^Trieste_/, ""); }
function pesoTratto(aBordo) { return aBordo === 0 ? 3 : Math.min(1 + 3 * aBordo, 13); }

// ---------- stato ----------
// sel: null, { tipo: "richiesta", i } oppure { tipo: "veicolo", k }
const stato = { istanza: 0, obiettivo: null, sel: null, diretti: false };
let disegno = { richieste: {}, tratti: [] };
let indice = {};    // i -> { veicolo, gruppo, rifiutata }
let nodi = {};      // id del nodo -> { vie, ... } per le schede dei tratti

function istanza() { return DATI.istanze[stato.istanza]; }
function soluzione() {
  const ss = istanza().soluzioni;
  return ss.find(s => s.chiave === stato.obiettivo) || ss[0];
}
function etichettaNodo(id) {
  const n = istanza().n;
  if (id === 0 || id === 2 * n + 1) return "deposito";
  return id <= n ? "P" + id : "D" + (id - n);
}

// ---------- mappa ----------
const mappa = L.map("mappa", { zoomSnap: 0.5 });
const attribuzione = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>';
const stradale = L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
  { subdomains: "abcd", maxZoom: 20, attribution: attribuzione }).addTo(mappa);
const chiara = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
  { subdomains: "abcd", maxZoom: 20, attribution: attribuzione });
L.control.layers({ "Mappa stradale": stradale, "Mappa chiara": chiara }, null, { position: "topright" }).addTo(mappa);
L.control.scale({ imperial: false }).addTo(mappa);
const strato = L.layerGroup().addTo(mappa);      // rotte, marker, percorsi delle rifiutate
const stratoDiretti = L.layerGroup();             // percorsi diretti: acceso dall'interruttore

function iconaNodo(testo, classe, colore) {
  return L.divIcon({
    className: "icona-nodo", iconSize: [0, 0], popupAnchor: [0, -14],
    html: `<span class="nodo ${classe}" style="--c:${colore};--t:${testoSu(colore)}">${testo}</span>`
  });
}
function iconaFreccia(angolo, colore) {
  return L.divIcon({
    className: "icona-nodo", iconSize: [0, 0],
    html: `<span class="freccia" style="--c:${colore};transform:translate(-50%,-50%) rotate(${angolo}deg)">${FRECCIA}</span>`
  });
}

// ---------- schede ----------
function schedaRichiesta(r, ruolo) {
  const ist = istanza(), sol = soluzione(), info = indice[r.i], colore = sol.colore[r.i];
  const nodo = ruolo === "pickup" ? r.pickup : r.dropoff;
  const lettera = ruolo === "pickup" ? "P" : "D";
  const viaggio = sol.calcolata ? sol.viaggi[r.i] : null;
  let h = `<div class="scheda"><h3><span class="segno" style="--c:${colore};--t:${testoSu(colore)}">${lettera}${r.i}</span>Nodo ${nodo.id}</h3>`;
  h += `<p class="ruolo">${ruolo} della richiesta ${r.i}, <a href="https://www.openstreetmap.org/node/${nodo.osm}" target="_blank" rel="noopener">nodo OSM ${nodo.osm}</a></p>`;
  h += "<dl>";
  h += `<dt>Pickup</dt><dd>nodo ${r.pickup.id}, ${esc(r.pickup.vie)}</dd>`;
  h += `<dt>Dropoff</dt><dd>nodo ${r.dropoff.id}, ${esc(r.dropoff.vie)}</dd>`;
  if (viaggio) {
    h += `<dt>A bordo</dt><dd><b>${num(viaggio.km, 2)} km, ${num(viaggio.minuti, 1)} min</b></dd>`;
    h += `<dt>Diretto</dt><dd>${num(r.km, 2)} km, ${num(r.minuti, 1)} min a ${num(ist.velocita, 0)} km/h</dd>`;
    if (viaggio.limite != null) h += `<dt>Massimo</dt><dd>${num(viaggio.limite, 1)} min a bordo (L<sub>${r.i}</sub>)</dd>`;
    h += `<dt>Orari</dt><dd>sale al minuto ${num(viaggio.salita, 1)}, scende al ${num(viaggio.discesa, 1)}</dd>`;
  } else {
    h += `<dt>Tragitto</dt><dd><b>${num(r.km, 2)} km</b></dd>`;
    h += `<dt>A ${num(ist.velocita, 0)} km/h</dt><dd><b>${num(r.minuti, 1)} min</b></dd>`;
  }
  h += "</dl>";
  const cosa = viaggio ? "Percorso diretto uguale" : "Uguale";
  h += r.coincide
    ? `<p class="verifica ok">✓ ${cosa} a costo_km dell'istanza (${num(r.km_matrice, 4)} km)</p>`
    : `<p class="verifica ko">✗ Percorso diretto diverso da costo_km: ${num(r.km, 4)} km contro ${num(r.km_matrice, 4)} km</p>`;
  if (nodo.stesso_punto.length) {
    h += `<p>Nello stesso punto anche: nodo ${elenca(nodo.stesso_punto)}</p>`;
  }
  if (sol.calcolata) {
    h += '<div class="pooling">';
    if (info.rifiutata) {
      h += "<p>Richiesta rifiutata dal modello</p>";
    } else {
      const compagni = sol.compagni[r.i] || [];
      h += compagni.length
        ? `<p>Veicolo ${info.veicolo}, a bordo insieme a: ${richieste(compagni)}</p>`
        : `<p>Veicolo ${info.veicolo}, viaggia da sola</p>`;
      const altri = info.gruppo.filter(j => j !== r.i && !compagni.includes(j));
      if (altri.length) {
        h += `<p>Nello stesso gruppo di pooling anche ${richieste(altri)}: a bordo con gli altri del gruppo, mai insieme a questa</p>`;
      }
    }
    h += "</div>";
  }
  return h + "</div>";
}

function schedaTratto(v, j) {
  const t = v.tratti[j], ist = istanza();
  const fermata = id => `${etichettaNodo(id)}, nodo ${id}, ${esc(nodi[id].vie)}`;
  let h = `<div class="scheda"><h3>Veicolo ${v.k}</h3><p class="ruolo">tratto ${j + 1} di ${v.tratti.length}</p><dl>`;
  h += `<dt>Da</dt><dd>${fermata(t.da)}</dd><dt>A</dt><dd>${fermata(t.a)}</dd>`;
  h += `<dt>Lunghezza</dt><dd><b>${num(t.km, 2)} km</b></dd>`;
  h += `<dt>A ${num(ist.velocita, 0)} km/h</dt><dd><b>${num(t.km * 60 / ist.velocita, 1)} min</b></dd></dl>`;
  h += `<p class="pooling">${t.a_bordo.length ? "A bordo: " + richieste(t.a_bordo) : "Veicolo vuoto"}</p></div>`;
  return h;
}

function schedaDeposito(d) {
  return `<div class="scheda"><h3><span class="nodo deposito" style="position:static;transform:none;box-shadow:none">${CASA}</span>Deposito</h3>`
    + `<p class="ruolo">nodi ${d.id} e ${d.id_finale}, <a href="https://www.openstreetmap.org/node/${d.osm}" target="_blank" rel="noopener">nodo OSM ${d.osm}</a></p>`
    + `<dl><dt>Via</dt><dd>${esc(d.vie)}</dd></dl></div>`;
}

// ---------- disegno di istanza + soluzione ----------
function costruisciIndice() {
  indice = {};
  const sol = soluzione();
  for (const v of sol.veicoli) {
    for (const g of v.gruppi) for (const i of g) indice[i] = { veicolo: v.k, gruppo: g, rifiutata: false };
  }
  for (const i of sol.rifiutate) indice[i] = { veicolo: null, gruppo: [i], rifiutata: true };
}

function disegna(inquadra) {
  strato.clearLayers();
  stratoDiretti.clearLayers();
  disegno = { richieste: {}, tratti: [] };
  stato.sel = null;
  costruisciIndice();
  const ist = istanza(), sol = soluzione();
  nodi = {};
  for (const r of ist.richieste) { nodi[r.pickup.id] = r.pickup; nodi[r.dropoff.id] = r.dropoff; }
  nodi[ist.deposito.id] = ist.deposito;
  nodi[ist.deposito.id_finale] = ist.deposito;

  // 1. percorsi diretti. Senza soluzione sono il disegno principale (con il bordo bianco);
  //    con la soluzione servono solo per le rifiutate e, a richiesta, per il confronto.
  for (const r of ist.richieste) {
    const tipo = !sol.calcolata ? "principale" : (indice[r.i].rifiutata ? "rifiutata" : "diretto");
    disegno.richieste[r.i] = { tipo };
    if (tipo === "principale") {
      disegno.richieste[r.i].bordo = L.polyline(r.percorso, { color: "#FFFFFF", weight: 7, opacity: 0.85, interactive: false }).addTo(strato);
    }
  }
  for (const r of ist.richieste) {
    const d = disegno.richieste[r.i];
    d.linea = L.polyline(r.percorso, {
      color: sol.colore[r.i], weight: d.tipo === "principale" ? 4 : 3, opacity: 0.9, lineCap: "round",
      dashArray: { principale: null, rifiutata: "2 7", diretto: "1 8" }[d.tipo], bubblingMouseEvents: false
    }).addTo(d.tipo === "diretto" ? stratoDiretti : strato);
    d.linea.on("click", () => seleziona({ tipo: "richiesta", i: r.i }, { popup: "pickup" }));
  }

  // 2. rotte dei veicoli: prima tutti i bordi bianchi, poi le linee, poi le frecce
  if (sol.calcolata) {
    for (const v of sol.veicoli) v.tratti.forEach((t, j) => {
      if (t.percorso.length < 2) return;       // due fermate nello stesso punto: niente da disegnare
      const peso = pesoTratto(t.a_bordo.length);
      disegno.tratti.push({ v, j, t, peso,
        bordo: L.polyline(t.percorso, { color: "#FFFFFF", weight: peso + 3, opacity: 0.85, interactive: false }).addTo(strato) });
    });
    for (const d of disegno.tratti) {
      d.linea = L.polyline(d.t.percorso, {
        color: d.t.colore, weight: d.peso, opacity: 0.9, lineCap: "round", lineJoin: "round",
        dashArray: d.t.a_bordo.length ? null : "7 7", bubblingMouseEvents: false
      }).addTo(strato);
      d.linea.bindPopup(() => schedaTratto(d.v, d.j), { maxWidth: 340 });
      d.linea.on("click", () => seleziona({ tipo: "veicolo", k: d.v.k }));
    }
    for (const d of disegno.tratti) {
      d.frecce = d.t.frecce.map(([lat, lon, angolo]) => L.marker([lat, lon], {
        icon: iconaFreccia(angolo, d.t.colore), interactive: false, keyboard: false, zIndexOffset: -1000
      }).addTo(strato));
    }
  }

  // 3. marker di pickup, dropoff e deposito
  for (const r of ist.richieste) {
    const colore = sol.colore[r.i], d = disegno.richieste[r.i];
    d.pickup = L.marker([r.pickup.lat, r.pickup.lon], { icon: iconaNodo("P" + r.i, "pickup", colore), title: `Pickup ${r.i}`, keyboard: true })
      .bindPopup(() => schedaRichiesta(r, "pickup"), { maxWidth: 360 }).addTo(strato);
    d.dropoff = L.marker([r.dropoff.lat, r.dropoff.lon], { icon: iconaNodo("D" + r.i, "dropoff", colore), title: `Dropoff ${r.i}`, keyboard: true })
      .bindPopup(() => schedaRichiesta(r, "dropoff"), { maxWidth: 360 }).addTo(strato);
    d.pickup.on("click", () => seleziona({ tipo: "richiesta", i: r.i }));
    d.dropoff.on("click", () => seleziona({ tipo: "richiesta", i: r.i }));
  }
  const dep = ist.deposito;
  L.marker([dep.lat, dep.lon], { icon: iconaNodo(CASA, "deposito", "#1B2A33"), title: "Deposito", zIndexOffset: 500 })
    .bindPopup(() => schedaDeposito(dep), { maxWidth: 340 }).addTo(strato);

  aggiornaDiretti();
  if (inquadra) {
    const punti = [[dep.lat, dep.lon]];
    for (const r of ist.richieste) for (const p of r.percorso) punti.push(p);
    for (const d of disegno.tratti) for (const p of d.t.percorso) punti.push(p);
    mappa.fitBounds(L.latLngBounds(punti).pad(0.05), { animate: !riduciMovimento });
  }
  pannello();
  applicaStile();
}

function aggiornaDiretti() {
  const visibili = soluzione().calcolata && stato.diretti;
  if (visibili && !mappa.hasLayer(stratoDiretti)) mappa.addLayer(stratoDiretti);
  if (!visibili && mappa.hasLayer(stratoDiretti)) mappa.removeLayer(stratoDiretti);
}

// ---------- evidenziazione ----------
// Quattro livelli: normale (niente selezionato), scelta (l'oggetto selezionato),
// veicolo (il resto della rotta del veicolo della richiesta scelta), sfondo (tutto il resto).
const STILE_TRATTO = {
  normale: { extra: 0, linea: 0.9,  bordo: 0.85, frecce: 1 },
  scelta:  { extra: 2, linea: 1,    bordo: 1,    frecce: 1 },
  veicolo: { extra: 0, linea: 0.55, bordo: 0.45, frecce: 0.55 },
  sfondo:  { extra: 0, linea: 0.12, bordo: 0,    frecce: 0.1 },
};
const OPACITA_MARKER = { normale: 1, scelta: 1, veicolo: 0.9, sfondo: 0.25 };

function livelloTratto(d) {
  const sel = stato.sel;
  if (!sel) return "normale";
  if (sel.tipo === "veicolo") return d.v.k === sel.k ? "scelta" : "sfondo";
  const info = indice[sel.i];
  if (info.rifiutata || d.v.k !== info.veicolo) return "sfondo";
  return d.t.a_bordo.includes(sel.i) ? "scelta" : "veicolo";
}

function livelloRichiesta(i) {
  const sel = stato.sel, sol = soluzione();
  if (!sel) return "normale";
  if (sel.tipo === "veicolo") return indice[i].veicolo === sel.k ? "normale" : "sfondo";
  if (i === sel.i) return "scelta";
  const info = indice[sel.i];
  if (sol.calcolata && !info.rifiutata && indice[i].veicolo === info.veicolo) return "veicolo";
  return "sfondo";
}

function applicaStile() {
  const scelti = [];
  for (const d of disegno.tratti) {
    const livello = livelloTratto(d), s = STILE_TRATTO[livello];
    const vuoto = d.t.a_bordo.length === 0;
    d.linea.setStyle({ weight: d.peso + s.extra, opacity: vuoto ? s.linea * 0.85 : s.linea });
    d.bordo.setStyle({ weight: d.peso + s.extra + 3, opacity: s.bordo });
    for (const f of d.frecce) f.setOpacity(s.frecce);
    if (livello === "scelta") scelti.push(d);
  }
  // in primo piano i tratti scelti: prima tutti i loro bordi, poi tutte le loro linee
  for (const d of scelti) d.bordo.bringToFront();
  for (const d of scelti) d.linea.bringToFront();

  for (const [chiave, d] of Object.entries(disegno.richieste)) {
    const i = Number(chiave), livello = livelloRichiesta(i);
    const base = d.tipo === "principale" ? 4 : 3;
    const opacita = { normale: 0.9, scelta: 1, veicolo: 0.6, sfondo: d.tipo === "principale" ? 0.15 : 0.12 }[livello];
    d.linea.setStyle({ weight: livello === "scelta" ? base + 2 : base, opacity: opacita });
    if (d.bordo) d.bordo.setStyle({ weight: (livello === "scelta" ? base + 2 : base) + 3, opacity: livello === "sfondo" ? 0 : 0.85 });
    if (livello === "scelta" && d.tipo !== "diretto") { if (d.bordo) d.bordo.bringToFront(); d.linea.bringToFront(); }
    d.pickup.setOpacity(OPACITA_MARKER[livello]);
    d.dropoff.setOpacity(OPACITA_MARKER[livello]);
    d.pickup.setZIndexOffset(livello === "scelta" ? 1000 : 0);
    d.dropoff.setZIndexOffset(livello === "scelta" ? 1000 : 0);
  }

  const sel = stato.sel;
  document.querySelectorAll(".riga").forEach(b =>
    b.setAttribute("aria-current", String(!!sel && sel.tipo === "richiesta" && Number(b.dataset.i) === sel.i)));
  document.querySelectorAll(".nome-veicolo").forEach(b =>
    b.setAttribute("aria-pressed", String(!!sel && sel.tipo === "veicolo" && Number(b.dataset.k) === sel.k)));
  document.getElementById("mostra-tutto").hidden = sel === null;
}

function puntiDaInquadrare(sel) {
  const punti = [];
  if (sel.tipo === "veicolo") {
    for (const d of disegno.tratti) if (d.v.k === sel.k) for (const p of d.t.percorso) punti.push(p);
    return punti;
  }
  // la richiesta: i tratti in cui e' a bordo, oppure il percorso diretto
  for (const d of disegno.tratti) if (d.t.a_bordo.includes(sel.i)) for (const p of d.t.percorso) punti.push(p);
  if (!punti.length) for (const p of istanza().richieste.find(r => r.i === sel.i).percorso) punti.push(p);
  return punti;
}

function seleziona(sel, opzioni = {}) {
  stato.sel = sel;
  applicaStile();
  if (opzioni.inquadra) {
    const punti = puntiDaInquadrare(sel);
    if (punti.length) mappa.fitBounds(L.latLngBounds(punti).pad(0.25), { animate: !riduciMovimento, maxZoom: 17 });
  }
  if (opzioni.popup && sel.tipo === "richiesta") disegno.richieste[sel.i][opzioni.popup].openPopup();
}

function deseleziona() {
  stato.sel = null;
  mappa.closePopup();
  applicaStile();
}

// ---------- pannello laterale ----------
function pannello() {
  const ist = istanza(), sol = soluzione();

  document.getElementById("scelta-istanza").innerHTML = DATI.istanze.map((x, k) =>
    `<button type="button" data-k="${k}" aria-pressed="${k === stato.istanza}">${esc(nomeCorto(x.nome))}</button>`).join("");

  const scelta = document.getElementById("scelta-obiettivo");
  if (ist.soluzioni.length > 1) {
    scelta.innerHTML = `<label class="obiettivo">Obiettivo<select id="obiettivo">`
      + ist.soluzioni.map(s => `<option value="${s.chiave}"${s.chiave === sol.chiave ? " selected" : ""}>${esc(s.etichetta)}</option>`).join("")
      + "</select></label>";
    document.getElementById("obiettivo").addEventListener("change", e => { stato.obiettivo = e.target.value; mappa.closePopup(); disegna(false); });
  } else {
    scelta.innerHTML = "";
  }

  const uguali = ist.richieste.filter(r => r.coincide).length;
  let f = `<p>${ist.n} richieste, capacità Q = ${ist.Q}, K = ${ist.K} veicoli</p>`;
  if (sol.calcolata) {
    const gruppi = sol.veicoli.flatMap(v => v.gruppi);
    const condivisi = gruppi.filter(g => g.length > 1);
    const inPooling = condivisi.reduce((s, g) => s + g.length, 0);
    f += `<p>${esc(sol.etichetta)}: ${esc(sol.stato)}, ${sol.veicoli.length} veicoli usati</p>`;
    const pooling = inPooling === 0 ? "Nessuna richiesta in pooling"
      : `${inPooling} richieste in pooling in ${condivisi.length} ${condivisi.length === 1 ? "gruppo" : "gruppi"}`;
    f += `<p>${pooling}` + (sol.rifiutate.length ? `, ${sol.rifiutate.length} rifiutate` : "") + "</p>";
    const rotteOk = sol.veicoli.filter(v => v.coincide).length;
    f += rotteOk === sol.veicoli.length
      ? `<p class="verifica ok">✓ ${rotteOk} rotte su ${sol.veicoli.length}: lunghezza uguale al costo del modello</p>`
      : `<p class="verifica ko">✗ solo ${rotteOk} rotte su ${sol.veicoli.length} hanno la lunghezza del modello</p>`;
  } else {
    f += `<p class="avviso">${esc(sol.messaggio)}</p>`;
  }
  f += uguali === ist.n
    ? `<p class="verifica ok">✓ ${uguali} percorsi diretti su ${ist.n} coincidono con costo_km</p>`
    : `<p class="verifica ko">✗ solo ${uguali} percorsi diretti su ${ist.n} coincidono con costo_km</p>`;
  document.getElementById("fatti").innerHTML = f;

  document.getElementById("legenda-rotte").hidden = !sol.calcolata;
  document.getElementById("fila-diretti").hidden = !sol.calcolata;
  document.getElementById("diretti").checked = stato.diretti;
  document.getElementById("aiuto").textContent = sol.calcolata
    ? "Clicca un nodo o una riga per seguire una richiesta, un tratto o il nome di un veicolo per vederne la rotta. Esc o un clic sulla mappa per tornare alla vista completa."
    : "Clicca un nodo, un percorso o una riga per evidenziare la richiesta. Esc o un clic sulla mappa per tornare alla vista completa.";

  const perRichiesta = Object.fromEntries(ist.richieste.map(r => [r.i, r]));
  const riga = i => {
    const r = perRichiesta[i], c = sol.colore[i], viaggio = sol.calcolata ? sol.viaggi[i] : null;
    const misure = viaggio
      ? `<b>${num(viaggio.km, 2)} km</b>${num(viaggio.minuti, 1)} min a bordo`
      : `<b>${num(r.km, 2)} km</b>${num(r.minuti, 1)} min`;
    return `<button class="riga" type="button" data-i="${i}" style="--c:${c};--t:${testoSu(c)}">`
      + `<span class="segno">${i}</span>`
      + `<span><span class="nome">Richiesta ${i}</span><span class="vie">${esc(breve(r.pickup.vie))} → ${esc(breve(r.dropoff.vie))}</span></span>`
      + `<span class="misure">${misure}</span></button>`;
  };
  const blocco = g => `<div class="gruppo" style="--c:${sol.colore[g[0]]}">`
    + (g.length > 1 ? `<p class="etichetta">in pooling</p>` : "")
    + g.map(riga).join("") + "</div>";

  let e = "";
  for (const v of sol.veicoli) {
    if (v.k === null) {
      e += `<div class="veicolo">${v.gruppi.map(blocco).join("")}</div>`;
      continue;
    }
    const nr = v.gruppi.reduce((s, g) => s + g.length, 0);
    const eventi = v.eventi.map(x => (x > 0 ? "+" : "−") + Math.abs(x)).join(" ");
    e += `<section class="veicolo"><h2><button class="nome-veicolo" type="button" data-k="${v.k}" aria-pressed="false">Veicolo ${v.k}</button>`
      + `<small>${num(v.km, 2)} km, ${nr} ${nr === 1 ? "richiesta" : "richieste"}</small></h2>`
      + `<p class="eventi">${eventi}</p>`
      + (v.coincide ? "" : `<p class="verifica ko">✗ rotta ${num(v.km, 4)} km, modello ${num(v.costo_modello, 4)} km</p>`)
      + v.gruppi.map(blocco).join("") + "</section>";
  }
  if (sol.rifiutate.length) {
    e += `<section class="veicolo"><h2>Rifiutate</h2>${sol.rifiutate.map(i => blocco([i])).join("")}</section>`;
  }
  document.getElementById("elenco").innerHTML = e;
}

// ---------- eventi ----------
document.getElementById("scelta-istanza").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  stato.istanza = Number(b.dataset.k);
  mappa.closePopup();
  disegna(true);
});
document.getElementById("elenco").addEventListener("click", e => {
  const v = e.target.closest(".nome-veicolo");
  if (v) { mappa.closePopup(); seleziona({ tipo: "veicolo", k: Number(v.dataset.k) }, { inquadra: true }); return; }
  const b = e.target.closest(".riga");
  if (b) seleziona({ tipo: "richiesta", i: Number(b.dataset.i) }, { inquadra: true, popup: "pickup" });
});
document.getElementById("diretti").addEventListener("change", e => {
  stato.diretti = e.target.checked;
  aggiornaDiretti();
  applicaStile();
});
document.getElementById("mostra-tutto").addEventListener("click", deseleziona);
document.addEventListener("keydown", e => { if (e.key === "Escape") deseleziona(); });
mappa.on("click", deseleziona);

stato.obiettivo = DATI.istanze[0].soluzioni[0].chiave;
disegna(true);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()