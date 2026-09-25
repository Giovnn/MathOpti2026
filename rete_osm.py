"""
Dal file .osm grezzo al grafo stradale percorribile in auto, usato da osm_city.py
per le distanze fra i punti delle istanze Trieste.

Fasi (ognuna stampa quanti nodi e archi restano):
    1. carica tutto il file, senza semplificare
    2. toglie gli archi non percorribili in auto
    3. tiene la componente fortemente connessa più grande
    4. semplifica (toglie i nodi intermedi che non sono incroci)

Produce, accanto al file .osm:
    <nome>_drive.graphml   il grafo finale, da ricaricare con ox.io.load_graphml
    <nome>_drive.png       la rete finale
    <nome>_scartati.png    in rosso i nodi carrabili tolti dalla componente forte

Uso:
    python rete_osm.py trieste_osm_highway.osm
"""

import sys
from pathlib import Path

import networkx as nx
import osmnx as ox

# Valori di "highway" percorribili in auto (lista bianca: tutto il resto è scartato).
HIGHWAY_CARRABILI = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential", "living_street",
}
ACCESSO_VIETATO = {"private", "no"}

# Tag che OSMnx di default non copia sugli archi: li aggiungiamo per poterli filtrare.
TAG_EXTRA = ["motor_vehicle", "motorcar"]


# ----------------------------------------------------------------------
# regola di filtro
# ----------------------------------------------------------------------
def arco_carrabile(dati: dict) -> bool:
    """True se un arco, con i suoi attributi OSM, e' percorribile da un'auto."""
    tipo = dati.get("highway")
    if isinstance(tipo, list):
        # succede solo su grafi gia' semplificati, dove un arco fonde piu' way
        raise ValueError("attributo 'highway' a lista: filtrare PRIMA di semplificare")
    if tipo not in HIGHWAY_CARRABILI:
        return False
    if dati.get("area") == "yes":
        return False
    if dati.get("access") in ACCESSO_VIETATO:
        return False
    if dati.get("motor_vehicle") == "no" or dati.get("motorcar") == "no":
        return False
    return True


# ----------------------------------------------------------------------
# fasi della costruzione
# ----------------------------------------------------------------------
def carica_da_xml(percorso: Path) -> nx.MultiDiGraph:
    """Fase 1: tutte le way del file diventano archi (nessun filtro, nessuna semplificazione)."""
    for tag in TAG_EXTRA:
        if tag not in ox.settings.useful_tags_way:
            ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + [tag]
    return ox.graph.graph_from_xml(percorso, simplify=False, retain_all=True)


def filtra_carrabili(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Fase 2: copia di G senza archi non carrabili e senza nodi rimasti isolati."""
    da_togliere = [(u, v, k) for u, v, k, dati in G.edges(keys=True, data=True)
                   if not arco_carrabile(dati)]
    H = G.copy()
    H.remove_edges_from(da_togliere)
    H.remove_nodes_from(list(nx.isolates(H)))
    return H


def componente_forte(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Fase 3: da ogni nodo si puo' raggiungere ogni altro nodo (rispettando i sensi unici)."""
    return ox.truncate.largest_component(G, strongly=True)


def riepilogo(fase: str, G: nx.MultiDiGraph) -> None:
    km = sum(d.get("length", 0.0) for _u, _v, d in G.edges(data=True)) / 1000
    print(f"{fase:<28} nodi = {G.number_of_nodes():>6}   archi = {G.number_of_edges():>6}"
          f"   km di archi = {km:6.1f}")


def costruisci_rete_da_xml(percorso: Path) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
    """Restituisce (grafo finale, grafo filtrato prima della componente forte)."""
    G = carica_da_xml(percorso)
    riepilogo("1. caricato (tutto)", G)

    G_filtrato = filtra_carrabili(G)
    riepilogo("2. solo carrabili", G_filtrato)

    # per confronto: la componente debole, quella che OSMnx tiene di default
    debole = max(nx.weakly_connected_components(G_filtrato), key=len)
    print(f"{'   componente debole max':<28} nodi = {len(debole):>6}")

    G_forte = componente_forte(G_filtrato)
    riepilogo("3. componente forte", G_forte)

    G_finale = ox.simplification.simplify_graph(G_forte)
    riepilogo("4. semplificato", G_finale)

    senso_unico = sum(1 for _u, _v, d in G_forte.edges(data=True) if d.get("oneway") is True)
    print(f"\nArchi a senso unico (prima della semplificazione): "
          f"{senso_unico} su {G_forte.number_of_edges()}")
    return G_finale, G_filtrato


def disegna_scartati(G_filtrato: nx.MultiDiGraph, G_forte_nodi: set, filepath: Path) -> None:
    """Rete filtrata in grigio, in rosso i nodi che la componente forte ha eliminato."""
    colori = ["red" if n not in G_forte_nodi else "none" for n in G_filtrato.nodes]
    taglie = [12 if n not in G_forte_nodi else 0 for n in G_filtrato.nodes]
    ox.plot.plot_graph(G_filtrato, node_color=colori, node_size=taglie,
                       edge_color="#999999", edge_linewidth=0.6, bgcolor="white",
                       show=False, save=True, close=True, filepath=filepath)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python rete_osm.py <file.osm>")
        sys.exit(1)

    percorso = Path(sys.argv[1])
    G, G_filtrato = costruisci_rete_da_xml(percorso)

    file_graphml = percorso.with_name(percorso.stem + "_drive.graphml")
    file_rete = percorso.with_name(percorso.stem + "_drive.png")
    file_scartati = percorso.with_name(percorso.stem + "_scartati.png")

    ox.io.save_graphml(G, file_graphml)
    ox.plot.plot_graph(G, node_size=0, edge_color="black", edge_linewidth=0.8,
                       bgcolor="white", show=False, save=True, close=True, filepath=file_rete)
    # nodi carrabili scartati: quelli di G_filtrato fuori dalla componente forte
    disegna_scartati(G_filtrato, set(componente_forte(G_filtrato).nodes), file_scartati)
    print(f"\nSalvati: {file_graphml.name}, {file_rete.name}, {file_scartati.name}")
