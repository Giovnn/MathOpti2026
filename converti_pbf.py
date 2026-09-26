"""
Da un estratto OpenStreetMap in formato .osm.pbf a un file .osm (XML) con le sole strade.

È il passo che precede rete_osm.py: OSMnx costruisce il grafo da file OSM XML
(graph_from_xml), non dal formato binario PBF. Si tengono tutte le way con tag
"highway", anche quelle pedonali: cosa è percorribile in auto lo decide soltanto
arco_carrabile() in rete_osm.py.

Dopo la conversione il file viene controllato: ogni nodo citato da una way deve
essere presente, altrimenti OSMnx non saprebbe dove si trova.

Per Trieste si è usato l'estratto comunale di "estratti OpenStreetMap Italia"
(Wikimedia Italia). Dati © contributori OpenStreetMap, licenza ODbL.

Requisito:  pip install osmium   (il pacchetto si chiama "osmium", non "pyosmium")

Uso (dalla cartella del progetto):
    python converti_pbf.py trieste_osm.pbf
Produce, nella stessa cartella, trieste_osm_highway.osm (da passare a rete_osm.py).
"""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import osmium


def nome_uscita(pbf: Path) -> Path:
    """trieste_osm.pbf (o trieste_osm.osm.pbf) -> trieste_osm_highway.osm, nella stessa cartella."""
    nome = pbf.name
    for suffisso in (".osm.pbf", ".pbf"):
        if nome.endswith(suffisso):
            nome = nome[: -len(suffisso)]
            break
    return pbf.with_name(nome + "_highway.osm")


def estrai_strade(pbf: Path, uscita: Path) -> int:
    """
    Scrive in `uscita` le way con tag highway e i nodi a cui fanno riferimento.
    Restituisce il numero di way scritte.
    """
    way_stradali = osmium.FileProcessor(str(pbf), osmium.osm.WAY).with_filter(
        osmium.filter.KeyFilter("highway"))

    quante = 0
    # alla chiusura il writer rilegge il .pbf e aggiunge i nodi citati dalle way;
    # il formato XML lo decide l'estensione .osm del file di uscita
    with osmium.BackReferenceWriter(str(uscita), ref_src=str(pbf), overwrite=True) as writer:
        for way in way_stradali:
            writer.add(way)
            quante += 1
    return quante


def verifica_completezza(percorso_osm: Path) -> set[str]:
    """
    Controlla che ogni nodo citato dalle way (<nd ref>) sia presente nel file e stampa
    un riepilogo. Restituisce gli id dei nodi mancanti.
    """
    nodi_presenti: set[str] = set()
    riferimenti: set[str] = set()
    n_way = 0
    lat_min = lon_min = math.inf
    lat_max = lon_max = -math.inf

    # lettura a flusso: il file di una città intera non viene caricato tutto in memoria
    for _evento, elem in ET.iterparse(percorso_osm, events=("end",)):
        if elem.tag == "node":
            nodi_presenti.add(elem.get("id"))
            lat, lon = float(elem.get("lat")), float(elem.get("lon"))
            lat_min, lat_max = min(lat_min, lat), max(lat_max, lat)
            lon_min, lon_max = min(lon_min, lon), max(lon_max, lon)
            elem.clear()
        elif elem.tag == "way":
            n_way += 1
            riferimenti.update(nd.get("ref") for nd in elem.findall("nd"))
            elem.clear()

    mancanti = riferimenti - nodi_presenti

    print(f"File controllato: {percorso_osm.name}")
    print(f"  way: {n_way}   nodi: {len(nodi_presenti)}   nodi citati dalle way: {len(riferimenti)}")
    if nodi_presenti:
        # un grado di latitudine vale circa 111.2 km; uno di longitudine va scalato per cos(lat)
        lat_media = (lat_min + lat_max) / 2
        alto_km = (lat_max - lat_min) * 111.2
        largo_km = (lon_max - lon_min) * 111.2 * math.cos(math.radians(lat_media))
        print(f"  estensione dei nodi: circa {largo_km:.1f} km (E-O) x {alto_km:.1f} km (N-S)")
    print(f"  nodi citati ma assenti: {len(mancanti)}")
    return mancanti


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python converti_pbf.py <file.osm.pbf>")

    pbf = Path(sys.argv[1])
    if not pbf.is_file():
        sys.exit(f"file non trovato: {pbf}")
    uscita = nome_uscita(pbf)

    print(f"Estraggo le strade da {pbf.name} ...")
    n = estrai_strade(pbf, uscita)
    print(f"Scritte {n} way con tag highway in {uscita.name}\n")

    mancanti = verifica_completezza(uscita)
    if mancanti:
        sys.exit(
            f"\nATTENZIONE: {len(mancanti)} nodi citati dalle way non sono nel file.\n"
            "L'estratto ha probabilmente tagliato le way al confine: OSMnx non potrebbe\n"
            "calcolare la lunghezza degli archi verso quei nodi. Non procedere con\n"
            "rete_osm.py: questo caso va gestito prima."
        )
    print(f"\nFile completo: si può passare a  python rete_osm.py {uscita}")