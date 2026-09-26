"""
converti_pbf.py — Da un estratto .osm.pbf a un file .osm (XML) con le sole strade.

Perche' serve: OSMnx costruisce grafi da file OSM XML (graph_from_xml), non dal
formato binario PBF. Qui teniamo TUTTE le way con tag "highway", anche quelle
pedonali: la decisione su cosa e' carrabile resta in un solo posto,
cioe' arco_carrabile() in rete_osm.py.

Dopo la conversione il file viene controllato: ogni nodo citato da una way
deve essere presente nel file, altrimenti OSMnx non saprebbe dove si trova.

Requisito:  pip install osmium      (il pacchetto si chiama "osmium", non "pyosmium")

Uso da terminale:
    python converti_pbf.py OpenMap\\trieste.osm.pbf
Produce:
    OpenMap\\trieste_highway.osm
"""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def nome_uscita(pbf: Path) -> Path:
    """trieste.osm.pbf -> trieste_highway.osm  (stessa cartella)."""
    nome = pbf.name
    for suffisso in (".osm.pbf", ".pbf"):
        if nome.endswith(suffisso):
            nome = nome[: -len(suffisso)]
            break
    return pbf.with_name(nome + "_highway.osm")


def estrai_strade(pbf: Path, uscita: Path) -> int:
    """Scrive in `uscita` le way con tag highway e i nodi a cui fanno riferimento."""
    import osmium   # importato qui: verifica_completezza() funziona anche senza pyosmium

    # legge SOLO le way (osmium.osm.WAY) e lascia passare solo quelle con chiave "highway"
    way_stradali = osmium.FileProcessor(str(pbf), osmium.osm.WAY) \
                         .with_filter(osmium.filter.KeyFilter("highway"))

    quante = 0
    # BackReferenceWriter: alla chiusura rilegge ref_src e aggiunge i nodi citati dalle way.
    # Il formato di uscita (XML) e' scelto dall'estensione .osm del file.
    with osmium.BackReferenceWriter(str(uscita), ref_src=str(pbf), overwrite=True) as writer:
        for way in way_stradali:
            writer.add(way)
            quante += 1
    return quante


def verifica_completezza(percorso_osm: Path) -> set[str]:
    """Controlla che ogni <nd ref> delle way abbia il suo <node>. Restituisce gli id mancanti."""
    nodi_presenti: set[str] = set()
    riferimenti: set[str] = set()
    n_way = 0
    lat_min = lon_min = math.inf
    lat_max = lon_max = -math.inf

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
        lat_media = (lat_min + lat_max) / 2
        alto_km = (lat_max - lat_min) * 111.2
        largo_km = (lon_max - lon_min) * 111.2 * math.cos(math.radians(lat_media))
        print(f"  estensione dei nodi: circa {largo_km:.1f} km (E-O) x {alto_km:.1f} km (N-S)")
    print(f"  nodi citati ma assenti: {len(mancanti)}")
    return mancanti


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python converti_pbf.py <file.osm.pbf>")
        sys.exit(1)

    pbf = Path(sys.argv[1])
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
    print("\nFile completo: si puo' passare a  python rete_osm.py", uscita)