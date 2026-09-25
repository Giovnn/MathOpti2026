"""
Risolve le istanze Trieste con Model II e sei degli obiettivi del paper e salva i
risultati in un CSV, una riga per istanza e obiettivo, da cui tabelle.py ricava le
Tabelle 8-12 (Sez. 4.2).

Si usano solo le istanze con K_stato = "tabella7" (vedi imposta_k.py), dalla più
piccola alla più grande, così i primi risultati arrivano in fretta.
Ogni riga viene scritta appena finito il run: se il PC si ferma il lavoro fatto resta,
e rilanciando lo stesso comando i run già presenti nel CSV vengono saltati. Per rifare
un run basta cancellarne la riga dal CSV.

Uso (dalla cartella del progetto):
    python esperimenti.py istanze_trieste
    python esperimenti.py istanze_trieste --time-limit 600 --threads 4
    python esperimenti.py istanze_trieste --solo Trieste_Q3.20.1 --obiettivi fc fr
"""

import argparse
import csv
import time
from pathlib import Path

from instances import leggi_istanza
from graph import Grafo
from objectives import costruisci_con_obiettivo
from results import risolvi
from imposta_k import istanze_osm, istanza_utilizzabile


# I sei obiettivi delle Tabelle 8-10, nell'ordine in cui compaiono nel paper.
# (fn non c'e': nel paper non ha tabelle sulle istanze urbane.)
OBIETTIVI_PAPER = ("fc", "fr", "fcr", "frcr", "frmax", "fcrmax")

# Colonne del CSV, in quest'ordine; un run è identificato da istanza e obiettivo.
CAMPI = [
    "istanza", "Q", "n", "K", "obiettivo", "variante",
    "stato", "obj", "bound", "gap", "tempo", "nodi_bb", "veicoli",
    "costo", "regret", "regret_grezzo", "regret_max", "regret_max_grezzo",
    "rifiuti", "ar", "regret_medio_servito",
    "time_limit", "mip_gap", "nota",
]


def run_gia_fatti(csv_path: Path) -> set[tuple[str, str]]:
    """Coppie (istanza, obiettivo) gia' presenti nel CSV: quei run si saltano."""
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {(r["istanza"], r["obiettivo"]) for r in csv.DictReader(f)}


def scrivi_riga(csv_path: Path, riga: dict) -> None:
    """Aggiunge una riga in fondo al CSV, creandolo con l'intestazione se non c'e'."""
    nuovo = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        scrittore = csv.DictWriter(f, fieldnames=CAMPI, extrasaction="ignore")
        if nuovo:
            scrittore.writeheader()
        scrittore.writerow(riga)


def esegui_uno(grafo: Grafo, obiettivo: str, *, time_limit: float,
               mip_gap: float, threads: int | None) -> dict:
    """
    Un singolo run: costruisce Model II con l'obiettivo dato, lo risolve e restituisce
    la riga da scrivere.

    Se qualcosa va storto (dati incoerenti, memoria, licenza) la riga ha stato "errore"
    e il messaggio nella colonna nota: un run fallito non deve fermare gli altri.
    """
    istanza = grafo.istanza
    comune = {"istanza": istanza.nome, "Q": istanza.Q, "n": istanza.n, "K": istanza.K,
              "obiettivo": obiettivo, "variante": "II",
              "time_limit": time_limit, "mip_gap": mip_gap}
    try:
        m = costruisci_con_obiettivo(grafo, obiettivo, variante="II", log=False,
                                     time_limit=time_limit, mip_gap=mip_gap,
                                     threads=threads)
        r = risolvi(m)
        return {**comune, **r.riga()}
    except Exception as errore:
        return {**comune, "stato": "errore", "nota": f"{type(errore).__name__}: {errore}"}


def elabora(cartella: Path, csv_path: Path, obiettivi: tuple[str, ...], *,
            time_limit: float, mip_gap: float, threads: int | None,
            solo: list[str] | None = None) -> None:
    """Esegue tutti i run mancanti e li scrive nel CSV, uno alla volta."""
    if not cartella.is_dir():
        raise SystemExit(f"cartella inesistente: {cartella.resolve()}")

    percorsi = [p for p in istanze_osm(cartella) if istanza_utilizzabile(p)]
    if solo:
        mancanti = set(solo) - {p.stem for p in percorsi}
        if mancanti:
            raise SystemExit(f"istanze non trovate o non utilizzabili: {sorted(mancanti)}")
        percorsi = [p for p in percorsi if p.stem in solo]
    if not percorsi:
        raise SystemExit("nessuna istanza utilizzabile: lanciare prima imposta_k.py")

    # dalla piu' piccola alla piu' grande
    istanze = sorted((leggi_istanza(p) for p in percorsi), key=lambda i: (i.n, i.Q, i.nome))
    fatti = run_gia_fatti(csv_path)
    da_fare = [(i, ob) for i in istanze for ob in obiettivi if (i.nome, ob) not in fatti]
    print(f"{len(istanze)} istanze, {len(obiettivi)} obiettivi: "
          f"{len(da_fare)} run da fare, {len(istanze) * len(obiettivi) - len(da_fare)} gia' nel CSV\n")

    grafo_corrente = None
    for k, (istanza, obiettivo) in enumerate(da_fare, start=1):
        # il grafo non dipende dall'obiettivo: si ricostruisce solo al cambio di istanza
        if grafo_corrente is None or grafo_corrente.istanza.nome != istanza.nome:
            grafo_corrente = Grafo.costruisci(istanza)

        inizio = time.perf_counter()
        riga = esegui_uno(grafo_corrente, obiettivo, time_limit=time_limit,
                          mip_gap=mip_gap, threads=threads)
        if riga["stato"] == "interrotto":
            # Ctrl+C durante la risoluzione: il run non va nel CSV, così al rilancio si rifà
            print(f"\ninterrotto durante {istanza.nome} {obiettivo}: run non salvato")
            return
        scrivi_riga(csv_path, riga)

        obj = riga.get("obj")
        coda = f"obj {obj:.3f}" if obj is not None else riga.get("nota", "")
        print(f"[{k}/{len(da_fare)}] {istanza.nome:<18} {obiettivo:<7} {riga['stato']:<12} "
              f"{time.perf_counter() - inizio:7.1f} s  {coda}")

    print(f"\nfatto: {csv_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Esperimenti sulle istanze Trieste (Model II).")
    parser.add_argument("cartella", type=Path, help="cartella con i JSON")
    parser.add_argument("--csv", type=Path, default=Path("risultati/risultati_trieste.csv"),
                        help="file dei risultati (default risultati/risultati_trieste.csv)")
    parser.add_argument("--time-limit", type=float, default=600.0,
                        help="secondi per ogni run (default 600)")
    parser.add_argument("--mip-gap", type=float, default=0.0,
                        help="gap relativo richiesto a Gurobi (default 0, cioe' ottimo)")
    parser.add_argument("--threads", type=int, default=None, help="thread di Gurobi")
    parser.add_argument("--obiettivi", nargs="+", default=list(OBIETTIVI_PAPER),
                        metavar="NOME", help=f"default: {' '.join(OBIETTIVI_PAPER)}")
    parser.add_argument("--solo", nargs="+", default=None, metavar="NOME",
                        help="esegui solo queste istanze")
    args = parser.parse_args()

    elabora(args.cartella, args.csv, tuple(args.obiettivi),
            time_limit=args.time_limit, mip_gap=args.mip_gap,
            threads=args.threads, solo=args.solo)


if __name__ == "__main__":
    main()