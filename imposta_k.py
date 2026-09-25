"""
Assegna a ogni istanza OSM il numero di veicoli |K| della Tabella 7 del paper
(Sez. 4.2).

La Tabella 7 non dà un K per istanza, ma un intervallo per ogni gruppo di cinque
istanze con gli stessi (Q, n), per esempio Q3.20.1-5 -> |K| = 4-5, e non dice quale
istanza riceve quale valore. La nostra regola, che non usa numeri esterni alla tabella:
si prova il valore più piccolo dell'intervallo e, se con quei veicoli nessuna soluzione
serve tutte le richieste, il successivo.
K è un tetto (vincolo (1d)): se K è fattibile lo è anche K+1, quindi il primo valore
fattibile è il K minimo dell'istanza. Un intervallo di due valori è proprio ciò che si
ottiene quando i K minimi del gruppo differiscono di uno.

La fattibilità si verifica con Model II, obiettivo nullo e SolutionLimit = 1: il solver
si ferma alla prima soluzione intera invece di cercare l'ottimo.

Ogni file riceve il campo K_stato:
    "tabella7"     K scritto nel file, istanza utilizzabile
    "infattibile"  nessun K dell'intervallo è fattibile (dimostrato da Gurobi)
    "indeciso"     time limit scaduto senza risposta; si può riprovare con --solo
                   e un time limit più lungo
Le istanze "infattibile" e "indeciso" sono escluse dagli esperimenti, come nelle note
a/b della Tabella 8 del paper, e conservano K = n: per questo gli script degli
esperimenti le filtrano con istanza_utilizzabile().

Uso (dalla cartella del progetto):
    python imposta_k.py istanze_trieste              verifica e scrive K nei JSON
    python imposta_k.py istanze_trieste --prova      verifica soltanto, non scrive nulla
    python imposta_k.py istanze_trieste --time-limit 600 --threads 4
    python imposta_k.py istanze_trieste --solo Trieste_Q3.80.2 Trieste_Q3.80.4 --time-limit 1800
"""

import argparse
import json
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from instances import leggi_istanza, FORMATI_JSON
from graph import Grafo
from model import costruisci_modello


# ----------------------------------------------------------------------
# La Tabella 7 del paper
# ----------------------------------------------------------------------

# TABELLA_7[Q][n] = (K piu' piccolo, K piu' grande) dell'intervallo pubblicato.
TABELLA_7: dict[int, dict[int, tuple[int, int]]] = {
    3: {20: (4, 5), 30: (5, 6), 40: (6, 7), 60: (9, 10), 80: (10, 11), 100: (13, 14)},
    6: {20: (5, 6), 30: (6, 7), 40: (8, 9), 60: (11, 12), 80: (13, 14), 100: (16, 17)},
}


def intervallo_tabella_7(Q: int, n: int) -> tuple[int, int]:
    """Intervallo |K| della Tabella 7 per un'istanza con capacita' Q e n richieste."""
    try:
        return TABELLA_7[Q][n]
    except KeyError:
        raise ValueError(f"la Tabella 7 non prevede istanze con Q={Q}, n={n}") from None


# ----------------------------------------------------------------------
# Verifica di fattibilita' per un K dato
# ----------------------------------------------------------------------

def verifica_fattibilita(grafo: Grafo, K: int, *, time_limit: float,
                         threads: int | None) -> str:
    """
    Dice se Model II ammette almeno una soluzione con al più K veicoli.
    Il grafo non dipende da K (compare solo nel vincolo (1d)), quindi si costruisce una
    volta sola e qui si cambia soltanto istanza.K, rimesso al valore originale alla fine.
    """
    istanza = grafo.istanza
    K_originale = istanza.K
    istanza.K = K
    try:
        m = costruisci_modello(grafo, variante="II", log=False,
                               time_limit=time_limit, threads=threads)
        m.setObjective(gp.LinExpr(), GRB.MINIMIZE)   # interessa solo l'esistenza
        m.Params.SolutionLimit = 1                    # fermati alla prima soluzione
        m.optimize()
        if m.SolCount > 0:
            return "fattibile"
        if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            # con obiettivo nullo il problema non puo' essere illimitato:
            # INF_OR_UNBD qui significa infattibile
            return "infattibile"
        return "indeciso"
    finally:
        istanza.K = K_originale


def scegli_K(grafo: Grafo, intervallo: tuple[int, int], verifica) -> tuple[int | None, dict[int, str]]:
    """
    Regola "più piccolo K fattibile nell'intervallo".
    verifica(grafo, K) restituisce "fattibile", "infattibile" o "indeciso".
    Restituisce (K scelto oppure None, {K provato: esito}).
    """
    lo, hi = intervallo
    esiti: dict[int, str] = {}
    for K in range(lo, hi + 1):
        esiti[K] = verifica(grafo, K)
        if esiti[K] == "fattibile":
            return K, esiti
        if esiti[K] == "indeciso":
            # se K fosse fattibile, un K più grande non sarebbe il minimo: ci si ferma qui
            return None, esiti
    return None, esiti


# ----------------------------------------------------------------------
# Lettura e scrittura dei JSON
# ----------------------------------------------------------------------

def stato_da_esiti(K: int | None, esiti: dict[int, str]) -> str:
    """Il valore di K_stato: 'tabella7', 'indeciso' oppure 'infattibile'."""
    if K is not None:
        return "tabella7"
    return "indeciso" if "indeciso" in esiti.values() else "infattibile"


def aggiorna_json(path: Path, K: int | None, intervallo: tuple[int, int],
                  esiti: dict[int, str], time_limit: float | None) -> None:
    """
    Scrive nel file K_stato, K_tabella7 (l'intervallo), K_esiti (l'esito di ogni K
    provato) e K_time_limit. K e scelte["K"] cambiano solo se l'istanza è utilizzabile;
    tutti gli altri campi restano identici.
    """
    dati = json.loads(path.read_text(encoding="utf-8"))
    lo, hi = intervallo
    dati["K_stato"] = stato_da_esiti(K, esiti)
    dati["K_tabella7"] = [lo, hi]
    dati["K_esiti"] = {str(k): e for k, e in esiti.items()}   # JSON vuole chiavi stringa
    dati["K_time_limit"] = time_limit
    if K is not None:
        dati["K"] = K
        dati.setdefault("scelte", {})["K"] = f"Tabella 7: minimo fattibile in [{lo}, {hi}]"
    path.write_text(json.dumps(dati, ensure_ascii=False), encoding="utf-8")


def istanza_utilizzabile(path: str | Path) -> bool:
    """
    True se l'istanza ha un K valido della Tabella 7: va controllato in ogni script
    che lancia esperimenti sulle istanze OSM.
    """
    dati = json.loads(Path(path).read_text(encoding="utf-8"))
    return dati.get("K_stato") == "tabella7"


def istanze_osm(cartella: Path) -> list[Path]:
    """I file .json della cartella che hanno il formato generato da osm_city.py."""
    trovati = []
    for path in sorted(cartella.glob("*.json")):
        formato = json.loads(path.read_text(encoding="utf-8")).get("formato")
        if formato in FORMATI_JSON:
            trovati.append(path)
    return trovati


# ----------------------------------------------------------------------
# Programma principale
# ----------------------------------------------------------------------

def elabora(cartella: Path, verifica, *, scrivi: bool,
            solo: list[str] | None = None,
            time_limit: float | None = None) -> list[str]:
    """
    Sceglie K per tutte le istanze della cartella e restituisce i nomi di quelle escluse
    (nessun K fattibile, indecise o con dati incoerenti). time_limit serve solo a
    registrarlo nei file: la verifica lo riceve già fissato.
    """
    # con una cartella sbagliata il ciclo non farebbe nulla e il messaggio finale
    # direbbe "tutte fattibili"
    if not cartella.is_dir():
        raise SystemExit(f"cartella inesistente: {cartella.resolve()}\n"
                         f"(su Windows un percorso che inizia con '\\' parte dalla radice del disco)")
    istanze = istanze_osm(cartella)
    if solo:
        # --solo: rielabora soltanto le istanze indicate (per esempio le "indeciso",
        # con un time limit piu' lungo), senza rifare le altre
        mancanti = set(solo) - {p.stem for p in istanze}
        if mancanti:
            raise SystemExit(f"istanze non trovate: {sorted(mancanti)}")
        istanze = [p for p in istanze if p.stem in solo]
    if not istanze:
        raise SystemExit(f"nessuna istanza in formato {FORMATI_JSON} in {cartella.resolve()}")
    print(f"{len(istanze)} istanze trovate in {cartella.resolve()}\n")

    problemi = []
    print(f"{'istanza':<22}{'Q':>3}{'n':>5}  {'tab.7':<8}{'esiti':<34}K")
    for path in istanze:
        try:
            istanza = leggi_istanza(path)
            intervallo = intervallo_tabella_7(istanza.Q, istanza.n)
            grafo = Grafo.costruisci(istanza)
            K, esiti = scegli_K(grafo, intervallo, verifica)
        except ValueError as errore:
            # un'istanza con dati incoerenti non deve fermare le altre 59:
            # la si segnala e si prosegue
            print(f"{path.stem:<22}ERRORE: {errore}")
            problemi.append(path.stem)
            continue

        testo_esiti = ", ".join(f"{k}:{e}" for k, e in esiti.items())
        testo_K = str(K) if K is not None else "--"
        print(f"{path.stem:<22}{istanza.Q:>3}{istanza.n:>5}  "
              f"{intervallo[0]}-{intervallo[1]:<6}{testo_esiti:<34}{testo_K}")

        if K is None:
            problemi.append(path.stem)
        if scrivi:
            aggiorna_json(path, K, intervallo, esiti, time_limit)
    return problemi


def main() -> None:
    parser = argparse.ArgumentParser(description="Assegna |K| dalla Tabella 7 alle istanze OSM.")
    parser.add_argument("cartella", type=Path, help="cartella con i JSON di osm_city.py")
    parser.add_argument("--prova", action="store_true", help="verifica senza scrivere i file")
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="secondi per ogni verifica di fattibilita' (default 300)")
    parser.add_argument("--threads", type=int, default=None, help="thread di Gurobi")
    parser.add_argument("--solo", nargs="+", default=None, metavar="NOME",
                        help="elabora solo queste istanze (nome del file senza .json)")
    args = parser.parse_args()

    # time limit e threads fissati qui: scegli_K vede solo (grafo, K)
    verifica = lambda grafo, K: verifica_fattibilita(
        grafo, K, time_limit=args.time_limit, threads=args.threads)

    problemi = elabora(args.cartella, verifica, scrivi=not args.prova, solo=args.solo,
                       time_limit=args.time_limit)

    print()
    if problemi:
        print(f"{len(problemi)} istanze escluse (K_stato diverso da 'tabella7', "
              f"K lasciato invariato):")
        for nome in problemi:
            print(f"  {nome}")
    else:
        print("Tutte le istanze hanno un K fattibile nell'intervallo della Tabella 7.")
    if args.prova:
        print("(modalita' --prova: nessun file scritto)")


if __name__ == "__main__":
    main()