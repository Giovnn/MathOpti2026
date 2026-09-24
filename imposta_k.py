"""
imposta_K.py — Assegna a ogni istanza OSM il numero di veicoli |K| della Tabella 7 del paper.

Riferimento: Gaul, Klamroth & Stiglmayr (2022), EJOR 301(3), Sez. 4.2, Tabella 7.

Il problema
-----------
La Tabella 7 non da' un K per ogni istanza, ma un INTERVALLO per ogni gruppo di cinque
istanze con gli stessi (Q, n): per esempio Q3.20.1-5 -> |K| = 4-5. Il paper non dice quale
istanza riceve 4 e quale 5.

La regola usata qui (l'unica che non introduce numeri esterni alla tabella):
    per ogni istanza si prova il valore piu' piccolo dell'intervallo; se con quel numero
    di veicoli non esiste nessuna soluzione che serva tutte le richieste, si prova il
    successivo. Il primo valore fattibile e' il K dell'istanza.

Perche' "il piu' piccolo fattibile": K e' un tetto (vincolo (1d): al piu' K veicoli
escono dal deposito). Se K e' fattibile lo e' anche K+1, quindi l'insieme dei K fattibili
e' {K_min, K_min+1, ...}. Un intervallo di due valori su cinque istanze e' esattamente
cio' che si ottiene dando a ciascuna istanza il suo K_min, quando i K_min del gruppo
differiscono di uno. Resta un'interpretazione, da dichiarare nella relazione.

Se nessun valore dell'intervallo e' fattibile, lo script NON inventa un K: l'istanza viene
ESCLUSA dagli esperimenti. E' lo stesso criterio delle note a/b della Tabella 8 del paper
(medie calcolate solo sulle istanze utilizzabili), applicato un passo prima: qui l'istanza
non e' utilizzabile perche' la flotta della Tabella 7 non basta a servire tutte le richieste
(oppure perche' Gurobi non l'ha deciso entro il time limit).

Il campo K_stato
----------------
Ogni file elaborato riceve K_stato, che e' cio' che gli script degli esperimenti devono
leggere (con istanza_utilizzabile, qui sotto):
    "tabella7"     K scritto nel file, istanza utilizzabile
    "infattibile"  nessun K dell'intervallo e' fattibile (dimostrato): esclusa
    "indeciso"     time limit scaduto senza risposta: esclusa, salvo nuovo tentativo
                   con --solo e un time limit piu' lungo
ATTENZIONE: nei file esclusi K resta quello originale (K = n). Usarli senza controllare
K_stato darebbe risultati sbagliati: e' il motivo per cui esiste istanza_utilizzabile.

Come si verifica la fattibilita'
--------------------------------
Si costruisce Model II con il K da provare e si chiede a Gurobi una soluzione qualsiasi:
obiettivo nullo e SolutionLimit = 1, cosi' il solver si ferma alla prima soluzione intera
trovata invece di cercare l'ottimo. Tre esiti possibili:
    "fattibile"    -> trovata una soluzione
    "infattibile"  -> Gurobi ha DIMOSTRATO che non ne esistono
    "indeciso"     -> time limit scaduto senza soluzione e senza prova di infattibilita'
Con "indeciso" non si puo' affermare che quel K sia il minimo, quindi l'istanza viene
segnalata come le infattibili.

Uso (dalla cartella del progetto):
    python imposta_K.py dati_trieste              verifica e scrive K nei JSON
    python imposta_K.py dati_trieste --prova      verifica soltanto, non scrive nulla
    python imposta_K.py dati_trieste --time-limit 600 --threads 4
    python imposta_K.py dati_trieste --solo Trieste_Q3.80.2 Trieste_Q3.80.4 --time-limit 1800
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
# Blocco 1 — La Tabella 7 del paper
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
# Blocco 2 — Verifica di fattibilita' per un K dato
# ----------------------------------------------------------------------

def verifica_fattibilita(grafo: Grafo, K: int, *, time_limit: float,
                         threads: int | None) -> str:
    """
    Dice se Model II ammette almeno una soluzione con al piu' K veicoli.

    Il grafo event-based NON dipende da K (K compare solo nel vincolo (1d)), quindi lo
    si costruisce una volta sola e qui si cambia soltanto istanza.K. Il valore originale
    viene rimesso a posto alla fine, anche in caso di errore (blocco finally).
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
    Applica la regola "piu' piccolo K fattibile nell'intervallo".

    verifica: funzione (grafo, K) -> "fattibile" / "infattibile" / "indeciso".
    E' un parametro, e non una chiamata diretta a verifica_fattibilita, per due motivi:
    si puo' provare la logica senza Gurobi, e time limit/threads restano fuori da qui.

    Restituisce (K scelto oppure None, {K provato: esito}).
    """
    lo, hi = intervallo
    esiti: dict[int, str] = {}
    for K in range(lo, hi + 1):
        esiti[K] = verifica(grafo, K)
        if esiti[K] == "fattibile":
            return K, esiti
        if esiti[K] == "indeciso":
            # non sappiamo se K e' fattibile: se lo fosse, un K piu' grande
            # non sarebbe il minimo. Meglio fermarsi e segnalare.
            return None, esiti
    return None, esiti


# ----------------------------------------------------------------------
# Blocco 3 — Lettura e scrittura dei JSON
# ----------------------------------------------------------------------

def stato_da_esiti(K: int | None, esiti: dict[int, str]) -> str:
    """Il valore di K_stato: 'tabella7', 'indeciso' oppure 'infattibile'."""
    if K is not None:
        return "tabella7"
    return "indeciso" if "indeciso" in esiti.values() else "infattibile"


def aggiorna_json(path: Path, K: int | None, intervallo: tuple[int, int],
                  esiti: dict[int, str], time_limit: float | None) -> None:
    """
    Registra nel file l'esito della scelta di K.

    Campi toccati:
        K_stato           "tabella7" / "infattibile" / "indeciso" (vedi docstring del modulo)
        K_tabella7        l'intervallo pubblicato, [lo, hi]
        K_esiti           l'esito di ogni K provato, per la relazione
        K_time_limit      secondi concessi a ogni verifica (conta per gli "indeciso")
    e, SOLO se l'istanza e' utilizzabile:
        K                 il valore usato da instances.py e dal vincolo (1d)
        scelte["K"]       la regola, in parole (prima valeva "n")
    Tutti gli altri campi restano identici, quindi nodi, finestre e costi non cambiano.
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
    True se l'istanza ha un K valido della Tabella 7. Da chiamare in OGNI script che
    lancia esperimenti sulle istanze OSM, prima di leggerle.
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
# Blocco 4 — Programma principale
# ----------------------------------------------------------------------

def elabora(cartella: Path, verifica, *, scrivi: bool,
            solo: list[str] | None = None,
            time_limit: float | None = None) -> list[str]:
    """
    Sceglie K per tutte le istanze della cartella. Restituisce i nomi delle istanze
    escluse: nessun K della Tabella 7 fattibile, indecise, o con dati incoerenti.
    time_limit serve solo a registrarlo nei file (la verifica lo riceve gia' fissato).
    """
    # Senza questi controlli una cartella sbagliata produce zero righe e il messaggio
    # finale "tutte fattibili": un successo falso. Meglio fermarsi subito.
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
        istanza = leggi_istanza(path)
        intervallo = intervallo_tabella_7(istanza.Q, istanza.n)
        grafo = Grafo.costruisci(istanza)
        try:
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

    # lambda: fissa time limit e threads, e lascia a scegli_K solo (grafo, K)
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