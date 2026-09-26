"""
Tabelle 13 e 14 del paper: le rotte dei veicoli di una singola istanza, risolta con
due obiettivi diversi (nel paper f_cr e f_rcr sull'istanza Q3n20.5). Per ogni veicolo
stampa la sequenza degli eventi con l'orario di ciascuno.

I run vanno rifatti: il CSV di esperimenti.py registra quanti veicoli sono stati usati,
non dove sono passati e a che ora. Su un'istanza da 20 utenti bastano pochi secondi.

Per ogni obiettivo stampa un riepilogo, una tabella per veicolo ("Location" con gli
eventi, "Time[m]" con gli orari), gli utenti con regret positivo e quelli rifiutati.
Con due obiettivi aggiunge un confronto finale, chi viene rifiutato e a chi cambia il
regret, come nel commento del paper alle Figure 2 e 3.

Notazione del paper: 15+ l'utente 15 sale, 15- l'utente 15 scende. Gli orari sono
minuti dall'inizio del servizio, presi dallo schedule minimo della rotta
(graph.schedule_minimo) e non dai B del solver, quindi riproducibili. Il deposito non
compare, come nelle tabelle del paper.

Uso (dalla cartella del progetto):
    python rotte.py istanze_trieste/Trieste_Q3.20.5.json
    python rotte.py <istanza> --obiettivi fcr frcr --salva tabelle --png tabelle
    python rotte.py <istanza> --obiettivi fc fr --time-limit 300
"""

import argparse
import csv
import textwrap
from pathlib import Path

from instances import leggi_istanza
from graph import Grafo
from objectives import costruisci_con_obiettivo, regret_utente
from results import risolvi, Risultato, Rotta
from tabelle import PASSO_CARATTERE, ALTEZZA_RIGA, MARGINE_ALTO, MARGINE_BASSO


# Gli obiettivi delle Tabelle 13 e 14 del paper.
PREDEFINITI = ("fcr", "frcr")

# Quale tabella del paper corrisponde a quale obiettivo, per intestare l'output.
TABELLA = {"fcr": "Tabella 13", "frcr": "Tabella 14"}


# ----------------------------------------------------------------------
# Eventi di una rotta
# ----------------------------------------------------------------------

def eventi(rotta: Rotta) -> list[tuple[str, float]]:
    """
    Gli eventi di un veicolo, senza il deposito iniziale e finale: (etichetta, orario).
    L'etichetta viene dalla prima componente del nodo: positiva sale, negativa scende.
    """
    lista = []
    for k in range(1, len(rotta.nodi) - 1):
        utente = rotta.nodi[k][0]
        segno = "+" if utente > 0 else "-"
        lista.append((f"{abs(utente)}{segno}", rotta.schedule[k]))
    return lista


def ordina(rotte: list[Rotta]) -> list[Rotta]:
    """Veicoli ordinati per orario del primo servizio: output stabile e leggibile."""
    return sorted(rotte, key=lambda r: r.schedule[1] if len(r.schedule) > 1 else 0.0)


def regret_per_utente(risultato: Risultato, istanza) -> dict[int, float]:
    """Regret di ogni utente servito, calcolato sugli stessi orari che vengono stampati."""
    valori = {}
    for rotta in risultato.rotte:
        for utente, arrivo in rotta.arrivi().items():
            valori[utente] = regret_utente(istanza, utente, arrivo)
    return valori


def positivi_e_rifiutati(regret: dict[int, float], istanza) -> tuple[dict[int, float], list[int]]:
    """Utenti con regret positivo (oltre gli arrotondamenti) e utenti rifiutati."""
    positivi = {i: r for i, r in sorted(regret.items()) if r > 1e-6}
    rifiutati = sorted(set(istanza.utenti()) - set(regret))
    return positivi, rifiutati


# ----------------------------------------------------------------------
# Stampa in stile paper
# ----------------------------------------------------------------------

def stampa_tour(numero: int, rotta: Rotta) -> None:
    """Le due righe di un veicolo: eventi sopra, orari sotto, incolonnati."""
    lista = eventi(rotta)
    if not lista:
        print(f"  Tour {numero}  (veicolo non utilizzato)")
        return

    etichette = [nome for nome, _ in lista]
    orari = [f"{t:.1f}" for _, t in lista]
    larghezze = [max(len(a), len(b)) + 2 for a, b in zip(etichette, orari)]

    intestazione = f"  Tour {numero}"
    rientro = " " * len(intestazione)
    print(f"{intestazione}  Location" + "".join(v.rjust(l) for v, l in zip(etichette, larghezze)))
    print(f"{rientro}  Time[m] " + "".join(v.rjust(l) for v, l in zip(orari, larghezze)))


def stampa_soluzione(risultato: Risultato, istanza) -> None:
    """Riepilogo, rotte, regret e rifiuti di una singola risoluzione."""
    titolo = TABELLA.get(risultato.obiettivo, "Rotte")
    print(f"\n{titolo} - rotte dei veicoli (senza deposito) dell'istanza "
          f"{risultato.istanza}, obiettivo {risultato.obiettivo}\n")

    if not risultato.ha_soluzione:
        print(f"  nessuna soluzione ({risultato.stato}, {risultato.tempo:.1f} s)")
        return

    c = risultato.canonici
    serviti = istanza.n - round(c["rifiuti"])
    print(f"  stato {risultato.stato} | obj {risultato.obj:.1f} | costo {c['costo']:.1f} | "
          f"regret {c['regret']:.1f} | regret max {c['regret_max']:.1f}")
    print(f"  serviti {serviti}/{istanza.n} | veicoli usati {len(risultato.rotte)}/{istanza.K} "
          f"| {risultato.tempo:.2f} s\n")

    for numero, rotta in enumerate(ordina(risultato.rotte), start=1):
        stampa_tour(numero, rotta)

    regret = regret_per_utente(risultato, istanza)
    positivi, rifiutati = positivi_e_rifiutati(regret, istanza)
    print()
    if positivi:
        testo = ", ".join(f"{i}: {r:.1f}" for i, r in positivi.items())
        print(f"  regret positivo (minuti): {testo}")
        print(f"  gli altri {len(regret) - len(positivi)} utenti serviti viaggiano "
              f"senza perdita di tempo")
    else:
        print("  nessun utente subisce perdita di tempo: regret nullo per tutti")

    if rifiutati:
        print(f"  utenti rifiutati: {', '.join(str(i) for i in rifiutati)}")


def stampa_confronto(prima: Risultato, seconda: Risultato, istanza) -> None:
    """
    Confronto fra due soluzioni della stessa istanza: e' il commento che nel paper
    accompagna le Figure 2 e 3 (chi viene rifiutato, a chi migliora il regret).
    """
    if not (prima.ha_soluzione and seconda.ha_soluzione):
        return

    a = regret_per_utente(prima, istanza)
    b = regret_per_utente(seconda, istanza)

    print(f"\nConfronto {prima.obiettivo} -> {seconda.obiettivo}\n")

    persi = sorted(set(a) - set(b))
    if persi:
        print(f"  serviti solo con {prima.obiettivo}: {', '.join(str(i) for i in persi)}")
    nuovi = sorted(set(b) - set(a))
    if nuovi:
        print(f"  serviti solo con {seconda.obiettivo}: {', '.join(str(i) for i in nuovi)}")

    comuni = sorted(set(a) & set(b))
    cambiati = [(i, a[i], b[i]) for i in comuni if abs(a[i] - b[i]) > 1e-6]
    if cambiati:
        print("  utenti serviti da entrambe con regret diverso (minuti):")
        for i, prima_r, dopo_r in cambiati:
            print(f"    utente {i:>3}: {prima_r:6.1f} -> {dopo_r:6.1f}")
    else:
        print("  nessun utente servito da entrambe cambia regret")

    for nome, risultato in ((prima.obiettivo, prima), (seconda.obiettivo, seconda)):
        c = risultato.canonici
        print(f"  {nome:>7}: costo {c['costo']:8.1f}  regret {c['regret']:7.1f}  "
              f"serviti {istanza.n - round(c['rifiuti'])}/{istanza.n}")


# ----------------------------------------------------------------------
# Immagine PNG
# ----------------------------------------------------------------------

def png_soluzione(risultato: Risultato, istanza, percorso: Path) -> None:
    """
    Disegna le rotte in un PNG da affiancare alla Tabella 13 o 14 del paper. Le rotte
    hanno lunghezze diverse, quindi tutte le celle hanno la larghezza della più larga,
    così i veicoli restano incolonnati fra loro.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")                     # nessuna finestra: si salva e basta
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("per i PNG serve matplotlib:  pip install matplotlib")

    if not risultato.ha_soluzione:
        print(f"  nessuna soluzione per {risultato.obiettivo}: PNG non generato")
        return

    tour = [(numero, eventi(rotta))
            for numero, rotta in enumerate(ordina(risultato.rotte), start=1)]
    tour = [(numero, lista) for numero, lista in tour if lista]

    testo_celle = [t for _, lista in tour for coppia in lista
                   for t in (coppia[0], f"{coppia[1]:.1f}")]
    cella = max(len(t) for t in testo_celle) + 2
    massimo_eventi = max(len(lista) for _, lista in tour)
    etichetta = "  Tour 00   Location  "

    # righe di testo sotto la tabella: riepilogo, regret, rifiuti
    positivi, rifiutati = positivi_e_rifiutati(regret_per_utente(risultato, istanza), istanza)
    piede = []
    if positivi:
        piede += textwrap.wrap(
            "regret positivo (minuti): "
            + ", ".join(f"{i}: {r:.1f}" for i, r in positivi.items()),
            width=int((len(etichetta) + massimo_eventi * cella) * 1.2))
    if rifiutati:
        piede.append("utenti rifiutati: " + ", ".join(str(i) for i in rifiutati))

    caratteri = len(etichetta) + massimo_eventi * cella
    righe = 3 * len(tour) - 1                     # due righe per veicolo piu' una vuota
    larghezza_fig = max(caratteri * PASSO_CARATTERE, 5.0)
    altezza_fig = (MARGINE_ALTO + righe * ALTEZZA_RIGA + MARGINE_BASSO
                   + 0.20 * len(piede))

    fig = plt.figure(figsize=(larghezza_fig, altezza_fig), dpi=200)
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.96))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    passo = ALTEZZA_RIGA / altezza_fig
    x_etichetta = len(etichetta) / caratteri
    c = risultato.canonici
    serviti = istanza.n - round(c["rifiuti"])

    y = 1.0 - 0.26 / altezza_fig
    ax.text(0.0, y, f"{TABELLA.get(risultato.obiettivo, 'Rotte')} - rotte dei veicoli "
                    f"(senza deposito), obiettivo {risultato.obiettivo}",
            fontsize=11, fontweight="bold", va="center")
    ax.text(0.0, y - 0.22 / altezza_fig,
            f"{risultato.istanza}: costo {c['costo']:.1f}, regret {c['regret']:.1f}, "
            f"regret max {c['regret_max']:.1f}, serviti {serviti}/{istanza.n}, "
            f"veicoli {len(risultato.rotte)}/{istanza.K}, {risultato.tempo:.2f} s",
            fontsize=8, color="0.35", va="center")

    y = 1.0 - MARGINE_ALTO / altezza_fig
    for numero, lista in tour:
        ax.plot([0, 1], [y + passo * 0.6] * 2, color="0.8", linewidth=0.6)
        for riga, (nome, prendi) in enumerate((("Location", lambda e: e[0]),
                                               ("Time[m]", lambda e: f"{e[1]:.1f}"))):
            y_riga = y - riga * passo
            if riga == 0:
                ax.text(0.0, y_riga, f"Tour {numero}", fontsize=9, fontweight="bold",
                        va="center")
            ax.text(x_etichetta, y_riga, nome, fontsize=8, color="0.35",
                    ha="right", va="center")
            for k, evento in enumerate(lista):
                x = x_etichetta + (k + 1) * cella / caratteri
                ax.text(x - 0.004, y_riga, prendi(evento), fontsize=9,
                        ha="right", va="center",
                        fontweight="bold" if riga == 0 else "normal")
        y -= 3 * passo

    y += passo
    for testo in piede:
        ax.text(0.0, y, testo, fontsize=7, color="0.35", va="center")
        y -= 0.20 / altezza_fig

    percorso.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(percorso, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  immagine in {percorso}")


# ----------------------------------------------------------------------
# Salvataggio
# ----------------------------------------------------------------------

def salva(risultati: list[Risultato], istanza, cartella: Path) -> None:
    """
    Scrive un CSV con una riga per evento: da li' si compone la tabella nella relazione
    senza dover ricopiare i numeri a mano.
    """
    cartella.mkdir(parents=True, exist_ok=True)
    percorso = cartella / f"rotte_{istanza.nome}.csv"
    with percorso.open("w", newline="", encoding="utf-8") as f:
        scrittore = csv.writer(f)
        scrittore.writerow(["istanza", "obiettivo", "tour", "posizione", "evento",
                            "utente", "tipo", "tempo"])
        for risultato in risultati:
            if not risultato.ha_soluzione:
                continue
            for numero, rotta in enumerate(ordina(risultato.rotte), start=1):
                for posizione, (etichetta, tempo) in enumerate(eventi(rotta), start=1):
                    utente = int(etichetta[:-1])
                    tipo = "pickup" if etichetta.endswith("+") else "dropoff"
                    scrittore.writerow([istanza.nome, risultato.obiettivo, numero,
                                        posizione, etichetta, utente, tipo,
                                        round(tempo, 4)])
    print(f"\n  salvato in {percorso}")


# ----------------------------------------------------------------------
# Programma principale
# -----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Tabelle 13 e 14: rotte di una singola istanza.")
    p.add_argument("istanza", type=Path, help="file dell'istanza (JSON Trieste o Cordeau)")
    p.add_argument("--obiettivi", nargs="*", default=list(PREDEFINITI),
                   help=f"obiettivi da risolvere (default: {' '.join(PREDEFINITI)})")
    p.add_argument("--time-limit", type=float, default=600.0)
    p.add_argument("--mip-gap", type=float, default=0.0)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--salva", type=Path, default=None, help="cartella in cui scrivere il CSV")
    p.add_argument("--png", type=Path, default=None,
                   help="cartella in cui scrivere le immagini PNG delle rotte")
    args = p.parse_args()

    istanza = leggi_istanza(args.istanza)
    print(f"{istanza.nome}: Q={istanza.Q}, n={istanza.n}, K={istanza.K}")
    if istanza.n > 30:
        print("  nota: le Tabelle 13/14 del paper usano un'istanza da 20 utenti; "
              "con molti utenti la tabella diventa illeggibile")

    grafo = Grafo.costruisci(istanza)

    risultati = []
    for obiettivo in args.obiettivi:
        m = costruisci_con_obiettivo(grafo, obiettivo, variante="II", log=False,
                                     time_limit=args.time_limit, mip_gap=args.mip_gap,
                                     threads=args.threads)
        risultato = risolvi(m)
        risultati.append(risultato)
        stampa_soluzione(risultato, istanza)
        if args.png:
            png_soluzione(risultato, istanza,
                          args.png / f"rotte_{istanza.nome}_{obiettivo}.png")

    if len(risultati) == 2:
        stampa_confronto(risultati[0], risultati[1], istanza)

    if args.salva:
        salva(risultati, istanza, args.salva)


if __name__ == "__main__":
    main()