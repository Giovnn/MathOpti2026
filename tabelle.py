"""
Dai risultati di esperimenti.py alle Tabelle 8-12 della Sez. 4.2 del paper.
Legge il CSV (una riga per istanza e obiettivo) e non risolve nulla: tutte le tabelle
sono aggregazioni dello stesso file.

    Tabella 8    valori medi con gli obiettivi puri f_c e f_r
    Tabella 9    valori medi con gli obiettivi composti f_cr e f_rcr
    Tabella 10   valori medi con gli obiettivi di regret massimo f_rmax e f_crmax
    Tabella 11   variazioni percentuali fra sei coppie di obiettivi
    Tabella 12   variazioni percentuali fra i tre obiettivi composti

Nelle Tabelle 8-10 si media dentro ogni gruppo (Q, n). Nelle Tabelle 11-12 si calcola
prima la variazione percentuale istanza per istanza e solo dopo si media: i due calcoli
non coincidono, e sui tempi la differenza è grande, perché vanno da centesimi di
secondo al time limit.

Convenzioni prese dal paper:
  - costo, regret, gap, ... si mediano solo sui run con una soluzione; dove ne manca
    qualcuna, sotto la tabella compare una nota;
  - il tempo si media su tutti i run del gruppo, anche quelli senza soluzione;
  - la riga Avg delle Tabelle 11 e 12 è la media semplice delle sei righe della
    famiglia, non pesata sul numero di istanze.

Nei nomi di colonna "d_" sta per la variazione percentuale (il Delta del paper).

Uso (dalla cartella del progetto):
    python tabelle.py risultati/risultati_trieste.csv
    python tabelle.py risultati/risultati_trieste.csv --tabelle 8 11
    python tabelle.py risultati/risultati_trieste.csv --salva tabelle --png tabelle --confronto
"""

import argparse
import textwrap
from pathlib import Path

import pandas as pd


# Le righe delle tabelle sono i gruppi (Q, n): tutte le istanze con la stessa capienza
# e lo stesso numero di utenti finiscono in una sola riga, mediate.
GRUPPO = ["Q", "n"]

# Colonne del CSV che si mediano solo sui run che hanno prodotto una soluzione.
VALORI = ["obj", "costo", "regret", "regret_max", "ar", "gap"]

# Come si chiamano gli obiettivi nelle intestazioni stampate.
NOMI = {"fc": "f_c", "fr": "f_r", "fcr": "f_cr",
        "frcr": "f_rcr", "frmax": "f_rmax", "fcrmax": "f_crmax"}

# Etichetta di colonna -> colonna del CSV. Per il regret si usano le colonne canoniche
# (schedule minimo), non quelle _grezzo: vedi results.py.
SORGENTE = {"Obj.v.": "obj", "f_c": "costo", "f_r": "regret",
            "f_rmax": "regret_max", "a.r.": "ar", "Gap": "gap", "CPU": "CPU"}

# Struttura delle Tabelle 8, 9 e 10: per ogni blocco, l'obiettivo usato nel MILP e le
# grandezze misurate sulla soluzione. Sotto f_c non c'e' Gap perche' nessun run con
# obiettivo di costo tocca il time limit: sarebbe una colonna di zeri, come nel paper.
STRUTTURA = {
    8: [("fc", ["f_c", "f_r", "CPU"]),
        ("fr", ["f_c", "f_r", "Gap", "CPU"])],
    9: [("fcr", ["Obj.v.", "f_c", "f_r", "Gap", "CPU"]),
        ("frcr", ["Obj.v.", "f_c", "f_r", "a.r.", "Gap", "CPU"])],
    10: [("frmax", ["f_c", "f_r", "f_rmax", "Gap", "CPU"]),
         ("fcrmax", ["Obj.v.", "f_c", "f_r", "f_rmax", "Gap", "CPU"])],
}

# Struttura delle Tabelle 11 e 12: (obiettivo nuovo, obiettivo di riferimento, metriche).
CONFRONTI = {
    11: [("fr", "fc", ["f_c", "f_r"]),          # quanto costa passare al regret puro
         ("fcr", "fc", ["f_c", "CPU"]),         # quanto costa il compromesso
         ("fcr", "fr", ["f_r", "CPU"]),         # il peso alpha e' ben scelto?
         ("frmax", "fc", ["f_c", "f_r"]),       # stesso trade-off, versione equita'
         ("fcrmax", "fc", ["f_c", "CPU"]),      # quanto costa il compromesso equita'
         ("fcrmax", "frmax", ["f_rmax", "CPU"])],  # il peso beta e' ben scelto?
    12: [("fcrmax", "fcr", ["f_c", "f_r", "CPU"]),        # quale compromesso conviene
         ("frcr", "fcr", ["f_c", "f_r", "CPU", "a.r."])],  # cosa si guadagna rifiutando
}

# Colonna del CSV su cui si calcola ogni variazione della Tabella 11 e 12.
SORGENTE_VARIAZIONI = {"f_c": "costo", "f_r": "regret", "f_rmax": "regret_max",
                       "a.r.": "ar", "CPU": "tempo"}


# ----------------------------------------------------------------------
# Lettura
# ----------------------------------------------------------------------

def leggi(percorso: Path) -> pd.DataFrame:
    """Carica il CSV dei run e sistema le unita' di misura."""
    d = pd.read_csv(percorso)
    # Gurobi restituisce il gap come frazione (0.157): il paper lo stampa in percentuale.
    d["gap"] = d["gap"] * 100
    return d


def controlla(d: pd.DataFrame) -> None:
    """Avvisi non bloccanti su cio' che potrebbe rendere le tabelle incomplete."""
    attesi = set(NOMI)
    mancanti = attesi - set(d["obiettivo"].unique())
    if mancanti:
        print(f"  ATTENZIONE: obiettivi assenti dal file: {', '.join(sorted(mancanti))}")

    per_istanza = d.groupby("istanza")["obiettivo"].nunique()
    incomplete = per_istanza[per_istanza < len(attesi)]
    if len(incomplete):
        print(f"  ATTENZIONE: {len(incomplete)} istanze non hanno tutti e sei i run")

    limiti = d["time_limit"].unique()
    if len(limiti) > 1:
        print(f"  ATTENZIONE: time limit non uniforme: {limiti}")

    senza = d[d["obj"].isna()]
    if len(senza):
        elenco = ", ".join(f"{r.istanza}/{r.obiettivo}" for r in senza.itertuples())
        print(f"  {len(senza)} run senza soluzione, esclusi dalle medie dei valori: {elenco}")


def numerosita(d: pd.DataFrame) -> pd.Series:
    """Quante istanze ha ogni gruppo (Q, n)."""
    return d.drop_duplicates("istanza").groupby(GRUPPO).size()


# ----------------------------------------------------------------------
# Medie per gruppo (Tabelle 8, 9, 10)
# ----------------------------------------------------------------------

def medie(d: pd.DataFrame, obiettivo: str) -> pd.DataFrame:
    """
    Medie di un blocco di tabella: un obiettivo, una riga per gruppo (Q, n).

    Le colonne "istanze" e "usate" non finiscono nella tabella: servono alle note.
    """
    righe = d[d["obiettivo"] == obiettivo]
    if righe.empty:
        raise SystemExit(f"nessun run con obiettivo '{obiettivo}' nel file")

    risolte = righe[righe["obj"].notna()]
    indice = righe.groupby(GRUPPO).size().index          # tutti i gruppi presenti

    m = risolte.groupby(GRUPPO)[VALORI].mean().reindex(indice)
    m["CPU"] = righe.groupby(GRUPPO)["tempo"].mean()     # il tempo si media su tutti
    m["istanze"] = righe.groupby(GRUPPO).size()
    m["usate"] = risolte.groupby(GRUPPO).size().reindex(indice).fillna(0).astype(int)
    return m


def tabella_medie(d: pd.DataFrame, numero: int) -> tuple[pd.DataFrame, list]:
    """Costruisce una delle Tabelle 8, 9, 10 secondo STRUTTURA."""
    struttura = STRUTTURA[numero]
    calcolate = {obiettivo: medie(d, obiettivo) for obiettivo, _ in struttura}

    primo = next(iter(calcolate.values()))
    t = pd.DataFrame(index=primo.index)
    t["N"] = primo["istanze"]

    blocchi = [("", "", ["N"])]
    for obiettivo, colonne in struttura:
        m = calcolate[obiettivo]
        for etichetta in colonne:
            t[f"{obiettivo}|{etichetta}"] = m[SORGENTE[etichetta]]
        blocchi.append((obiettivo, NOMI[obiettivo], colonne))

    t["note"] = note({NOMI[ob]: m["usate"] for ob, m in calcolate.items()},
                     primo["istanze"], "media")
    return t, blocchi


# ----------------------------------------------------------------------
# Variazioni per istanza (Tabelle 11, 12)
# ----------------------------------------------------------------------

def variazione(d: pd.DataFrame, nuovo: str, base: str, colonna: str) -> pd.Series:
    """
    Variazione percentuale fra due obiettivi su ogni singola istanza: le due serie sono
    indicizzate per istanza, quindi la sottrazione confronta sempre la stessa istanza.
    Un'istanza senza soluzione con uno dei due obiettivi, o con riferimento zero, dà un
    valore vuoto e non entra nella media.
    """
    def serie(obiettivo: str) -> pd.Series:
        righe = d[d["obiettivo"] == obiettivo]
        return righe.set_index(["Q", "n", "istanza"])[colonna]

    a, b = serie(nuovo), serie(base)
    b = b.where(b != 0)
    return (a - b) / b * 100


def tabella_variazioni(d: pd.DataFrame, numero: int) -> tuple[pd.DataFrame, list]:
    """Costruisce la Tabella 11 o la 12 secondo CONFRONTI, con la riga Avg finale."""
    istanze = numerosita(d)
    t = pd.DataFrame(index=istanze.index)
    t["N"] = istanze

    blocchi = [("", "", ["N"])]
    usate = {}
    for nuovo, base, metriche in CONFRONTI[numero]:
        prefisso = f"{nuovo}_vs_{base}"
        etichetta = f"{NOMI[nuovo]} vs {NOMI[base]}"
        minimo = None
        for metrica in metriche:
            v = variazione(d, nuovo, base, SORGENTE_VARIAZIONI[metrica])
            t[f"{prefisso}|d_{metrica}"] = v.groupby(level=GRUPPO).mean()
            conteggio = v.groupby(level=GRUPPO).count()
            minimo = conteggio if minimo is None else minimo.combine(conteggio, min)
        blocchi.append((prefisso, etichetta, [f"d_{m}" for m in metriche]))
        usate[etichetta] = minimo

    t["note"] = note(usate, istanze, "confronto")
    return aggiungi_avg(t), blocchi


def aggiungi_avg(t: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge in fondo a ogni famiglia (Q3, Q6) la riga Avg del paper."""
    numeriche = [c for c in t.columns if c != "note"]
    pezzi = []
    for q in sorted(set(t.index.get_level_values("Q"))):
        blocco = t[t.index.get_level_values("Q") == q]
        pezzi.append(blocco)

        media = blocco[numeriche].mean()
        media["N"] = float("nan")                        # Avg non ha un numero di istanze
        riga = pd.DataFrame([media],
                            index=pd.MultiIndex.from_tuples([(q, "Avg")], names=GRUPPO))
        riga["note"] = ""
        pezzi.append(riga)
    return pd.concat(pezzi)


# ----------------------------------------------------------------------
# Note a pie' di tabella
# ----------------------------------------------------------------------

def note(usate: dict[str, pd.Series], totale: pd.Series, verbo: str) -> pd.Series:
    """
    Note nello stile delle note a/b della Tabella 8 del paper: dove qualche istanza
    manca all'appello, si dice su quante e' stata costruita la media.
    """
    testi = {}
    for chiave in totale.index:
        pezzi = []
        for etichetta, quante in usate.items():
            disponibili = int(quante.get(chiave, 0))
            tutte = int(totale.loc[chiave])
            if disponibili < tutte:
                pezzi.append(f"{etichetta}: {verbo} su {disponibili} istanze su {tutte}")
        testi[chiave] = "; ".join(pezzi)
    return pd.Series(testi)


# ----------------------------------------------------------------------
# Stampa e salvataggio
# ----------------------------------------------------------------------

def formatta(nome: str, valore) -> str:
    """Un numero come lo stampa il paper: i decimali dipendono dalla colonna."""
    if pd.isna(valore):
        return "-"
    if nome == "N":
        return f"{int(valore)}"
    if nome == "CPU":
        # tempi piccoli con due decimali, tempi grandi senza
        return f"{valore:.2f}" if valore < 10 else f"{valore:.0f}"
    if nome in ("Gap", "a.r.") or nome.startswith("d_"):
        return f"{valore:.0f}"
    return f"{valore:.1f}"


def stampa(titolo: str, t: pd.DataFrame, blocchi: list,
           indice: tuple = ("Q", "n")) -> None:
    """Stampa la tabella a terminale con la doppia intestazione: blocco sopra, colonne sotto."""
    intestazione, corpo, spanne = celle(t, blocchi, indice)

    larghezze = [max(len(intestazione[k]), *(len(r[k]) for r in corpo)) + 2
                 for k in range(len(intestazione))]

    # se l'etichetta di un blocco e' piu' larga delle sue colonne, le colonne si allargano
    # quanto basta, altrimenti le etichette di due blocchi vicini si toccherebbero
    for etichetta, inizio, quante in spanne:
        manca = len(etichetta) + 2 - sum(larghezze[inizio:inizio + quante])
        for j in range(max(manca, 0)):
            larghezze[inizio + j % quante] += 1

    # prima riga di intestazione: il nome del blocco, centrato sulle sue colonne
    prima = ""
    fine_precedente = 0
    for etichetta, inizio, quante in spanne:
        prima += " " * (sum(larghezze[fine_precedente:inizio]))
        prima += etichetta.center(sum(larghezze[inizio:inizio + quante]))
        fine_precedente = inizio + quante

    print(f"\n{titolo}\n")
    print(prima)
    print("".join(v.rjust(l) for v, l in zip(intestazione, larghezze)))
    print("-" * sum(larghezze))
    for valori in corpo:
        print("".join(v.rjust(l) for v, l in zip(valori, larghezze)))

    for chiave, testo in t["note"].items():
        if testo:
            print(f"  Q{chiave[0]}, {indice[1]}={chiave[1]}: {testo}")


def salva(t: pd.DataFrame, cartella: Path, nome: str) -> None:
    """Scrive la tabella in CSV, per poterla riprendere nella relazione."""
    cartella.mkdir(parents=True, exist_ok=True)
    percorso = cartella / f"{nome}.csv"
    t.round(4).to_csv(percorso, encoding="utf-8")
    print(f"  salvata in {percorso}")


# ----------------------------------------------------------------------
# Immagine PNG
# ----------------------------------------------------------------------

# Misure del disegno, in pollici
PASSO_CARATTERE = 0.085      # larghezza di un carattere
ALTEZZA_RIGA = 0.26          # altezza di una riga
MARGINE_ALTO = 0.75          # spazio per titolo e sottotitolo
MARGINE_BASSO = 0.30         # spazio di base sotto la tabella


def celle(t: pd.DataFrame, blocchi: list,
          indice: tuple = ("Q", "n")) -> tuple[list, list, list]:
    """
    Contenuto della tabella come testo (etichette dei blocchi, intestazioni, righe di
    dati), usato sia dalla stampa a terminale sia dal PNG.
    """
    colonne = [(prefisso, c) for prefisso, _, gruppo in blocchi for c in gruppo]
    intestazione = list(indice) + [c for _, c in colonne]

    corpo = []
    for chiave, riga in t.iterrows():
        valori = [str(x) for x in chiave]
        for prefisso, col in colonne:
            valori.append(formatta(col, riga[f"{prefisso}|{col}" if prefisso else col]))
        corpo.append(valori)

    # (etichetta, prima colonna, quante colonne) per la riga di intestazione superiore
    spanne = []
    k = len(indice)
    for _, etichetta, gruppo in blocchi:
        if etichetta:
            spanne.append((etichetta, k, len(gruppo)))
        k += len(gruppo)
    return intestazione, corpo, spanne


def png(titolo: str, sottotitolo: str, t: pd.DataFrame, blocchi: list, percorso: Path,
        indice: tuple = ("Q", "n")) -> None:
    """
    Disegna la tabella in un PNG da affiancare a quella del paper nella relazione: il
    testo viene scritto cella per cella, così l'impaginazione è la stessa del terminale.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")                     # nessuna finestra: si salva e basta
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("per i PNG serve matplotlib:  pip install matplotlib")

    intestazione, corpo, spanne = celle(t, blocchi, indice)

    # larghezza di ogni colonna in caratteri, dal contenuto piu' lungo
    larghezze = [max(len(intestazione[k]), *(len(r[k]) for r in corpo)) + 2
                 for k in range(len(intestazione))]
    for etichetta, inizio, quante in spanne:
        manca = len(etichetta) + 2 - sum(larghezze[inizio:inizio + quante])
        for j in range(max(manca, 0)):
            larghezze[inizio + j % quante] += 1

    totale = sum(larghezze)

    # le note vanno mandate a capo, altrimenti escono dal bordo dell'immagine: sono
    # scritte in corpo piu' piccolo, quindi in una riga ci sta circa un quinto in piu'
    note_presenti = []
    for chiave, testo in t["note"].items():
        if testo:
            note_presenti += textwrap.wrap(f"Q{chiave[0]}, {indice[1]}={chiave[1]}: {testo}",
                                           width=int(totale * 1.2))
    bordi = [sum(larghezze[:k + 1]) / totale for k in range(len(larghezze))]

    def centro(k: int) -> float:
        sinistra = bordi[k - 1] if k else 0.0
        return (sinistra + bordi[k]) / 2

    righe_testo = 2 + len(corpo)
    altezza_note = 0.20 * len(note_presenti)
    larghezza_fig = max(totale * PASSO_CARATTERE, 5.0)
    altezza_fig = MARGINE_ALTO + righe_testo * ALTEZZA_RIGA + MARGINE_BASSO + altezza_note

    fig = plt.figure(figsize=(larghezza_fig, altezza_fig), dpi=200)
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.96))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # in coordinate 0-1: y=1 in alto. Si scende di un passo per ogni riga.
    passo = ALTEZZA_RIGA / altezza_fig
    y_titolo = 1.0 - 0.26 / altezza_fig
    y_blocchi = 1.0 - MARGINE_ALTO / altezza_fig
    y_colonne = y_blocchi - passo
    y_prima = y_colonne - passo * 1.25

    ax.text(0.0, y_titolo, titolo, fontsize=11, fontweight="bold", va="center")
    if sottotitolo:
        ax.text(0.0, y_titolo - 0.22 / altezza_fig, sottotitolo, fontsize=8,
                color="0.35", va="center")

    for etichetta, inizio, quante in spanne:
        sinistra = bordi[inizio - 1] if inizio else 0.0
        destra = bordi[inizio + quante - 1]
        ax.text((sinistra + destra) / 2, y_blocchi, etichetta, fontsize=9,
                fontstyle="italic", ha="center", va="center")
        ax.plot([sinistra + 0.004, destra - 0.004], [y_blocchi - passo * 0.45] * 2,
                color="0.45", linewidth=0.7)

    for k, nome in enumerate(intestazione):
        ax.text(centro(k), y_colonne, nome, fontsize=9, ha="center", va="center")
    ax.plot([0, 1], [y_colonne - passo * 0.5] * 2, color="0.1", linewidth=1.0)

    for j, valori in enumerate(corpo):
        y = y_prima - j * passo
        finale = valori[1] == "Avg"
        cambia_famiglia = j > 0 and valori[0] != corpo[j - 1][0]
        if finale or cambia_famiglia:
            ax.plot([0, 1], [y + passo * 0.5] * 2, color="0.75", linewidth=0.6)
        for k, testo in enumerate(valori):
            ax.text(bordi[k] - 0.004, y, testo, fontsize=9, ha="right", va="center",
                    fontweight="bold" if finale else "normal")
    ax.plot([0, 1], [y_prima - (len(corpo) - 0.5) * passo] * 2, color="0.1", linewidth=1.0)

    y = y_prima - (len(corpo) + 0.4) * passo
    for testo in note_presenti:
        ax.text(0.0, y, testo, fontsize=7, color="0.35", va="center")
        y -= 0.20 / altezza_fig

    cartella = percorso.parent
    cartella.mkdir(parents=True, exist_ok=True)
    fig.savefig(percorso, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  immagine in {percorso}")


TITOLI = {
    8: "Tabella 8 - valori medi con gli obiettivi f_c e f_r",
    9: "Tabella 9 - valori medi con gli obiettivi f_cr e f_rcr",
    10: "Tabella 10 - valori medi con gli obiettivi f_rmax e f_crmax",
    11: "Tabella 11 - variazioni percentuali fra f_c, f_r, f_cr, f_rmax e f_crmax",
    12: "Tabella 12 - variazioni percentuali fra f_cr, f_crmax e f_rcr",
}


# ----------------------------------------------------------------------
# Confronto con il paper
# ----------------------------------------------------------------------

# Righe Avg delle Tabelle 11 e 12 del paper (Wuppertal), ricopiate dall'articolo.
# Servono solo a questa tabella di confronto: nessun altro calcolo le usa.
PAPER = {
    3: {"fr_vs_fc|d_f_c": 27, "fr_vs_fc|d_f_r": -84,
        "fcr_vs_fc|d_f_c": 16, "fcr_vs_fr|d_f_r": 10,
        "frmax_vs_fc|d_f_c": 27, "frmax_vs_fc|d_f_r": -59,
        "fcrmax_vs_frmax|d_f_rmax": 4, "fcrmax_vs_fcr|d_f_r": 312,
        "frcr_vs_fcr|d_f_c": -7, "frcr_vs_fcr|d_f_r": -55, "frcr_vs_fcr|d_a.r.": -5},
    6: {"fr_vs_fc|d_f_c": 16, "fr_vs_fc|d_f_r": -80,
        "fcr_vs_fc|d_f_c": 9, "fcr_vs_fr|d_f_r": 5,
        "frmax_vs_fc|d_f_c": 18, "frmax_vs_fc|d_f_r": -50,
        "fcrmax_vs_frmax|d_f_rmax": 1, "fcrmax_vs_fcr|d_f_r": 238,
        "frcr_vs_fcr|d_f_c": -8, "frcr_vs_fcr|d_f_r": -65, "frcr_vs_fcr|d_a.r.": -5},
}

# Quali indicatori mettere a confronto, nell'ordine in cui compaiono nella tabella.
CONFRONTO = [("fr_vs_fc", "f_r vs f_c", ["d_f_c", "d_f_r"]),
             ("fcr_vs_fc", "f_cr vs f_c", ["d_f_c"]),
             ("fcr_vs_fr", "f_cr vs f_r", ["d_f_r"]),
             ("frmax_vs_fc", "f_rmax vs f_c", ["d_f_c", "d_f_r"]),
             ("fcrmax_vs_frmax", "f_crmax vs f_rmax", ["d_f_rmax"]),
             ("fcrmax_vs_fcr", "f_crmax vs f_cr", ["d_f_r"]),
             ("frcr_vs_fcr", "f_rcr vs f_cr", ["d_f_c", "d_f_r", "d_a.r."])]


def tabella_confronto(d: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Affianca, indicatore per indicatore, le righe Avg del paper (Wuppertal) a quelle di
    Trieste. Contano i segni e gli ordini di grandezza, non i valori esatti: città,
    solver e time limit sono diversi.
    """
    variazioni = {11: tabella_variazioni(d, 11)[0], 12: tabella_variazioni(d, 12)[0]}

    ordinate = [f"{prefisso}|{c}" for prefisso, _, colonne in CONFRONTO for c in colonne]
    righe = {}
    for q in (3, 6):
        nostri = {}
        for nome in ordinate:
            fonte = variazioni[11] if nome in variazioni[11].columns else variazioni[12]
            nostri[nome] = fonte.loc[(q, "Avg"), nome]
        righe[(q, "Wuppertal")] = {k: PAPER[q][k] for k in ordinate}
        righe[(q, "Trieste")] = nostri

    t = pd.DataFrame.from_dict(righe, orient="index")[ordinate]
    t.index = pd.MultiIndex.from_tuples(t.index, names=["Q", "dati"])
    t["note"] = ""

    blocchi = [(prefisso, etichetta, colonne) for prefisso, etichetta, colonne in CONFRONTO]
    return t, blocchi


def costruisci(d: pd.DataFrame, numero: int) -> tuple[pd.DataFrame, list]:
    """Smista fra i due tipi di tabella."""
    if numero in STRUTTURA:
        return tabella_medie(d, numero)
    return tabella_variazioni(d, numero)


# ----------------------------------------------------------------------
# Programma principale
# ----------------------------------------------------------------------

def sottotitolo(d: pd.DataFrame) -> str:
    """Riga di contesto sotto il titolo dell'immagine: dice su cosa sono stati fatti i run."""
    limite = d["time_limit"].dropna().unique()
    testo_limite = f"{limite[0]:.0f} s" if len(limite) == 1 else "variabile"
    return (f"Trieste - {d['istanza'].nunique()} istanze, Model II, Gurobi, "
            f"time limit {testo_limite}; medie per gruppo (Q, n)")


def main() -> None:
    p = argparse.ArgumentParser(description="Tabelle 8-12 della Sez. 4.2 dal CSV dei run.")
    p.add_argument("csv", type=Path, help="file prodotto da esperimenti.py")
    p.add_argument("--salva", type=Path, default=None, help="cartella in cui scrivere i CSV")
    p.add_argument("--png", type=Path, default=None,
                   help="cartella in cui scrivere le immagini PNG delle tabelle")
    p.add_argument("--tabelle", type=int, nargs="*", default=sorted(TITOLI),
                   choices=sorted(TITOLI), help="quali tabelle produrre (default: tutte)")
    p.add_argument("--confronto", action="store_true",
                   help="aggiunge la tabella che affianca le righe Avg del paper alle nostre")
    args = p.parse_args()

    d = leggi(args.csv)
    print(f"{len(d)} run, {d['istanza'].nunique()} istanze, "
          f"obiettivi: {', '.join(sorted(d['obiettivo'].unique()))}")
    controlla(d)

    contesto = sottotitolo(d)
    for numero in args.tabelle:
        t, blocchi = costruisci(d, numero)
        stampa(TITOLI[numero], t, blocchi)
        if args.salva:
            salva(t, args.salva, f"tabella_{numero}")
        if args.png:
            png(TITOLI[numero], contesto, t, blocchi,
                args.png / f"tabella_{numero}.png")

    if args.confronto:
        titolo = "Confronto delle tendenze medie: Wuppertal (paper) e Trieste"
        nota = ("variazioni percentuali medie per famiglia; Wuppertal: CPLEX, time limit "
                "7200 s, 5 istanze per gruppo")
        t, blocchi = tabella_confronto(d)
        stampa(titolo, t, blocchi, indice=("Q", "dati"))
        if args.salva:
            salva(t, args.salva, "confronto")
        if args.png:
            png(titolo, nota, t, blocchi, args.png / "confronto.png", indice=("Q", "dati"))


if __name__ == "__main__":
    main()