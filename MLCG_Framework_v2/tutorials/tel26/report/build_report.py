#!/usr/bin/env python3
"""Report PDF AA contro CG-ML per TEL26 (dati: report_data.npz, figure: make_figs.py)."""
import json
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FD + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", FD + "DejaVuSans-BoldOblique.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402
registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, INK2, RULE = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d6d5d0")
ACC = colors.HexColor("#2a78d6")
TINT = colors.HexColor("#f3f2ee")

S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=19, leading=24, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=10.5, leading=14, textColor=INK2, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=13, leading=17, textColor=INK, spaceBefore=12, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=10.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.3, leading=13.4, textColor=INK, spaceAfter=6,
                           alignment=TA_LEFT),
    "bul": ParagraphStyle("bul", fontName="DV", fontSize=9.3, leading=13.4, textColor=INK, leftIndent=12,
                          bulletIndent=2, spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="DV", fontSize=8.2, leading=11.2, textColor=INK2, spaceBefore=3,
                          spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.2, leading=10.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8.2, leading=10.6, textColor=INK),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9.3, leading=13.4, textColor=INK, leftIndent=12,
                          bulletIndent=2, spaceAfter=3),
}

d = np.load("report_data_v2.npz")
meta = json.loads(str(d["meta_json"]))
st = json.load(open("fig_stats.json"))
OV = meta["overlaps"]
CH = meta["channels"]
cv = st["cv"]
ms = st["msd"]
ts = {e["label"]: e for e in meta["timescales"]}
AA_NSDAY, CG_NSDAY = 137.0, 54.2
R = CG_NSDAY / AA_NSDAY


def it(x, n=3):
    """numero con virgola decimale"""
    return f"{x:.{n}f}".replace(".", ",")


def P(t, s="body"):
    return Paragraph(t, S[s])


def B(t, s="bul"):
    return Paragraph(t, S[s], bulletText="•")


def fig(path, width_cm, caption):
    img = Image(path)
    w = width_cm * cm
    img.drawHeight = img.imageHeight * w / img.imageWidth
    img.drawWidth = w
    return KeepTogether([img, P(caption, "cap")])


def table(rows, widths, header=True, zebra=True, bold_last=False):
    data = [[c if not isinstance(c, str) else Paragraph(c, S["cellb" if (header and i == 0) else "cell"])
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="LEFT", repeatRows=1 if header else 0)
    sty = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
           ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 5),
           ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE)]
    if header:
        sty += [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2), ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK2)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if (i % 2 == 0):
                sty.append(("BACKGROUND", (0, i), (-1, i), TINT))
    t.setStyle(TableStyle(sty))
    return t


def boxed(flow):
    t = Table([[flow]], colWidths=[17 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACC),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def on_page(c, doc):
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(INK2)
    c.drawString(2 * cm, 1.2 * cm, "TEL26 · confronto AA contro CG-ML · PaiNN-LT / MLCG_Framework_v2")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    c.restoreState()


# ── numeri derivati ──
rate = {k: 6 * ms["AA"][k] * AA_NSDAY / 24 for k in ("D_trans", "D_rot")}
rate_cg = {k: 6 * ms["prod"][k] * CG_NSDAY / 24 for k in ("D_trans", "D_rot")}
tauAA = {k: ts["AA"][k][0] for k in ("tau_rot", "tau_core", "tau_core_sites", "tau_Rg", "tau_loops")}
tauCG = {k: 0.5 * (ts["r0"][k][0] + ts["r1"][k][0]) for k in tauAA}
alpha = {k: tauAA[k] / tauCG[k] for k in tauAA}
ov_min_cg = min(OV[c]["CG-ML"]["intra"] for c in CH)
ov_max_cg = max(OV[c]["CG-ML"]["intra"] for c in CH)
CEIL = [OV[c]["AA meta' 2"]["intra"] for c in CH]

story = []
story += [P("TEL26: confronto fra dinamica all-atom e CG-ML", "title"),
          P("G-quadruplex ibrido 3+1 (2JPZ, (TTAGGG)<sub>4</sub>TT), 10 copie × 26 nt. "
            "Struttura (P(r), g(r), R<sub>g</sub> e coordinate collettive) e dinamica (MSD traslazionale e "
            "rotazionale, anche per ora di GPU), con la descrizione del modello e il confronto con i soli prior. Progetto PaiNN-LT · MLCG_Framework_v2 · 8 ottobre 2026", "sub")]

OVP = {c: OV[c]["lp2"]["intra"] for c in CH}
summary = [
    P("<b>Sintesi</b>", "body"),
    B(f"<b>Il residuo PaiNN porta la struttura al tetto AA.</b> Con lo stesso protocollo (2 × 20 ns, stesso stato "
      f"iniziale) e i soli prior lp2, la sovrapposizione intra di tutti i siti è {it(OVP[CH[0]])}; con PaiNN "
      f"{it(OV[CH[0]]['CG-ML']['intra'])}, contro un tetto di {it(CEIL[0])}. Nel canale "
      f"B3–B3 (legami di Hoogsteen) si passa da {it(OVP[CH[1]])} a {it(OV[CH[1]]['CG-ML']['intra'])}; l'rmsd "
      f"del nucleo scende da {it(cv['rmsd_core']['pri_mean'])} a {it(cv['rmsd_core']['cg_mean'])} nm "
      f"(AA {it(cv['rmsd_core']['aa_mean'])}).", "box"),
    B(f"<b>Nucleo e struttura locale riprodotti entro il rumore del riferimento.</b> Le P(r) intramolecolari "
      f"per canale si sovrappongono a quelle AA per {it(ov_min_cg)}–{it(ov_max_cg)}, contro un tetto "
      f"(metà contro metà della traiettoria AA) di "
      f"{it(min(CEIL))}–{it(max(CEIL))}; "
      f"rmsd del nucleo: sovrapposizione {it(cv['rmsd_core']['ov'], 2)} (tetto {it(cv['rmsd_core']['ceil'], 2)}).", "box"),
    B(f"<b>R<sub>g</sub> medio corretto</b>: {it(cv['Rg']['cg_mean'])} nm contro {it(cv['Rg']['aa_mean'])} nm "
      f"(AA); sovrapposizione {it(cv['Rg']['ov'], 2)} (tetto {it(cv['Rg']['ceil'], 2)}). Le copie AA restano per "
      f"decine di ns su R<sub>g</sub> diversi; nel CG tutte le copie oscillano attorno allo stesso valore.", "box"),
    B(f"<b>Limite: i loop.</b> rmsd dei loop: sovrapposizione {it(cv['rmsd_loops']['ov'], 2)} (tetto "
      f"{it(cv['rmsd_loops']['ceil'], 2)}), media {it(cv['rmsd_loops']['cg_mean'])} contro "
      f"{it(cv['rmsd_loops']['aa_mean'])} nm. Il CG ha un solo bacino, mentre nell'AA ci sono stati metastabili "
      f"che durano ns.", "box"),
    B(f"<b>Efficienza per ora di GPU</b> (A100; 2 replicas per GPU, {it(CG_NSDAY, 1)} ns/day aggregati, contro "
      f"{AA_NSDAY:.0f} ns/day dell'AA): traslazione {rate_cg['D_trans'] / rate['D_trans']:.0f}× (MSD); "
      f"rotazione {alpha['tau_rot'] * R:.0f}× (decorrelazione); rmsd del nucleo "
      f"{alpha['tau_core'] * R:.0f}×, che è il minimo fra gli osservabili confrontabili.", "box"),
]
story += [boxed(summary), Spacer(1, 10)]

# ── 1. Dati ──
story += [P("1. Sistemi, modelli e dati", "h1")]
rows = [
    ["", "All-atom (riferimento)", "CG-ML"],
    ["Rappresentazione", "169 000 atomi, solvente esplicito",
     "10 × 86 = 860 beads (G: backbone S + basi B1–B5; T, A: un sito), solvente implicito"],
    ["Hamiltoniana", "campo di forze atomistico",
     "prior fisici <i>lp2</i> (legami, angoli, diedri, 54 contatti Morse per copia, WCA) + residuo PaiNN "
     "(<i>re1_it30</i>, allenato a entropia relativa); sezione 2"],
    ["Dinamica", "dt 2 fs", "Langevin, dt 4 fs, γ = 2; ESPResSo + plugin PaiNN"],
    ["Traiettorie", f"1 × {it(meta['aa_frames'] * meta['aa_dt_ps'] / 1000, 1)} ns, 10 copie",
     "2 replicas × 20 ns (4 segmenti × 5 ns), 10 copie per replica"],
    ["Frame analizzati", f"{meta['aa_frames']} frame ogni {meta['aa_dt_ps']:.0f} ps (P(r): uno ogni "
     f"{meta['aa_stride']})", f"struttura: ogni {meta['cg_stride']} ps ({meta['runs'][0]['frames_used']} frame); "
     "MSD e tempi: ogni 1 ps"],
    ["Controllo", "–", "soli prior lp2 (PaiNN spento), stessa catena 2 × 20 ns dallo stesso stato iniziale"],
    ["Throughput", f"{AA_NSDAY:.0f} ns/day (1 A100 + 32 core)",
     f"{it(CG_NSDAY, 1)} ns/day aggregati (2 replicas su 1 A100 + 8 core)"],
]
story += [table(rows, [3.0, 5.3, 8.7]), Spacer(1, 4),
          P("Le coordinate collettive (R<sub>g</sub>, rmsd del nucleo e dei loop, Q) sono calcolate per ogni copia "
            "dopo aver ricostruito la molecola attraverso il bordo periodico, rispetto alla struttura media della "
            "prima metà della traiettoria AA. Le sovrapposizioni fra distribuzioni sono "
            "∫ min(p<sub>AA</sub>, p<sub>CG</sub>) dx (1 = identiche). Il <b>tetto</b> è la sovrapposizione fra "
            "seconda e prima metà della traiettoria AA: è il massimo accordo che si può chiedere con questa "
            "statistica di riferimento.")]

# ── 2. Modello ──
story += [PageBreak(), P("2. Il modello CG-ML: mappatura, prior e residuo PaiNN", "h1"),
          P("L'energia potenziale del modello è la somma di due parti: U(R) = U<sub>prior</sub>(R) + "
            "U<sub>ML</sub>(R). U<sub>prior</sub> contiene termini analitici con un significato fisico "
            "(connettività, geometria locale, legami di Hoogsteen, impilamento, volume escluso), tarati sulla "
            "struttura all-atom mappata a 300 K. U<sub>ML</sub> è un residuo appreso con una rete PaiNN e "
            "corregge ciò che i prior non descrivono. Solvente e ioni sono impliciti: nel modello di produzione "
            "non c'è un termine elettrostatico."),
          fig("fig_model.png", 17.0,
              "Figura 1. Il modello CG sulla struttura nativa 2JPZ (struttura di partenza delle simulazioni AA, "
              "mappata sui siti CG; in grigio trasparente il DNA atomistico). <b>a</b>: siti colorati per tipo; "
              "legami armonici del backbone e scheletro rigido delle guanine in grigio scuro; contatti Morse "
              "tratteggiati per classe. <b>b</b>: la tetrade 1 (G4, G12, G16, G22) vista lungo l'asse del "
              "quadruplex, con i sei Morse B3–B3 (lati e diagonali) e i quattro Morse di Hoogsteen B2–B4. "
              "<b>c</b>: il grafo su cui agisce PaiNN per il sito B3 di G16; la sfera ha raggio r<sub>c</sub> = "
              "1,26 nm e contiene 76 degli altri 85 siti della copia. Non sono disegnati angoli, diedri e WCA. "
              "Figura generata con PyMOL.")]

story += [P("2.1 Mappatura", "h2"),
          P("Ogni copia di 26 nucleotidi è rappresentata da 86 siti di 8 tipi. Ogni sito è il centro di massa "
            "di un gruppo di atomi:"),
          B("<b>guanina</b>: sei siti uniti in un <b>corpo rigido</b>. S raccoglie zucchero e fosfato (P, O5′, "
            "C5′, C4′, O4′, C1′, C3′, C2′, O3′). B1–B5 dividono la base: B1 = N9, C4; B2 = N3, C2, N2; "
            "B3 = N1, C6, O6; B4 = C5, N7; B5 = C8. B3 porta l'O6 rivolto al canale, B2 e B4 i donatori e "
            "accettori del secondo legame di Hoogsteen (N2–H···N7), B5 la faccia di impilamento;"),
          B("<b>timina e adenina</b>: un solo sito, il centro di massa dell'intero nucleotide;"),
          B("<b>dinamica</b>: Langevin in ESPResSo. Le 12 guanine sono corpi rigidi con massa e tensore "
            "d'inerzia dal mapping (6 gradi di libertà); A e T sono punti materiali. In tutto ci sono 114 gradi "
            "di libertà per copia."),
          P("Nota: la definizione di S elenca anche gli ossigeni non pontanti con i nomi O1P/O2P, mentre il campo "
            "di forze li chiama OP1/OP2. Non entrano quindi nel centro di massa, allo stesso modo nel riferimento "
            "e in tutti i dataset.", "cap")]

story += [P("2.2 Prior fisici (insieme <i>lp2</i>)", "h2"),
          P("I parametri vengono dalla struttura AA mappata a 300 K (10 copie, 136 ns). Legami e angoli sono "
            "ottenuti per inversione di Boltzmann. Per i Morse r<sub>0</sub> è la mediana della distanza, e "
            "a = √(kT / 2Dσ<sup>2</sup>) usa la larghezza osservata σ, poi affinata iterativamente sulle "
            "distribuzioni del riferimento. I contatti sono specifici per coppia: legano due siti precisi della "
            "piega nativa. A runtime sono interazioni non legate su siti marcatori, con una coda smorzata fino a "
            "zero a r<sub>cut</sub>, quindi si possono rompere senza errori di \"bond broken\".")]
rows = [["termine", "forma", "siti", "per copia", "parametri"],
        ["legami del backbone", "½ k (r − r<sub>0</sub>)<super>2</super>", "sito S, A o T di residui consecutivi",
         "25", "k 136–3755 kJ mol<super>−1</super> nm<super>−2</super>; r<sub>0</sub> 0,58–0,70 nm"],
        ["angoli del backbone", "½ k (θ − θ<sub>0</sub>)<super>2</super>", "tre residui consecutivi", "24",
         "k 5–324 kJ mol<super>−1</super> rad<super>−2</super>; θ<sub>0</sub> 117–157°"],
        ["diedri del backbone", "coseno (n = 1), smorzato quando un angolo tende a 180°", "quattro residui consecutivi",
         "23", "K 2,4–74 kJ/mol"],
        ["torsione di impilamento", "coseno (n = 1)", "B3–B5–B5–B3 di guanine sovrapposte", "8",
         "K 391–774 kJ/mol (157–310 kT)"],
        ["Morse nella tetrade", "D [1 − e<super>−a(r−r<sub>0</sub>)</super>]<super>2</super>",
         "B3–B3, lati e diagonali (K<sub>4</sub>)", "18", "r<sub>0</sub> 0,38–0,65 nm; a 3,5–11,5 nm<super>−1</super>"],
        ["Morse di Hoogsteen", "idem", "B2–B4 (N2–H···N7)", "12", "r<sub>0</sub> 0,31–0,57 nm; a 6,1–11,0 nm<super>−1</super>"],
        ["Morse di impilamento", "idem", "B5–B5, stesso tratto, tetradi adiacenti", "8",
         "r<sub>0</sub> 0,37–0,45 nm; a 5,9–8,9 nm<super>−1</super>"],
        ["cappucci dei loop", "idem", "basi di loop/code contro la faccia delle tetradi esterne", "16",
         "r<sub>0</sub> 0,45–0,89 nm; a 2,3–5,1 nm<super>−1</super>"],
        ["volume escluso", "WCA, LJ 12-6 troncato e traslato a 2<super>1/6</super>σ", "tutte le coppie di tipi (esclusi 1-2 e 1-3)",
         "36 coppie di tipi", "ε 3,15 kJ/mol; σ 0,24–0,68 nm"]]
story += [table(rows, [3.1, 3.6, 4.1, 1.6, 4.6]), Spacer(1, 4),
          P("Tutti i Morse hanno D = 50 kJ/mol (circa 20 kT a 300 K). In totale sono 2700 kJ/mol per copia: è un "
            "modello del solo stato nativo, di tipo Gō, adatto a 300 K ma non all'unfolding (vedi sezione 9).", "cap")]

story += [P("2.3 Residuo PaiNN", "h2"),
          P("U<sub>ML</sub> è la somma di energie di sito, U<sub>ML</sub> = Σ<sub>i</sub> ε<sub>i</sub>, calcolate da "
            "una rete PaiNN (polarizable atom interaction neural network, Schütt et al. 2021). La rete è "
            "equivariante per rotazioni e invariante per traslazioni e per permutazioni di siti dello stesso tipo:"),
          B("<b>ingresso</b>: tipo del sito (8 specie → embedding di dimensione D = 64) e posizioni. Il grafo "
            "comprende tutte le coppie di siti entro r<sub>c</sub> = 1,2616 nm, anche fra copie diverse "
            "(figura 1c);"),
          B("<b>base radiale</b>: 32 gaussiane con centri fra 0 e r<sub>c</sub> e larghezza r<sub>c</sub>/32, "
            "moltiplicate per la funzione di taglio di Toxvaerd x<super>4</super>/(x<super>4</super> + "
            "α<super>4</super>), con x = (r<sub>c</sub> − r)/r<sub>c</sub> e α = 0,1. L'energia è liscia a "
            "r<sub>c</sub> fino alla derivata terza;"),
          B("<b>interazione</b>: 2 blocchi messaggio + aggiornamento sulle feature scalari s<sub>i</sub> ∈ "
            "ℝ<super>64</super> e vettoriali v<sub>i</sub> ∈ ℝ<super>3×64</super> (forma canonica di PaiNN, "
            "attivazioni SiLU);"),
          B("<b>uscita</b>: MLP 64 → 32 → 1 sullo scalare finale di ogni sito. A ogni ε<sub>i</sub> si sottrae il "
            "valore della stessa specie isolata, così U<sub>ML</sub> → 0 per siti lontani da tutto. "
            "In tutto la rete ha 106 049 parametri;"),
          B("<b>forze</b>: F = −∇U<sub>ML</sub> per differenziazione automatica sulle posizioni dei siti. Sulle "
            "guanine si riducono a forza e momento del corpo rigido. La rete gira nel plugin ESPResSo in "
            "libtorch, FP32 su GPU, e le forze sono conservative per costruzione."),
          P("<b>Allenamento: entropia relativa, non force matching.</b> Le forze AA istantanee proiettate sui siti "
            "CG sono dominate dal rumore: la parte che si può apprendere vale R<super>2</super> ≈ 0,5–1 %. Il "
            "force matching sopra prior già tarati peggiorava la struttura (sovrapposizione intra da 0,976 a "
            "0,94). Il residuo è stato quindi allenato minimizzando l'entropia relativa "
            "S<sub>rel</sub> = ⟨ln p<sub>AA</sub>/p<sub>CG</sub>⟩<sub>AA</sub>, che usa solo le posizioni. Il "
            "gradiente è β[⟨∂U/∂θ⟩<sub>AA</sub> − ⟨∂U/∂θ⟩<sub>CG</sub>]; i prior non vi compaiono perché non "
            "dipendono da θ."),
          B("ogni iterazione fa 25 ps di MD CG con il modello corrente; i campioni vengono riusati per "
            "ripesatura finché la taglia efficiente per copia resta ≥ 50 %;"),
          B("il passo è controllato da una regione di fiducia, e il 20 % finale della traiettoria AA è tenuto "
            "fuori come controllo;"),
          B("<b>re0</b>: 30 iterazioni sopra i prior lp1, partendo da U<sub>ML</sub> ≡ 0 (ultimo strato "
            "dell'uscita a zero). <b>re1</b>: 30 iterazioni sopra lp2 (lp1 più i cappucci dei loop), partendo "
            "da re0_it30. Il modello di produzione è re1_it30;"),
          B("dati AA: posizioni mappate di prod-1 (136 ns, 10 copie, un frame ogni 20 ps) a 300 K."),
          P(f"Effetto del residuo sulla sovrapposizione intra (tutti i siti), con lo stesso protocollo: soli prior lp2 "
            f"{it(OVP[CH[0]])} → lp2 + re1_it30 {it(OV[CH[0]]['CG-ML']['intra'])}, con un tetto AA contro AA di "
            f"{it(CEIL[0])}. Il guadagno è maggiore nel canale B3–B3 "
            f"({it(OVP[CH[1]], 2)} → {it(OV[CH[1]]['CG-ML']['intra'], 2)}); sezione 3.", "cap")]

# ── 2. P(r) ──
story += [PageBreak(), P("3. Distribuzioni di coppia intramolecolari P(r)", "h1"),
          P("Per ogni canale, P(r) è l'istogramma normalizzato delle distanze fra siti della stessa copia "
            "(stessi bin e stessa normalizzazione dello script 44). I canali separano i contributi principali "
            "della struttura: B3–B3 per gli appaiamenti di Hoogsteen nelle tetradi, B5–B5 per l'impilamento delle "
            "tetradi, S–S per il backbone; il canale \"tutti i siti\" li somma, loop compresi. Per isolare il "
            "contributo del residuo PaiNN si confronta anche una corsa con i <b>soli prior lp2</b>: stessa catena "
            "(2 × 20 ns), stesso stato iniziale, stesso termostato, PaiNN spento."),
          fig("fig_Pr_all.png", 15.5,
              "Figura 2. P(r) di tutti i siti (in alto) e scarto dall'AA, P(r) − P<sub>AA</sub>(r) (in basso). Blu: AA; "
              "verde: soli prior lp2; arancione: lp2 + PaiNN. La banda grigia è la differenza fra le due metà della "
              "traiettoria AA, cioè il rumore del riferimento. Con i soli prior lo scarto arriva a ±0,1 nm<super>−1</super> "
              "attorno a 0,4, 0,6 e 0,75 nm (geometria della tetrade e impilamento) e resta sistematico fino a 2 nm; "
              "con PaiNN rientra quasi ovunque nella banda."),
          fig("fig_Pr_intra.png", 16.5,
              "Figura 3. P(r) intramolecolari per canale. Blu: AA su 136 ns; arancione: CG-ML (lp2 + PaiNN) su "
              "2 × 20 ns; verde: soli prior lp2 su 2 × 20 ns; grigio tratteggiato e punteggiato: prima e seconda metà "
              "dell'AA. Nei titoli le sovrapposizioni con l'AA e il tetto (metà 2 contro metà 1).")]
rows = [["canale", "soli prior lp2", "CG-ML (lp2 + PaiNN)", "tetto (metà 2 – metà 1 AA)", "CG-ML − tetto",
         "scarto chiuso da PaiNN"]]
for c in CH:
    a, h, p0 = OV[c]["CG-ML"]["intra"], OV[c]["AA meta' 2"]["intra"], OV[c]["lp2"]["intra"]
    rows.append([c, it(p0), it(a), it(h), f"{a - h:+.3f}".replace(".", ","), f"{100 * (a - p0) / (h - p0):.0f} %"])
story += [table(rows, [3.3, 2.4, 2.8, 3.3, 2.2, 3.0]), Spacer(1, 6),
          P("PaiNN chiude fra il 91 e il 98 % della distanza fra i soli prior e il tetto. Il guadagno maggiore è "
            "nel canale B3–B3: con i soli prior il secondo picco (diagonale della tetrade, 0,65 nm) è basso e "
            "largo (2,2 contro 3,3 nm<super>−1</super>) e c'è una coda fino a 1,2 nm, cioè tetradi deformate; con "
            "PaiNN entrambi i picchi tornano all'altezza AA. Nel canale B5–B5 i prior impastano i picchi a "
            "0,75, 1,0 e 1,27 nm (orientazione relativa delle tetradi impilate), che PaiNN ricostruisce."),
          P("In tutti i canali il CG-ML raggiunge il tetto: lo scarto è al più 0,004, cioè dentro la variazione "
            "fra le due metà del riferimento. Posizioni e altezze dei picchi coincidono: per B3–B3 0,34, 0,445, "
            "0,65 e 0,84 nm; per S–S il primo vicino a 0,595 nm. Le differenze residue, per esempio i picchi S–S a "
            "1,17 e 1,42 nm leggermente più alti nel CG, hanno la stessa ampiezza delle fluttuazioni fra le metà "
            "AA. La geometria del nucleo (tetradi, impilamento, backbone) è quindi riprodotta a livello di "
            "distribuzione, non solo in media.")]

# ── 3. g(r) inter ──
story += [PageBreak(), P("4. g(r) fra copie diverse (corto raggio)", "h1"),
          P(f"La g(r) intermolecolare conta le coppie di siti su copie diverse. Fino a {it(2.0, 1)} nm, la distanza "
            f"massima calcolata, descrive solo gli incontri ravvicinati fra copie: con R<sub>g</sub> ≈ 0,8 nm le "
            f"copie si toccano di rado e g(r) resta molto sotto 1, nella zona di volume escluso."),
          fig("fig_gr_inter.png", 16.5,
              "Figura 4. g(r) fra siti di copie diverse, stessi colori della figura 3. Le due metà AA differiscono "
              "di un fattore ~5: gli incontri fra copie nell'AA sono pochi eventi lunghi.")]
rows = [["canale", "soli prior lp2", "CG-ML contro AA", "metà 2 – metà 1 AA"]]
for c in CH:
    rows.append([c, it(OV[c]["lp2"]["inter"]), it(OV[c]["CG-ML"]["inter"]), it(OV[c]["AA meta' 2"]["inter"])])
story += [table(rows, [4.4, 3.0, 3.6, 4.4]), Spacer(1, 6),
          P("Nell'AA la prima metà quasi non ha contatti fra copie, la seconda molti di più: questi contatti sono "
            "eventi rari e non ergodici su 136 ns, e il riferimento non li campiona. La sovrapposizione CG–AA (0,68–"
            "0,93) supera quella fra le due metà AA (0–0,85), quindi qui la metrica non discrimina. Il CG-ML ha "
            "contatti fra copie un po' più frequenti della media AA, simili alla seconda metà. "
            "<b>Conclusione</b>: con questo riferimento la struttura fra copie a corto raggio non è verificabile. "
            "Per la g(r) completa fra copie (distanze fino a metà box) basta rilanciare lo script 53 con "
            "<font name='DV-I'>--rmax</font> più grande, ma l'incertezza AA resterà quella dettata da 10 copie in "
            "una sola box.")]

# ── 4. Rg ──
story += [PageBreak(), P("5. Raggio di girazione", "h1"),
          fig("fig_Rg.png", 16.5,
              "Figura 5. Sinistra: distribuzione di R<sub>g</sub> (AA: singole copie in linea sottile, insieme in "
              "linea spessa; CG-ML: tutte le copie e replicas; verde: soli prior lp2). Destra: serie temporali con media mobile di 1 ns, "
              "AA su 136 ns e CG-ML (replica r0) su 20 ns, stessa scala verticale.")]
rows = [["", "AA", "CG-ML", "soli prior lp2"],
        ["media ± deviazione standard (nm)", f"{it(cv['Rg']['aa_mean'])} ± {it(cv['Rg']['aa_std'])}",
         f"{it(cv['Rg']['cg_mean'])} ± {it(cv['Rg']['cg_std'])}", f"{it(cv['Rg']['pri_mean'])} ± {it(cv['Rg']['pri_std'])}"],
        ["medie delle singole copie (nm)", f"{it(cv['Rg']['aa_copy_min'])}–{it(cv['Rg']['aa_copy_max'])}",
         "tutte ≈ 0,800", ""],
        ["τ a 1/e (script 47)", f"{it(tauAA['tau_Rg'] / 1000, 1)} ns", f"{it(tauCG['tau_Rg'], 1)} ps", ""],
        ["sovrapposizione con l'AA (tetto)", "–", f"{it(cv['Rg']['ov'], 2)} ({it(cv['Rg']['ceil'], 2)})",
         it(cv['Rg']['pri_ov'], 2)]]
story += [table(rows, [5.4, 3.8, 3.8, 3.8]), Spacer(1, 6),
          P("Il valore medio è corretto entro 0,002 nm. La distribuzione CG è un po' più larga di quella AA "
            "complessiva e molto più larga di quella di una singola copia AA. Il motivo è nelle serie temporali: "
            "nell'AA ogni copia resta per decine di ns su un proprio R<sub>g</sub> (la copia più compatta a "
            "0,78 nm per gran parte della traiettoria), con un rilassamento lento di 5–10 ns legato ai "
            "riarrangiamenti dei loop. Nel CG la media su 1 ns è piatta e uguale per tutte le copie, perché "
            "R<sub>g</sub> si decorrela in pochi ps. Il CG campiona quindi un unico bacino \"medio\" al posto "
            "degli stati che l'AA visita lentamente. Per R<sub>g</sub> il rapporto fra i τ "
            f"(~{round(alpha['tau_Rg'], -2):.0f}) misura processi diversi e non è un'accelerazione.")]

# ── 5. CV ──
story += [KeepTogether([P("6. Altre coordinate collettive", "h1"),
          fig("fig_cv.png", 16.5,
              "Figura 6. Distribuzioni di rmsd del nucleo, rmsd dei loop e frazione di contatti nativi Q (grigio: "
              "metà AA; verde: soli prior lp2). Nei titoli la sovrapposizione CG–AA e il tetto.")])]
rows = [["coordinata", "AA media ± std", "CG-ML media ± std", "sovr. CG-ML", "soli prior media ± std", "sovr. soli prior", "tetto"]]
names = {"Rg": "R<sub>g</sub> (nm)", "rmsd_core": "rmsd nucleo (nm)", "rmsd_loops": "rmsd loop (nm)", "Q": "Q"}
for v in ["Rg", "rmsd_core", "rmsd_loops", "Q"]:
    s = cv[v]
    rows.append([names[v], f"{it(s['aa_mean'])} ± {it(s['aa_std'])}", f"{it(s['cg_mean'])} ± {it(s['cg_std'])}",
                 it(s["ov"], 2), f"{it(s['pri_mean'])} ± {it(s['pri_std'])}", it(s["pri_ov"], 2), it(s["ceil"], 2)])
story += [table(rows, [2.6, 2.6, 2.6, 1.8, 2.8, 2.0, 1.4]), Spacer(1, 6),
          P("Con i soli prior il nucleo ha un rmsd doppio (0,088 contro 0,044 nm) e la sua distribuzione quasi non si "
            "sovrappone a quella AA (0,04): le tetradi restano formate (Q persino un po' più alto) ma deformate e "
            "più mobili. PaiNN porta il nucleo al tetto e riduce anche l'rmsd dei loop (0,31 → 0,27 nm)."),
          P("Il nucleo è al tetto: rmsd 0,044 nm in entrambi, sovrapposizione 0,93 contro 0,94. Q ha la stessa "
            "media ma una distribuzione un po' più stretta nel CG (std 0,009 contro 0,013). I loop sono l'unico "
            "scarto netto: nel CG sono più mobili e lontani dalla struttura media (0,27 contro 0,22 nm), e manca la "
            "popolazione a rmsd basso (0,13–0,2 nm) che nell'AA corrisponde agli stati compatti dei loop. Le "
            "diagnosi degli script 49–52 attribuiscono lo scarto a stati metastabili assenti nel CG (stato 1 del "
            "loop 3, stato minore del loop 2) e a un registro di appaiamenti in parte sbagliato (T2–A15 in "
            "eccesso)."),
          fig("fig_fes.png", 15.0,
              "Figura 7. Energia libera F = −kT ln p(rmsd nucleo, Q): AA, soli prior lp2, CG-ML. Il minimo dei soli prior "
              "è spostato a rmsd ≈ 0,09 nm. La JSD fra le FES AA e CG-ML (script 46) è "
              "0,0787, uguale al tetto AA contro sé stesso (0,0786); il CG è più rumoroso per il numero minore di "
              "frame (20 080 contro 67 880 copie-frame).")]

# ── 6. MSD ──
story += [PageBreak(), P("7. Dinamica: MSD traslazionale e rotazionale", "h1"),
          P("MSD del centro di massa di ogni copia e MSD angolare dell'orientazione del nucleo (allineamento di "
            "Kabsch, rotazioni fra frame successivi sommate come vettori di rotazione nel sistema del laboratorio; "
            "script 48). Le curve si fermano a un quarto della durata di ogni traiettoria. Nella riga in basso "
            "l'asse dei tempi è convertito in ore di GPU con il throughput misurato."),
          fig("fig_msd.png", 16.5,
              "Figura 8. In alto: MSD in funzione del tempo simulato; tratteggiato 6Dt con D dal tratto lineare. "
              "In basso: le stesse curve in funzione delle ore di GPU A100 (AA 137 ns/day; CG-ML 54,2 ns/day "
              "aggregati su 2 replicas).")]
rows = [["", "AA", "CG-ML", "rapporto CG/AA"],
        ["D (nm<super>2</super>/ns), retta", it(ms["AA"]["D_trans"]), it(ms["prod"]["D_trans"], 1),
         f"{ms['prod']['D_trans'] / ms['AA']['D_trans']:.0f}"],
        ["τ<sub>v</sub> dal fit di Langevin (ps)", it(ms["AA"]["tau_v_ps"], 1), f"{ms['prod']['tau_v_ps']:.0f}", ""],
        ["MSD traslazionale per ora di GPU (nm<super>2</super>)", it(rate["D_trans"], 2),
         f"{rate_cg['D_trans']:.0f}", f"<b>{rate_cg['D_trans'] / rate['D_trans']:.0f}×</b>"],
        ["D<sub>rot</sub> (rad<super>2</super>/ns), retta", it(ms["AA"]["D_rot"]), it(ms["prod"]["D_rot"], 1),
         f"{ms['prod']['D_rot'] / ms['AA']['D_rot']:.0f}"],
        ["τ<sub>P2</sub> (ps)", f"{ms['AA']['tau_P2']:.0f}", it(ms["prod"]["tau_P2"], 1),
         f"α = {ms['AA']['tau_P2'] / ms['prod']['tau_P2']:.0f}"],
        ["1/(6D<sub>rot</sub>) (ps)", f"{1000 / (6 * ms['AA']['D_rot']):.0f}", it(1000 / (6 * ms['prod']['D_rot']), 1), ""],
        ["MSD angolare per ora di GPU (rad<super>2</super>)", it(rate["D_rot"], 2), f"{rate_cg['D_rot']:.0f}",
         f"{rate_cg['D_rot'] / rate['D_rot']:.0f}× (sovrastima)"],
        ["decorrelazione dell'orientazione per ora di GPU", "", "",
         f"<b>{ms['AA']['tau_P2'] / ms['prod']['tau_P2'] * R:.0f}×</b>"]]
story += [table(rows, [7.0, 2.6, 2.6, 3.8]), Spacer(1, 6),
          B("<b>Traslazione</b>: nel CG il regime è balistico fino a ~0,1–0,3 ns (τ<sub>v</sub> ≈ 100 ps), poi "
            "diffusivo; D è 0,68 del valore atteso kT/(Nγ) per una molecola isolata. Nel regime diffusivo "
            "l'MSD per ora di GPU vale 6D × (ns per ora), cioè α × rapporto di throughput: "
            f"{rate_cg['D_trans'] / rate['D_trans']:.0f}×. A pochi minuti di calcolo il CG è ancora balistico e "
            "il guadagno è minore."),
          B("<b>Rotazione</b>: nell'AA τ<sub>P2</sub> ≈ 1/(6D<sub>rot</sub>), quindi la rotazione è diffusiva e "
            "il metodo è confermato. Nel CG τ<sub>P2</sub> è ~7 volte 1/(6D<sub>rot</sub>): l'orientazione si "
            "decorrela già nel regime inerziale (γ basso), e l'MSD angolare sovrastima il guadagno sul "
            "campionamento delle orientazioni. Per la rotazione il numero da usare è quello della decorrelazione, "
            f"{ms['AA']['tau_P2'] / ms['prod']['tau_P2'] * R:.0f}×.")]

# ── 7. efficienza ──
story += [P("8. Efficienza per GPU", "h1"),
          P("Efficienza = α × (ns/day<sub>CG</sub> / ns/day<sub>AA</sub>), con α = τ<sub>AA</sub>/τ<sub>CG</sub> "
            f"(tempi a 1/e dello script 47) oppure il rapporto dei D; ns/day<sub>CG</sub>/ns/day<sub>AA</sub> = "
            f"{it(CG_NSDAY, 1)}/{AA_NSDAY:.0f} = {it(R, 3)}.")]
rows = [["osservabile", "τ AA", "τ CG-ML", "α", "efficienza per GPU"],
        ["traslazione (MSD)", "", "", f"{ms['prod']['D_trans'] / ms['AA']['D_trans']:.0f} (rapporto dei D)",
         f"{rate_cg['D_trans'] / rate['D_trans']:.0f}×"],
        ["siti del nucleo", f"{tauAA['tau_core_sites']:.0f} ps", "0,48 ps*", "363*", "~140×"],
        ["rotazione (τ<sub>P2</sub>)", f"{it(tauAA['tau_rot'] / 1000, 2)} ns", f"{it(tauCG['tau_rot'], 1)} ps",
         f"{alpha['tau_rot']:.0f}", f"{alpha['tau_rot'] * R:.0f}×"],
        ["<b>rmsd del nucleo</b>", f"{it(tauAA['tau_core'], 1)} ps", f"{it(tauCG['tau_core'], 2)} ps",
         f"{alpha['tau_core']:.0f}", f"<b>{alpha['tau_core'] * R:.0f}×</b> (minimo)"],
        ["R<sub>g</sub>, loop", f"{it(tauAA['tau_Rg'] / 1000, 1)}, {it(tauAA['tau_loops'] / 1000, 1)} ns",
         f"{it(tauCG['tau_Rg'], 1)}, {it(tauCG['tau_loops'], 1)} ps", "~2400", "non interpretabile"]]
story += [table(rows, [4.2, 2.6, 2.6, 3.4, 4.2]), Spacer(1, 4),
          P("* Con un frame ogni ps il τ CG dei siti del nucleo (~0,5 ps) non è risolto (lo script 47 dà "
            f"{it(tauCG['tau_core_sites'], 2)} ps per interpolazione); si usano τ e α dalla corsa campionata "
            "ogni 0,5 ps.", "cap"),
          P("Il numero da citare è il minimo fra gli osservabili confrontabili: <b>~28× per GPU</b> "
            "(rmsd del nucleo), con traslazione e moti locali del nucleo a ~140× e rotazione a ~44×. Per budget di "
            "nodo il vantaggio può crescere fino a ~4×, perché l'AA occupa 32 core contro 8 del CG; serve il "
            "throughput AA con 8 core per GPU per fissarlo.")]

# ── 8. limiti ──
story += [P("9. Limiti e avvertenze", "h1"),
          B("<b>Loop</b>: un solo bacino al posto degli stati metastabili AA. Mancano lo stato 1 del loop 3 (26 %) "
            "e lo stato minore del loop 2; la coda 5' è troppo mobile; il contatto T2–A15 è in eccesso "
            "(0,56 contro 0,24 ± 0,09). I contatti dipendenti dallo stato (lp2c0) sono stati provati e non adottati: "
            "con un solo sito per T e A la specificità del registro non si ottiene con contatti fra coppie."),
          B("<b>Riferimento AA non ergodico sui loop</b>: il 17–61 % della varianza dei loop è fra copie e i tempi "
            "lenti sono 4–10 ns su 136 ns. Le popolazioni AA degli stati dei loop hanno errori grandi, e i tetti "
            "su loop e R<sub>g</sub> sono ottimistici: le due metà condividono le stesse copie bloccate."),
          B("<b>Solo stato nativo</b>: i 54 Morse a 50 kJ/mol per copia tengono il quadruplex chiuso a qualunque "
            "temperatura accessibile. Con i soli prior, su una copia, nessuna tetrade si apre fra 300 e 600 K in "
            "250 ns, e l'energia potenziale cresce come quella di un sistema armonico. Il modello serve per la "
            "struttura e la dinamica attorno al nativo; per la termodinamica di unfolding le profondità dei contatti "
            "vanno ricalibrate (lavoro in corso, cartella <font name='DV-I'>tel26_unfold</font>)."),
          B("<b>g(r) fra copie</b>: verificata solo a corto raggio (≤ 2 nm), dove anche l'AA non converge."),
          B("<b>Efficienza</b>: per GPU A100, a parità di hardware; il rapporto per GPU-ora dipende dal throughput "
            "del nodo, che varia di ~20 % fra nodi a parità di codice."),
          Spacer(1, 6)]

# ── appendice ──
story += [P("Appendice: provenienza dei dati", "h2"),
          P("Dati: <font name='DV-I'>report_data.npz</font> generato su Leonardo da "
            "<font name='DV-I'>53_tel26_report_data.py</font> (P(r) e g(r) con le routine dello script 44; "
            "coordinate collettive con lo script 46; curve MSD da <font name='DV-I'>msd_wall_prod.json</font>, "
            "script 48; tempi da <font name='DV-I'>ts_prod.json</font>, script 47). Riferimento AA: "
            "<font name='DV-I'>tel26_lp1_dataset.bin</font>. CG-ML: <font name='DV-I'>prod_re1_s0{1..4}_r{0,1}"
            ".samples.npz</font> (catena prod_re1, modello lp2 + re1_it30); soli prior: <font name='DV-I'>pri_lp2_s0{1..4}_r{0,1}"
            ".samples.npz</font> (stessa catena con DISABLE_ML=1), dati in <font name='DV-I'>report_data_v2.npz</font>. Figure e PDF: "
            "<font name='DV-I'>make_figs.py</font>, <font name='DV-I'>build_report.py</font> nella cartella "
            "<font name='DV-I'>tutorials/tel26/report</font>. Parametri: P(r) con "
            f"un frame AA ogni {meta['aa_stride']}, CG ogni {meta['cg_stride']} ps, "
            "200 bin fino a 2 nm.", "cap")]

doc = SimpleDocTemplate("tel26_report_AA_vs_CGML.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="TEL26: AA contro CG-ML",
                        author="PaiNN-LT", subject="Confronto strutturale e dinamico AA contro CG-ML")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("ok")
