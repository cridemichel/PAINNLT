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

d = np.load("report_data.npz")
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
            "rotazionale, anche per ora di GPU). Progetto PaiNN-LT · MLCG_Framework_v2 · 5 ottobre 2026", "sub")]

summary = [
    P("<b>Sintesi</b>", "body"),
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
     "prior fisici <i>lp2</i> (legami, angoli, tetradi, 16 contatti di loop per copia) + residuo PaiNN "
     "(<i>re1_it30</i>: force matching, poi relative entropy)"],
    ["Dinamica", "dt 2 fs", "Langevin, dt 4 fs, γ = 2; ESPResSo + plugin PaiNN"],
    ["Traiettorie", f"1 × {it(meta['aa_frames'] * meta['aa_dt_ps'] / 1000, 1)} ns, 10 copie",
     "2 replicas × 20 ns (4 segmenti × 5 ns), 10 copie per replica"],
    ["Frame analizzati", f"{meta['aa_frames']} frame ogni {meta['aa_dt_ps']:.0f} ps (P(r): uno ogni "
     f"{meta['aa_stride']})", f"struttura: ogni {meta['cg_stride']} ps ({meta['runs'][0]['frames_used']} frame); "
     "MSD e tempi: ogni 1 ps"],
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

# ── 2. P(r) ──
story += [PageBreak(), P("2. Distribuzioni di coppia intramolecolari P(r)", "h1"),
          P("Per ogni canale, P(r) è l'istogramma normalizzato delle distanze fra siti della stessa copia "
            "(stessi bin e stessa normalizzazione dello script 44). I canali separano i contributi principali "
            "della struttura: B3–B3 per gli appaiamenti di Hoogsteen nelle tetradi, B5–B5 per l'impilamento delle "
            "tetradi, S–S per il backbone; il canale \"tutti i siti\" li somma, loop compresi."),
          fig("fig_Pr_intra.png", 16.5,
              "Figura 1. P(r) intramolecolari per canale. Blu: AA su 136 ns; arancione: CG-ML su 2 × 20 ns; grigio "
              "tratteggiato e punteggiato: prima e seconda metà dell'AA. In alto a destra la sovrapposizione CG–AA e "
              "il tetto (metà 2 contro metà 1).")]
rows = [["canale", "CG-ML contro AA", "tetto (metà 2 – metà 1 AA)", "differenza"]]
for c in CH:
    a, h = OV[c]["CG-ML"]["intra"], OV[c]["AA meta' 2"]["intra"]
    rows.append([c, it(a), it(h), f"{a - h:+.3f}".replace(".", ",")])
story += [table(rows, [4.4, 3.6, 5.0, 3.0]), Spacer(1, 6),
          P("In tutti i canali il CG-ML raggiunge il tetto: lo scarto è al più 0,004, cioè dentro la variazione "
            "fra le due metà del riferimento. Posizioni e altezze dei picchi coincidono: per B3–B3 0,34, 0,445, "
            "0,65 e 0,84 nm; per S–S il primo vicino a 0,595 nm. Le differenze residue, per esempio i picchi S–S a "
            "1,17 e 1,42 nm leggermente più alti nel CG, hanno la stessa ampiezza delle fluttuazioni fra le metà "
            "AA. La geometria del nucleo (tetradi, impilamento, backbone) è quindi riprodotta a livello di "
            "distribuzione, non solo in media.")]

# ── 3. g(r) inter ──
story += [PageBreak(), P("3. g(r) fra copie diverse (corto raggio)", "h1"),
          P(f"La g(r) intermolecolare conta le coppie di siti su copie diverse. Fino a {it(2.0, 1)} nm, la distanza "
            f"massima calcolata, descrive solo gli incontri ravvicinati fra copie: con R<sub>g</sub> ≈ 0,8 nm le "
            f"copie si toccano di rado e g(r) resta molto sotto 1, nella zona di volume escluso."),
          fig("fig_gr_inter.png", 16.5,
              "Figura 2. g(r) fra siti di copie diverse, stessi colori della figura 1. Le due metà AA differiscono "
              "di un fattore ~5: gli incontri fra copie nell'AA sono pochi eventi lunghi.")]
rows = [["canale", "CG-ML contro AA", "metà 2 – metà 1 AA"]]
for c in CH:
    rows.append([c, it(OV[c]["CG-ML"]["inter"]), it(OV[c]["AA meta' 2"]["inter"])])
story += [table(rows, [4.4, 3.6, 4.4]), Spacer(1, 6),
          P("Nell'AA la prima metà quasi non ha contatti fra copie, la seconda molti di più: questi contatti sono "
            "eventi rari e non ergodici su 136 ns, e il riferimento non li campiona. La sovrapposizione CG–AA (0,68–"
            "0,93) supera quella fra le due metà AA (0–0,85), quindi qui la metrica non discrimina. Il CG-ML ha "
            "contatti fra copie un po' più frequenti della media AA, simili alla seconda metà. "
            "<b>Conclusione</b>: con questo riferimento la struttura fra copie a corto raggio non è verificabile. "
            "Per la g(r) completa fra copie (distanze fino a metà box) basta rilanciare lo script 53 con "
            "<font name='DV-I'>--rmax</font> più grande, ma l'incertezza AA resterà quella dettata da 10 copie in "
            "una sola box.")]

# ── 4. Rg ──
story += [PageBreak(), P("4. Raggio di girazione", "h1"),
          fig("fig_Rg.png", 16.5,
              "Figura 3. Sinistra: distribuzione di R<sub>g</sub> (AA: singole copie in linea sottile, insieme in "
              "linea spessa; CG-ML: tutte le copie e replicas). Destra: serie temporali con media mobile di 1 ns, "
              "AA su 136 ns e CG-ML (replica r0) su 20 ns, stessa scala verticale.")]
rows = [["", "AA", "CG-ML"],
        ["media ± deviazione standard (nm)", f"{it(cv['Rg']['aa_mean'])} ± {it(cv['Rg']['aa_std'])}",
         f"{it(cv['Rg']['cg_mean'])} ± {it(cv['Rg']['cg_std'])}"],
        ["medie delle singole copie (nm)", f"{it(cv['Rg']['aa_copy_min'])}–{it(cv['Rg']['aa_copy_max'])}",
         "tutte ≈ 0,800"],
        ["τ a 1/e (script 47)", f"{it(tauAA['tau_Rg'] / 1000, 1)} ns", f"{it(tauCG['tau_Rg'], 1)} ps"],
        ["sovrapposizione con l'AA (tetto)", "–", f"{it(cv['Rg']['ov'], 2)} ({it(cv['Rg']['ceil'], 2)})"]]
story += [table(rows, [6.0, 5.0, 5.0]), Spacer(1, 6),
          P("Il valore medio è corretto entro 0,002 nm. La distribuzione CG è un po' più larga di quella AA "
            "complessiva e molto più larga di quella di una singola copia AA. Il motivo è nelle serie temporali: "
            "nell'AA ogni copia resta per decine di ns su un proprio R<sub>g</sub> (la copia più compatta a "
            "0,78 nm per gran parte della traiettoria), con un rilassamento lento di 5–10 ns legato ai "
            "riarrangiamenti dei loop. Nel CG la media su 1 ns è piatta e uguale per tutte le copie, perché "
            "R<sub>g</sub> si decorrela in pochi ps. Il CG campiona quindi un unico bacino \"medio\" al posto "
            "degli stati che l'AA visita lentamente. Per R<sub>g</sub> il rapporto fra i τ "
            f"(~{round(alpha['tau_Rg'], -2):.0f}) misura processi diversi e non è un'accelerazione.")]

# ── 5. CV ──
story += [KeepTogether([P("5. Altre coordinate collettive", "h1"),
          fig("fig_cv.png", 16.5,
              "Figura 4. Distribuzioni di rmsd del nucleo, rmsd dei loop e frazione di contatti nativi Q (grigio: "
              "metà AA). Nei titoli la sovrapposizione CG–AA e il tetto.")])]
rows = [["coordinata", "AA media ± std", "CG-ML media ± std", "sovrapposizione", "tetto"]]
names = {"Rg": "R<sub>g</sub> (nm)", "rmsd_core": "rmsd nucleo (nm)", "rmsd_loops": "rmsd loop (nm)", "Q": "Q"}
for v in ["Rg", "rmsd_core", "rmsd_loops", "Q"]:
    s = cv[v]
    rows.append([names[v], f"{it(s['aa_mean'])} ± {it(s['aa_std'])}", f"{it(s['cg_mean'])} ± {it(s['cg_std'])}",
                 it(s["ov"], 2), it(s["ceil"], 2)])
story += [table(rows, [3.4, 3.5, 3.5, 3.2, 1.9]), Spacer(1, 6),
          P("Il nucleo è al tetto: rmsd 0,044 nm in entrambi, sovrapposizione 0,93 contro 0,94. Q ha la stessa "
            "media ma una distribuzione un po' più stretta nel CG (std 0,009 contro 0,013). I loop sono l'unico "
            "scarto netto: nel CG sono più mobili e lontani dalla struttura media (0,27 contro 0,22 nm), e manca la "
            "popolazione a rmsd basso (0,13–0,2 nm) che nell'AA corrisponde agli stati compatti dei loop. Le "
            "diagnosi degli script 49–52 attribuiscono lo scarto a stati metastabili assenti nel CG (stato 1 del "
            "loop 3, stato minore del loop 2) e a un registro di appaiamenti in parte sbagliato (T2–A15 in "
            "eccesso)."),
          fig("fig_fes.png", 15.0,
              "Figura 5. Energia libera F = −kT ln p(rmsd nucleo, Q), AA e CG-ML. La JSD fra le FES (script 46) è "
              "0,0787, uguale al tetto AA contro sé stesso (0,0786); il CG è più rumoroso per il numero minore di "
              "frame (20 080 contro 67 880 copie-frame).")]

# ── 6. MSD ──
story += [PageBreak(), P("6. Dinamica: MSD traslazionale e rotazionale", "h1"),
          P("MSD del centro di massa di ogni copia e MSD angolare dell'orientazione del nucleo (allineamento di "
            "Kabsch, rotazioni fra frame successivi sommate come vettori di rotazione nel sistema del laboratorio; "
            "script 48). Le curve si fermano a un quarto della durata di ogni traiettoria. Nella riga in basso "
            "l'asse dei tempi è convertito in ore di GPU con il throughput misurato."),
          fig("fig_msd.png", 16.5,
              "Figura 6. In alto: MSD in funzione del tempo simulato; tratteggiato 6Dt con D dal tratto lineare. "
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
story += [P("7. Efficienza per GPU", "h1"),
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
story += [P("8. Limiti e avvertenze", "h1"),
          B("<b>Loop</b>: un solo bacino al posto degli stati metastabili AA. Mancano lo stato 1 del loop 3 (26 %) "
            "e lo stato minore del loop 2; la coda 5' è troppo mobile; il contatto T2–A15 è in eccesso "
            "(0,56 contro 0,24 ± 0,09). I contatti dipendenti dallo stato (lp2c0) sono stati provati e non adottati: "
            "con un solo sito per T e A la specificità del registro non si ottiene con contatti fra coppie."),
          B("<b>Riferimento AA non ergodico sui loop</b>: il 17–61 % della varianza dei loop è fra copie e i tempi "
            "lenti sono 4–10 ns su 136 ns. Le popolazioni AA degli stati dei loop hanno errori grandi, e i tetti "
            "su loop e R<sub>g</sub> sono ottimistici: le due metà condividono le stesse copie bloccate."),
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
            ".samples.npz</font> (catena prod_re1, modello lp2 + re1_it30). Figure e PDF: "
            "<font name='DV-I'>make_figs.py</font>, <font name='DV-I'>build_report.py</font> nella cartella "
            "<font name='DV-I'>tutorials/tel26/report</font>. Parametri: P(r) con "
            f"un frame AA ogni {meta['aa_stride']}, CG ogni {meta['cg_stride']} ps, "
            "200 bin fino a 2 nm.", "cap")]

doc = SimpleDocTemplate("tel26_report_AA_vs_CGML.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="TEL26: AA contro CG-ML",
                        author="PaiNN-LT", subject="Confronto strutturale e dinamico AA contro CG-ML")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("ok")
