#!/usr/bin/env python3
"""English PDF report, AA vs CG-ML for TEL26 (data: report_data.npz, figures: make_figs_en.py)."""
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
st = json.load(open("fig_stats_en.json"))
OV = meta["overlaps"]
CH = meta["channels"]
CH_EN = {"tutti i siti": "all sites", "B3-B3 (legami H)": "B3-B3 (H bonds)", "B5-B5 (stacking)": "B5-B5 (stacking)", "S-S (backbone)": "S-S (backbone)"}
cv = st["cv"]
ms = st["msd"]
ts = {e["label"]: e for e in meta["timescales"]}
AA_NSDAY, CG_NSDAY = 137.0, 54.2
R = CG_NSDAY / AA_NSDAY


def it(x, n=3):
    """number with a decimal point"""
    return f"{x:.{n}f}"


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
    c.drawString(2 * cm, 1.2 * cm, "TEL26 · AA vs CG-ML comparison · PaiNN-LT / MLCG_Framework_v2")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    c.restoreState()


# ── derived numbers ──
rate = {k: 6 * ms["AA"][k] * AA_NSDAY / 24 for k in ("D_trans", "D_rot")}
rate_cg = {k: 6 * ms["prod"][k] * CG_NSDAY / 24 for k in ("D_trans", "D_rot")}
tauAA = {k: ts["AA"][k][0] for k in ("tau_rot", "tau_core", "tau_core_sites", "tau_Rg", "tau_loops")}
tauCG = {k: 0.5 * (ts["r0"][k][0] + ts["r1"][k][0]) for k in tauAA}
alpha = {k: tauAA[k] / tauCG[k] for k in tauAA}
ov_min_cg = min(OV[c]["CG-ML"]["intra"] for c in CH)
ov_max_cg = max(OV[c]["CG-ML"]["intra"] for c in CH)
CEIL = [OV[c]["AA meta' 2"]["intra"] for c in CH]

story = []
story += [P("TEL26: all-atom vs CG-ML dynamics", "title"),
          P("Hybrid-2 (3+1) G-quadruplex (2JPZ, (TTAGGG)<sub>4</sub>TT), 10 copies × 26 nt. "
            "Structure (P(r), g(r), R<sub>g</sub> and collective variables), dynamics (translational and "
            "rotational MSD, also per GPU hour) and a description of the model. Project PaiNN-LT · "
            "MLCG_Framework_v2 · 7 October 2026", "sub")]

summary = [
    P("<b>Summary</b>", "body"),
    B(f"<b>Core and local structure reproduced within the noise of the reference.</b> The intramolecular "
      f"P(r) per channel overlap with the AA ones by {it(ov_min_cg)}–{it(ov_max_cg)}, against a ceiling "
      f"(half vs half of the AA trajectory) of {it(min(CEIL))}–{it(max(CEIL))}; "
      f"core rmsd: overlap {it(cv['rmsd_core']['ov'], 2)} (ceiling {it(cv['rmsd_core']['ceil'], 2)}).", "box"),
    B(f"<b>Correct mean R<sub>g</sub></b>: {it(cv['Rg']['cg_mean'])} nm vs {it(cv['Rg']['aa_mean'])} nm "
      f"(AA); overlap {it(cv['Rg']['ov'], 2)} (ceiling {it(cv['Rg']['ceil'], 2)}). The AA copies stay on "
      f"different R<sub>g</sub> values for tens of ns; in the CG all copies fluctuate around the same value.", "box"),
    B(f"<b>Limitation: the loops.</b> Loop rmsd: overlap {it(cv['rmsd_loops']['ov'], 2)} (ceiling "
      f"{it(cv['rmsd_loops']['ceil'], 2)}), mean {it(cv['rmsd_loops']['cg_mean'])} vs "
      f"{it(cv['rmsd_loops']['aa_mean'])} nm. The CG has a single basin, whereas the AA has metastable states "
      f"lasting ns.", "box"),
    B(f"<b>Efficiency per GPU hour</b> (A100; 2 replicas per GPU, {it(CG_NSDAY, 1)} ns/day aggregate, vs "
      f"{AA_NSDAY:.0f} ns/day for the AA): translation {rate_cg['D_trans'] / rate['D_trans']:.0f}× (MSD); "
      f"rotation {alpha['tau_rot'] * R:.0f}× (decorrelation); core rmsd "
      f"{alpha['tau_core'] * R:.0f}×, the minimum among comparable observables.", "box"),
]
story += [boxed(summary), Spacer(1, 10)]

# ── 1. Data ──
story += [P("1. Systems, models and data", "h1")]
rows = [
    ["", "All-atom (reference)", "CG-ML"],
    ["Representation", "169 000 atoms, explicit solvent",
     "10 × 86 = 860 beads (G: backbone S + bases B1–B5; T, A: one site), implicit solvent"],
    ["Hamiltonian", "atomistic force field",
     "physical priors <i>lp2</i> (bonds, angles, dihedrals, 54 Morse contacts per copy, WCA) + PaiNN residual "
     "(<i>re1_it30</i>, trained by relative entropy); section 2"],
    ["Dynamics", "dt 2 fs", "Langevin, dt 4 fs, γ = 2; ESPResSo + PaiNN plugin"],
    ["Trajectories", f"1 × {it(meta['aa_frames'] * meta['aa_dt_ps'] / 1000, 1)} ns, 10 copies",
     "2 replicas × 20 ns (4 segments × 5 ns), 10 copies per replica"],
    ["Analysed frames", f"{meta['aa_frames']} frames every {meta['aa_dt_ps']:.0f} ps (P(r): one every "
     f"{meta['aa_stride']})", f"structure: every {meta['cg_stride']} ps ({meta['runs'][0]['frames_used']} frames); "
     "MSD and times: every 1 ps"],
    ["Throughput", f"{AA_NSDAY:.0f} ns/day (1 A100 + 32 cores)",
     f"{it(CG_NSDAY, 1)} ns/day aggregate (2 replicas on 1 A100 + 8 cores)"],
]
story += [table(rows, [3.0, 5.3, 8.7]), Spacer(1, 4),
          P("The collective variables (R<sub>g</sub>, core and loop rmsd, Q) are computed for each copy after "
            "rebuilding the molecule across the periodic boundary, relative to the mean structure of the first "
            "half of the AA trajectory. Overlaps between distributions are "
            "∫ min(p<sub>AA</sub>, p<sub>CG</sub>) dx (1 = identical). The <b>ceiling</b> is the overlap between "
            "the second and the first half of the AA trajectory: it is the best agreement one can ask for with "
            "this reference statistics.")]

# ── 2. Model ──
story += [PageBreak(), P("2. The CG-ML model: mapping, priors and PaiNN residual", "h1"),
          P("The potential energy of the model is the sum of two parts: U(R) = U<sub>prior</sub>(R) + "
            "U<sub>ML</sub>(R). U<sub>prior</sub> contains analytic terms with a physical meaning "
            "(connectivity, local geometry, Hoogsteen bonds, stacking, excluded volume), fitted to the mapped "
            "all-atom structure at 300 K. U<sub>ML</sub> is a residual learned with a PaiNN network; it corrects "
            "what the priors do not describe. Solvent and ions are implicit: the production model has no "
            "electrostatic term."),
          fig("fig_model_en.png", 17.0,
              "Figure 1. The CG model on the native 2JPZ structure (starting structure of the AA simulations, "
              "mapped onto the CG sites; the atomistic DNA in transparent grey). <b>a</b>: sites coloured by type; "
              "harmonic backbone bonds and the rigid skeleton of the guanines in dark grey; Morse contacts "
              "dashed, by class. <b>b</b>: tetrad 1 (G4, G12, G16, G22) viewed along the quadruplex axis, with "
              "the six B3–B3 Morse terms (sides and diagonals) and the four Hoogsteen B2–B4 Morse terms. "
              "<b>c</b>: the graph on which PaiNN acts for the B3 site of G16; the sphere has radius "
              "r<sub>c</sub> = 1.26 nm and contains 76 of the other 85 sites of the copy. Angles, dihedrals and "
              "WCA are not drawn. Rendered with PyMOL.")]

story += [P("2.1 Mapping", "h2"),
          P("Each 26-nucleotide copy is represented by 86 sites of 8 types. Each site is the centre of mass of a "
            "group of atoms:"),
          B("<b>guanine</b>: six sites joined in a <b>rigid body</b>. S collects sugar and phosphate (P, O5′, "
            "C5′, C4′, O4′, C1′, C3′, C2′, O3′). B1–B5 split the base: B1 = N9, C4; B2 = N3, C2, N2; "
            "B3 = N1, C6, O6; B4 = C5, N7; B5 = C8. B3 carries the O6 facing the channel, B2 and B4 the donor "
            "and acceptor of the second Hoogsteen bond (N2–H···N7), B5 the stacking face;"),
          B("<b>thymine and adenine</b>: a single site, the centre of mass of the whole nucleotide;"),
          B("<b>dynamics</b>: Langevin in ESPResSo. The 12 guanines are rigid bodies with mass and inertia "
            "tensor from the mapping (6 degrees of freedom); A and T are point masses. In total there are 114 "
            "degrees of freedom per copy."),
          P("Note: the definition of S also lists the non-bridging oxygens under the names O1P/O2P, whereas the "
            "force field calls them OP1/OP2. They therefore do not enter the centre of mass, in the same way in "
            "the reference and in all datasets.", "cap")]

story += [P("2.2 Physical priors (set <i>lp2</i>)", "h2"),
          P("The parameters come from the mapped AA structure at 300 K (10 copies, 136 ns). Bonds and angles are "
            "obtained by Boltzmann inversion. For the Morse terms r<sub>0</sub> is the median distance, and "
            "a = √(kT / 2Dσ<sup>2</sup>) uses the observed width σ, later refined iteratively on the distributions "
            "of the reference. The contacts are pair-specific: each one binds two given sites of the native fold. "
            "At run time they are non-bonded interactions on marker sites, with a tail smoothly switched to zero "
            "at r<sub>cut</sub>, so they can break without \"bond broken\" errors.")]
rows = [["term", "form", "sites", "per copy", "parameters"],
        ["backbone bonds", "½ k (r − r<sub>0</sub>)<super>2</super>", "site S, A or T of consecutive residues",
         "25", "k 136–3755 kJ mol<super>−1</super> nm<super>−2</super>; r<sub>0</sub> 0.58–0.70 nm"],
        ["backbone angles", "½ k (θ − θ<sub>0</sub>)<super>2</super>", "three consecutive residues", "24",
         "k 5–324 kJ mol<super>−1</super> rad<super>−2</super>; θ<sub>0</sub> 117–157°"],
        ["backbone dihedrals", "cosine (n = 1), damped when an angle approaches 180°", "four consecutive residues",
         "23", "K 2.4–74 kJ/mol"],
        ["stacking twist", "cosine (n = 1)", "B3–B5–B5–B3 of stacked guanines", "8",
         "K 391–774 kJ/mol (157–310 kT)"],
        ["Morse within a tetrad", "D [1 − e<super>−a(r−r<sub>0</sub>)</super>]<super>2</super>",
         "B3–B3, sides and diagonals (K<sub>4</sub>)", "18", "r<sub>0</sub> 0.38–0.65 nm; a 3.5–11.5 nm<super>−1</super>"],
        ["Hoogsteen Morse", "same", "B2–B4 (N2–H···N7)", "12", "r<sub>0</sub> 0.31–0.57 nm; a 6.1–11.0 nm<super>−1</super>"],
        ["stacking Morse", "same", "B5–B5, same G-tract, adjacent tetrads", "8",
         "r<sub>0</sub> 0.37–0.45 nm; a 5.9–8.9 nm<super>−1</super>"],
        ["loop caps", "same", "loop/tail bases against the face of the outer tetrads", "16",
         "r<sub>0</sub> 0.45–0.89 nm; a 2.3–5.1 nm<super>−1</super>"],
        ["excluded volume", "WCA, LJ 12-6 truncated and shifted at 2<super>1/6</super>σ", "all pairs of types (1-2 and 1-3 excluded)",
         "36 type pairs", "ε 3.15 kJ/mol; σ 0.24–0.68 nm"]]
story += [table(rows, [3.1, 3.6, 4.1, 1.6, 4.6]), Spacer(1, 4),
          P("All Morse terms have D = 50 kJ/mol (about 20 kT at 300 K), 2700 kJ/mol per copy in total: this is a "
            "native-state, Gō-like model, suited to 300 K but not to unfolding (see section 9).", "cap")]

story += [P("2.3 PaiNN residual", "h2"),
          P("U<sub>ML</sub> is a sum of site energies, U<sub>ML</sub> = Σ<sub>i</sub> ε<sub>i</sub>, computed by a "
            "PaiNN network (polarizable atom interaction neural network, Schütt et al. 2021). The network is "
            "equivariant under rotations and invariant under translations and permutations of sites of the same "
            "type:"),
          B("<b>input</b>: site type (8 species → embedding of dimension D = 64) and positions. The graph "
            "includes all pairs of sites within r<sub>c</sub> = 1.2616 nm, also between different copies "
            "(figure 1c);"),
          B("<b>radial basis</b>: 32 Gaussians centred between 0 and r<sub>c</sub>, width r<sub>c</sub>/32, "
            "multiplied by the Toxvaerd cutoff function x<super>4</super>/(x<super>4</super> + "
            "α<super>4</super>), with x = (r<sub>c</sub> − r)/r<sub>c</sub> and α = 0.1. The energy is smooth at "
            "r<sub>c</sub> up to the third derivative;"),
          B("<b>interaction</b>: 2 message + update blocks on scalar features s<sub>i</sub> ∈ "
            "ℝ<super>64</super> and vector features v<sub>i</sub> ∈ ℝ<super>3×64</super> (canonical PaiNN form, "
            "SiLU activations);"),
          B("<b>output</b>: MLP 64 → 32 → 1 on the final scalar of each site. The value of the same species in "
            "isolation is subtracted from each ε<sub>i</sub>, so that U<sub>ML</sub> → 0 for sites far from "
            "everything. The network has 106 049 parameters in total;"),
          B("<b>forces</b>: F = −∇U<sub>ML</sub> by automatic differentiation with respect to the site positions. "
            "On the guanines they are reduced to force and torque on the rigid body. The network runs in the "
            "ESPResSo plugin in libtorch, FP32 on GPU, and the forces are conservative by construction."),
          P("<b>Training: relative entropy, not force matching.</b> The instantaneous AA forces projected onto the "
            "CG sites are dominated by noise: the learnable part is R<super>2</super> ≈ 0.5–1 %. Force matching "
            "on top of already fitted priors degraded the structure (intramolecular overlap from 0.976 to 0.94). "
            "The residual was therefore trained by minimising the relative entropy "
            "S<sub>rel</sub> = ⟨ln p<sub>AA</sub>/p<sub>CG</sub>⟩<sub>AA</sub>, which uses positions only. The "
            "gradient is β[⟨∂U/∂θ⟩<sub>AA</sub> − ⟨∂U/∂θ⟩<sub>CG</sub>]; the priors do not appear because they do "
            "not depend on θ."),
          B("each iteration runs 25 ps of CG MD with the current model; the samples are reused by reweighting as "
            "long as the effective sample size per copy stays ≥ 50 %;"),
          B("the step is controlled by a trust region, and the last 20 % of the AA trajectory is held out as a "
            "check;"),
          B("<b>re0</b>: 30 iterations on top of the lp1 priors, starting from U<sub>ML</sub> ≡ 0 (last output "
            "layer set to zero). <b>re1</b>: 30 iterations on top of lp2 (lp1 plus the loop caps), starting from "
            "re0_it30. The production model is re1_it30;"),
          B("AA data: mapped positions of prod-1 (136 ns, 10 copies, one frame every 20 ps) at 300 K."),
          P("Effect of the residual on the intramolecular overlap (all sites), with the same protocol: lp2 priors "
            "0.981 → lp2 + re1_it30 0.994, against an AA vs AA ceiling of 0.9955. The largest gain is in the "
            "B3–B3 channel (0.84 → 0.97).", "cap")]

# ── 3. P(r) ──
story += [PageBreak(), P("3. Intramolecular pair distributions P(r)", "h1"),
          P("For each channel, P(r) is the normalised histogram of the distances between sites of the same copy "
            "(same bins and normalisation as script 44). The channels separate the main contributions to the "
            "structure: B3–B3 for the Hoogsteen pairing in the tetrads, B5–B5 for tetrad stacking, S–S for the "
            "backbone; the \"all sites\" channel sums them, loops included."),
          fig("fig_Pr_intra_en.png", 16.5,
              "Figure 2. Intramolecular P(r) per channel. Blue: AA over 136 ns; orange: CG-ML over 2 × 20 ns; grey "
              "dashed and dotted: first and second half of the AA. Top right: the CG–AA overlap and the ceiling "
              "(half 2 vs half 1).")]
rows = [["channel", "CG-ML vs AA", "ceiling (AA half 2 – half 1)", "difference"]]
for c in CH:
    a, h = OV[c]["CG-ML"]["intra"], OV[c]["AA meta' 2"]["intra"]
    rows.append([CH_EN.get(c, c), it(a), it(h), f"{a - h:+.3f}"])
story += [table(rows, [4.4, 3.6, 5.0, 3.0]), Spacer(1, 6),
          P("In every channel the CG-ML reaches the ceiling: the gap is at most 0.004, within the variation between "
            "the two halves of the reference. Peak positions and heights coincide: for B3–B3 0.34, 0.445, 0.65 and "
            "0.84 nm; for S–S the first neighbour at 0.595 nm. The remaining differences, for instance the S–S "
            "peaks at 1.17 and 1.42 nm slightly higher in the CG, have the same amplitude as the fluctuations "
            "between the AA halves. The geometry of the core (tetrads, stacking, backbone) is therefore reproduced "
            "at the level of distributions, not only on average.")]

# ── 4. g(r) inter ──
story += [PageBreak(), P("4. g(r) between different copies (short range)", "h1"),
          P(f"The intermolecular g(r) counts pairs of sites on different copies. Up to {it(2.0, 1)} nm, the largest "
            f"distance computed, it describes only close encounters between copies: with R<sub>g</sub> ≈ 0.8 nm the "
            f"copies rarely touch and g(r) stays well below 1, in the excluded-volume region."),
          fig("fig_gr_inter_en.png", 16.5,
              "Figure 3. g(r) between sites of different copies, same colours as figure 2. The two AA halves "
              "differ by a factor of ~5: the encounters between copies in the AA are few long events.")]
rows = [["channel", "CG-ML vs AA", "AA half 2 – half 1"]]
for c in CH:
    rows.append([CH_EN.get(c, c), it(OV[c]["CG-ML"]["inter"]), it(OV[c]["AA meta' 2"]["inter"])])
story += [table(rows, [4.4, 3.6, 4.4]), Spacer(1, 6),
          P("In the AA the first half has almost no contacts between copies, the second half many more: these "
            "contacts are rare, non-ergodic events over 136 ns, and the reference does not sample them. The CG–AA "
            "overlap (0.68–0.93) exceeds that between the two AA halves (0–0.85), so here the metric does not "
            "discriminate. The CG-ML has somewhat more frequent contacts between copies than the AA average, similar "
            "to the second half. <b>Conclusion</b>: with this reference the short-range structure between copies "
            "cannot be verified. For the full intermolecular g(r) (distances up to half the box) it is enough to "
            "rerun script 53 with a larger <font name='DV-I'>--rmax</font>, but the AA uncertainty will remain "
            "the one set by 10 copies in a single box.")]

# ── 5. Rg ──
story += [PageBreak(), P("5. Radius of gyration", "h1"),
          fig("fig_Rg_en.png", 16.5,
              "Figure 4. Left: R<sub>g</sub> distribution (AA: single copies as thin lines, all copies as a thick "
              "line; CG-ML: all copies and replicas). Right: time series with a 1 ns running mean, AA over 136 ns "
              "and CG-ML (replica r0) over 20 ns, same vertical scale.")]
rows = [["", "AA", "CG-ML"],
        ["mean ± standard deviation (nm)", f"{it(cv['Rg']['aa_mean'])} ± {it(cv['Rg']['aa_std'])}",
         f"{it(cv['Rg']['cg_mean'])} ± {it(cv['Rg']['cg_std'])}"],
        ["means of the single copies (nm)", f"{it(cv['Rg']['aa_copy_min'])}–{it(cv['Rg']['aa_copy_max'])}",
         "all ≈ 0.800"],
        ["τ at 1/e (script 47)", f"{it(tauAA['tau_Rg'] / 1000, 1)} ns", f"{it(tauCG['tau_Rg'], 1)} ps"],
        ["overlap with the AA (ceiling)", "–", f"{it(cv['Rg']['ov'], 2)} ({it(cv['Rg']['ceil'], 2)})"]]
story += [table(rows, [6.0, 5.0, 5.0]), Spacer(1, 6),
          P("The mean value is correct within 0.002 nm. The CG distribution is somewhat wider than the overall AA "
            "one and much wider than that of a single AA copy. The reason is in the time series: in the AA each "
            "copy stays for tens of ns on its own R<sub>g</sub> (the most compact copy at 0.78 nm for most of the "
            "trajectory), with a slow 5–10 ns relaxation linked to loop rearrangements. In the CG the 1 ns average "
            "is flat and identical for all copies, because R<sub>g</sub> decorrelates in a few ps. The CG thus "
            "samples a single \"average\" basin instead of the states that the AA visits slowly. For R<sub>g</sub> "
            f"the ratio of the τ values (~{round(alpha['tau_Rg'], -2):.0f}) measures different processes and is "
            "not a speed-up.")]

# ── 6. CV ──
story += [KeepTogether([P("6. Other collective variables", "h1"),
          fig("fig_cv_en.png", 16.5,
              "Figure 5. Distributions of core rmsd, loop rmsd and fraction of native contacts Q (grey: AA halves). "
              "The titles report the CG–AA overlap and the ceiling.")])]
rows = [["variable", "AA mean ± std", "CG-ML mean ± std", "overlap", "ceiling"]]
names = {"Rg": "R<sub>g</sub> (nm)", "rmsd_core": "core rmsd (nm)", "rmsd_loops": "loop rmsd (nm)", "Q": "Q"}
for v in ["Rg", "rmsd_core", "rmsd_loops", "Q"]:
    s = cv[v]
    rows.append([names[v], f"{it(s['aa_mean'])} ± {it(s['aa_std'])}", f"{it(s['cg_mean'])} ± {it(s['cg_std'])}",
                 it(s["ov"], 2), it(s["ceil"], 2)])
story += [table(rows, [3.4, 3.5, 3.5, 3.2, 1.9]), Spacer(1, 6),
          P("The core is at the ceiling: rmsd 0.044 nm in both, overlap 0.93 vs 0.94. Q has the same mean but a "
            "somewhat narrower distribution in the CG (std 0.009 vs 0.013). The loops are the only clear "
            "discrepancy: in the CG they are more mobile and further from the mean structure (0.27 vs 0.22 nm), and "
            "the population at low rmsd (0.13–0.2 nm), which in the AA corresponds to the compact loop states, is "
            "missing. The diagnostics of scripts 49–52 attribute the gap to metastable states absent in the CG "
            "(state 1 of loop 3, minor state of loop 2) and to a partly wrong pairing register (T2–A15 in excess)."),
          fig("fig_fes_en.png", 15.0,
              "Figure 6. Free energy F = −kT ln p(core rmsd, Q), AA and CG-ML. The JSD between the FES (script 46) "
              "is 0.0787, equal to the AA vs AA ceiling (0.0786); the CG is noisier because of the smaller number "
              "of frames (20 080 vs 67 880 copy-frames).")]

# ── 7. MSD ──
story += [PageBreak(), P("7. Dynamics: translational and rotational MSD", "h1"),
          P("MSD of the centre of mass of each copy and angular MSD of the core orientation (Kabsch alignment, "
            "rotations between successive frames summed as rotation vectors in the laboratory frame; script 48). "
            "The curves stop at a quarter of the length of each trajectory. In the bottom row the time axis is "
            "converted into GPU hours with the measured throughput."),
          fig("fig_msd_en.png", 16.5,
              "Figure 7. Top: MSD vs simulated time; dashed 6Dt with D from the linear regime. Bottom: the same "
              "curves vs A100 GPU hours (AA 137 ns/day; CG-ML 54.2 ns/day aggregate over 2 replicas).")]
rows = [["", "AA", "CG-ML", "CG/AA ratio"],
        ["D (nm<super>2</super>/ns), linear fit", it(ms["AA"]["D_trans"]), it(ms["prod"]["D_trans"], 1),
         f"{ms['prod']['D_trans'] / ms['AA']['D_trans']:.0f}"],
        ["τ<sub>v</sub> from the Langevin fit (ps)", it(ms["AA"]["tau_v_ps"], 1), f"{ms['prod']['tau_v_ps']:.0f}", ""],
        ["translational MSD per GPU hour (nm<super>2</super>)", it(rate["D_trans"], 2),
         f"{rate_cg['D_trans']:.0f}", f"<b>{rate_cg['D_trans'] / rate['D_trans']:.0f}×</b>"],
        ["D<sub>rot</sub> (rad<super>2</super>/ns), linear fit", it(ms["AA"]["D_rot"]), it(ms["prod"]["D_rot"], 1),
         f"{ms['prod']['D_rot'] / ms['AA']['D_rot']:.0f}"],
        ["τ<sub>P2</sub> (ps)", f"{ms['AA']['tau_P2']:.0f}", it(ms["prod"]["tau_P2"], 1),
         f"α = {ms['AA']['tau_P2'] / ms['prod']['tau_P2']:.0f}"],
        ["1/(6D<sub>rot</sub>) (ps)", f"{1000 / (6 * ms['AA']['D_rot']):.0f}", it(1000 / (6 * ms['prod']['D_rot']), 1), ""],
        ["angular MSD per GPU hour (rad<super>2</super>)", it(rate["D_rot"], 2), f"{rate_cg['D_rot']:.0f}",
         f"{rate_cg['D_rot'] / rate['D_rot']:.0f}× (overestimate)"],
        ["orientation decorrelation per GPU hour", "", "",
         f"<b>{ms['AA']['tau_P2'] / ms['prod']['tau_P2'] * R:.0f}×</b>"]]
story += [table(rows, [7.0, 2.6, 2.6, 3.8]), Spacer(1, 6),
          B("<b>Translation</b>: in the CG the regime is ballistic up to ~0.1–0.3 ns (τ<sub>v</sub> ≈ 100 ps), then "
            "diffusive; D is 0.68 of the value kT/(Nγ) expected for an isolated molecule. In the diffusive regime "
            "the MSD per GPU hour equals 6D × (ns per hour), i.e. α × throughput ratio: "
            f"{rate_cg['D_trans'] / rate['D_trans']:.0f}×. With a few minutes of computation the CG is still "
            "ballistic and the gain is smaller."),
          B("<b>Rotation</b>: in the AA τ<sub>P2</sub> ≈ 1/(6D<sub>rot</sub>), so rotation is diffusive and the "
            "method is confirmed. In the CG τ<sub>P2</sub> is ~7 times 1/(6D<sub>rot</sub>): the orientation "
            "decorrelates already in the inertial regime (low γ), and the angular MSD overestimates the gain in "
            "sampling orientations. For rotation the number to use is that of the decorrelation, "
            f"{ms['AA']['tau_P2'] / ms['prod']['tau_P2'] * R:.0f}×.")]

# ── 8. efficiency ──
story += [P("8. Efficiency per GPU", "h1"),
          P("Efficiency = α × (ns/day<sub>CG</sub> / ns/day<sub>AA</sub>), with α = τ<sub>AA</sub>/τ<sub>CG</sub> "
            f"(1/e times from script 47) or the ratio of the D values; ns/day<sub>CG</sub>/ns/day<sub>AA</sub> = "
            f"{it(CG_NSDAY, 1)}/{AA_NSDAY:.0f} = {it(R, 3)}.")]
rows = [["observable", "τ AA", "τ CG-ML", "α", "efficiency per GPU"],
        ["translation (MSD)", "", "", f"{ms['prod']['D_trans'] / ms['AA']['D_trans']:.0f} (ratio of D)",
         f"{rate_cg['D_trans'] / rate['D_trans']:.0f}×"],
        ["core sites", f"{tauAA['tau_core_sites']:.0f} ps", "0.48 ps*", "363*", "~140×"],
        ["rotation (τ<sub>P2</sub>)", f"{it(tauAA['tau_rot'] / 1000, 2)} ns", f"{it(tauCG['tau_rot'], 1)} ps",
         f"{alpha['tau_rot']:.0f}", f"{alpha['tau_rot'] * R:.0f}×"],
        ["<b>core rmsd</b>", f"{it(tauAA['tau_core'], 1)} ps", f"{it(tauCG['tau_core'], 2)} ps",
         f"{alpha['tau_core']:.0f}", f"<b>{alpha['tau_core'] * R:.0f}×</b> (minimum)"],
        ["R<sub>g</sub>, loops", f"{it(tauAA['tau_Rg'] / 1000, 1)}, {it(tauAA['tau_loops'] / 1000, 1)} ns",
         f"{it(tauCG['tau_Rg'], 1)}, {it(tauCG['tau_loops'], 1)} ps", "~2400", "not interpretable"]]
story += [table(rows, [4.2, 2.6, 2.6, 3.4, 4.2]), Spacer(1, 4),
          P("* With one frame per ps the CG τ of the core sites (~0.5 ps) is not resolved (script 47 gives "
            f"{it(tauCG['tau_core_sites'], 2)} ps by interpolation); τ and α are taken from the run sampled every "
            "0.5 ps.", "cap"),
          P("The number to quote is the minimum among the comparable observables: <b>~28× per GPU</b> (core rmsd), "
            "with translation and local core motions at ~140× and rotation at ~44×. Per node budget the advantage "
            "can grow up to ~4×, because the AA uses 32 cores against 8 for the CG; the AA throughput with 8 cores "
            "per GPU is needed to pin this down.")]

# ── 9. limitations ──
story += [P("9. Limitations and caveats", "h1"),
          B("<b>Loops</b>: a single basin instead of the AA metastable states. State 1 of loop 3 (26 %) and the "
            "minor state of loop 2 are missing; the 5′ tail is too mobile; the T2–A15 contact is in excess "
            "(0.56 vs 0.24 ± 0.09). State-dependent contacts (lp2c0) were tried and not adopted: with a single site "
            "for T and A the register specificity cannot be obtained with pair contacts."),
          B("<b>AA reference not ergodic on the loops</b>: 17–61 % of the loop variance is between copies and the "
            "slow times are 4–10 ns over 136 ns. The AA populations of the loop states have large errors, and the "
            "ceilings on loops and R<sub>g</sub> are optimistic: the two halves share the same stuck copies."),
          B("<b>Native state only</b>: the 54 Morse terms at 50 kJ/mol per copy keep the quadruplex closed at any "
            "accessible temperature. With the priors alone, on one copy, no tetrad opens between 300 and 600 K in "
            "250 ns, and the potential energy grows like that of a harmonic system. The model is meant for the "
            "structure and dynamics around the native state; for unfolding thermodynamics the contact depths must "
            "be recalibrated (work in progress, folder <font name='DV-I'>tel26_unfold</font>)."),
          B("<b>g(r) between copies</b>: verified only at short range (≤ 2 nm), where the AA does not converge "
            "either."),
          B("<b>Efficiency</b>: per A100 GPU, on the same hardware; the ratio per GPU hour depends on the node "
            "throughput, which varies by ~20 % between nodes with the same code."),
          Spacer(1, 6)]

# ── appendix ──
story += [P("Appendix: data provenance", "h2"),
          P("Data: <font name='DV-I'>report_data.npz</font> generated on Leonardo by "
            "<font name='DV-I'>53_tel26_report_data.py</font> (P(r) and g(r) with the routines of script 44; "
            "collective variables with script 46; MSD curves from <font name='DV-I'>msd_wall_prod.json</font>, "
            "script 48; times from <font name='DV-I'>ts_prod.json</font>, script 47). AA reference: "
            "<font name='DV-I'>tel26_lp1_dataset.bin</font>. CG-ML: <font name='DV-I'>prod_re1_s0{1..4}_r{0,1}"
            ".samples.npz</font> (chain prod_re1, model lp2 + re1_it30). Figures and PDF: "
            "<font name='DV-I'>make_figs_en.py</font>, <font name='DV-I'>build_report_en.py</font> and "
            "<font name='DV-I'>model_fig/</font> in the folder <font name='DV-I'>tutorials/tel26/report</font>. "
            f"Parameters: P(r) with one AA frame every {meta['aa_stride']}, CG every {meta['cg_stride']} ps, "
            "200 bins up to 2 nm.", "cap")]

doc = SimpleDocTemplate("tel26_report_AA_vs_CGML_en.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="TEL26: AA vs CG-ML",
                        author="PaiNN-LT", subject="Structural and dynamical comparison, AA vs CG-ML")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("ok")
