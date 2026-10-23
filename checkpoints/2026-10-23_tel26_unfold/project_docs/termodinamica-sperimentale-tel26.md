# TEL26 (wtTel26, 2JPZ): dati sperimentali di unfolding per la calibrazione del CG

Raccolti il 7/10/2026 per l'obiettivo finale del modello unfoldable: Tm, anche con ligandi. Vedi `unfold-tel26.md`.

## Nomenclatura

- La nostra sequenza (TTAGGG)4TT = 5′-TTAGGGTTAGGGTTAGGGTTAGGGTT-3′ è la **wtTel26**, ibrido-2, PDB **2JPZ** (Dai et al., NAR 2007, 35, 4927).
- In letteratura "Tel26" indica spesso AAAGGG(TTAGGG)3AA, ibrido-1, PDB 2HY9: è un'altra sequenza.

## Buscaglia, Gray, Chaires, Biopolymers 2013 (doi 10.1002/bip.22247; supplementari NIHMS648440)

"Thermodynamic Characterization of Human Telomere Quadruplex Unfolding". Articolo e supplementari forniti da Cristiano.

- **Sequenza 2JPZ identica alla nostra.** Altre sequenze dello studio: ibrido-1 2GKU, 2HY9, 2JSM; ibrido-2 2JPZ, 2JSL; ibrido-3 2KF8, 2KKA; Tel22.
- **Condizioni**: 25 mM KCl, 10 mM tetrabutilammonio fosfato, 1 mM EDTA, pH 7,0; DNA 3–4 µM; CD 220–350 nm, 4–94 °C, ricotto. Melting riproducibili.
- **Meccanismo**: non a due stati. SVD con 4 specie significative; sequenziale **F ⇌ I1 ⇌ I2 ⇌ U**, fit globale dei primi 4 vettori V pesati, errori da Monte Carlo.
  - I1: riarrangiamento sottile di fianchi e impilamento del nucleo, spettro ancora "ibrido".
  - I2: spettro da tripla elica G-rica, comune a tutte le sequenze. Un tratto G esce e restano tripletti (Gray, Buscaglia, Chaires, JACS 2012, 134, 16834, su Tel22).
  - Nessun intermedio duplex; nessuna conversione a forme basket/chair lungo il cammino.

### Parametri della 2JPZ (Supplemental Tables 1–3)

ΔH di folding in kcal/mol (negativi), Tm in °C, ΔG a 20 °C con ΔCp = 0:

| transizione | ΔH | Tm | ΔG(20 °C) | Tm iniziale (derivate CD) |
|---|---|---|---|---|
| F ⇌ I1 | −7,8 ± 1,8 | 37,5 ± 22,5 | −0,4 ± 0,5 | 38 |
| I1 ⇌ I2 | −36,4 ± 4,7 | 43,7 ± 2,9 | −2,7 ± 0,5 | 46 |
| I2 ⇌ U | −39,9 ± 4,8 | 54,3 ± 1,7 | −4,2 ± 0,7 | 56 |
| totale | **−84,1** (352 kJ/mol) | | **−7,3 ± 0,6** (30,7 kJ/mol) | |

- ΔG_i = ΔH_i (1 − T/Tm_i) riproduce la Supplemental Table 3 (verificato).
- Confronto con le altre sequenze:
  - 2JSL (ibrido-2, 25 nt): ΔH −13,3/−30,7/−38,8, Tm 39,2/47,4/52,9, ΔG_tot −7,3;
  - 2GKU (ibrido-1) è la più stabile: ΔG_tot −12,0, Tm3 60,8.
- **Popolazioni ricostruite** (`tel26_unfold/ref/tm_2jpz.py`, `tm_2jpz.json`, `popolazioni_2jpz.png`; equilibri sequenziali di van 't Hoff, ΔCp = 0):
  - **F 50 % a 33,9 °C** (Monte Carlo 16–84 %: 18–43 °C, dominato dall'incertezza di Tm1);
  - **U 50 % a 55,3 °C** (MC 53,8–57,0 °C): è la "Tm" utile;
  - I2 massimo ~0,51 a ~50 °C;
  - frazioni F/I1/I2/U a 293 K 0,68/0,32/0,003/0; a 300 K 0,60/0,38/0,015/0; a 310 K 0,44/0,43/0,12/0,004; a 330 K 0,02/0,04/0,36/0,58; a 350 K U 0,98.
  - Il testo dell'articolo dice I1 ~15 % a 20 °C; dai parametri della 2JPZ viene 32 %, ma con Tm1 ± 22,5 °C è dentro l'incertezza.
- **Estrapolazione a 400 K** (127 °C, ΔCp = 0, 25 mM K⁺): U ≈ 1. In AA a 400 K (~45 mM K⁺, 1 K⁺ nel canale) il quadruplex resta con tetradi 1–2 chiuse per 746 ns: o barriera cinetica su µs, o sovrastabilizzazione del campo di forza, o entrambe. Il pilota senza K⁺ aiuta a distinguere.
- **In 50 % acetonitrile** (propeller): tre stati; per la 2JPZ ΔH −55,9/−46,4, Tm 44,3/83,4, ΔG_tot −12,5. Non rilevante.

## Altri dati utili trovati (ricerca web del 7/10)

- **Legame del K⁺ alla 26TTA (2JPZ)** (Marchand & Gabelica, NAR 2016, 44, 10999): 2 K⁺ specifici, KD1 ≈ 1,5 mM, KD2 ≈ 0,19 mM a 20 °C. Molto più deboli che nell'ibrido-1 2GKU (0,27 mM, 10 µM). Ripiegamento in minuti–ore, cammini ramificati.
- **FRET-melting** (Nat. Commun. 2018, 9, 3496; Pt-tripode), 60 mM K⁺ cacodilato:
  - Tel26 (ibrido-1) libero 46,1 °C;
  - wtTel26 con 1 equivalente di ligando 54,0 °C.

  Con marcatori, quindi non usare come bersaglio assoluto. Utile per i ΔTm con ligandi.
- **Firme**: formazione entalpica (ΔH < 0, ΔS < 0); le unità tetrade–K⁺–tetrade sono fortemente entalpiche (Gabelica, bioRxiv 370254).
- **Modelli multistato con ΔH, ΔS e ΔCp per Tel22 in funzione di K⁺/Na⁺/PEG**: Bončina et al., Angew. Chem. 2016 (PMC5705027), parametri nelle tabelle S1–S2 da recuperare. Sempre di Bončina: JACS 2012, 134, 9657.

## Conseguenze per il CG

- **Bersaglio**: popolazioni di F, I1, I2 e U in funzione di T a 25 mM K⁺, con ΔH_tot ≈ 352 kJ/mol e ΔG_tot(20 °C) ≈ 31 kJ/mol. La Tm (U 50 %) è 55 °C.
- **Testa entalpica e testa entropica.** Con U = U_H − T·U_S, le ΔH di van 't Hoff vincolano U_H direttamente (ΔH_i = ⟨U_H⟩ differenze fra stati, se ΔCp ≈ 0), le Tm vincolano il rapporto.
- **Coordinate d'ordine CG** che separino gli stati:
  - Q per tetrade;
  - partecipazione di ciascun tratto G (I2 = un tratto fuori, tripletti restanti);
  - impilamento dei fianchi e delle tetradi esterne (I1, probabilmente non risolvibile nel CG: ΔH1 piccolo, Tm1 incerta).

  Bersaglio minimo robusto: F+I1 contro I2 contro U.
- **Confronto AA a 400 K**: nella corsa di Giulia la tetrade 3 si divide in due coppie di Hoogsteen (G18–G24, G6–G10), non esce un tratto intero. Il pilota senza K⁺ dirà se l'unfolding passa per una tripla elica: è un controllo strutturale diretto.
- [K⁺] sperimentale 25 mM; nelle corse AA ~45 mM (solo controioni). Il termine del K⁺ implicito va tarato con KD1 e KD2.
