---
name: painn-lt
description: Il progetto PaiNN-LT (MLCG_Framework_v2, residuo ML PaiNN su prior CG per G-quadruplex) — obiettivo e decisioni; leggere per qualunque lavoro sul progetto
sources: [cowork]
aliases: [mlcg, mlcg_framework_v2, tel26, tel22]
---

[stated] Obiettivo del progetto: riprodurre la g(r) delle simulazioni atomistiche (all-atom) con il modello CG (prior + residuo ML).
[stated] Fase attuale: studio di fattibilità su prior analitici non convenzionali (Morse, FENE, LJ, ecc.); tetradi per ora tenute formate.
[stated] Scopo finale: aggiungere il residuo ML al modello TEL26 unfoldable sviluppato con una collega (LJ per legami H/elettrostatici, ioni impliciti); i dettagli di quel modello non sono ancora disponibili.
[stated] Vincolo sui prior: devono corrispondere a interazioni fisiche (ok contatti di Hoogsteen); niente reti elastiche di vincoli fra basi lontane. Criterio ultimo resta una g(r) corretta.
[stated] asked to remember: le lezioni del TEL26 per il prossimo sistema (ricetta non ancora validata: finora nessun modello riproduce la g(r)) (dettaglio nel doc di progetto claude/lezioni-cg-ml.md) — separare subito struttura intra e interazioni inter; solo prior fisici (Hoogsteen, impilamento fra basi sovrapposte, torsione) con parametri aggiustati per iterazione sulla g(r); automatizzare le iterazioni e lanciarle in parallelo; ML per ultimo come correzione, sapendo che ottimizza le forze e non la struttura; salvare le forze all-atom spesso (nstxout = nstfout = 500); giudicare sulla g(r) per canale, non sulle forze.
[stated] asked to remember (stesse lezioni): per la sola struttura locale su ns–µs conviene l'all-atom; il CG ripaga su scale non raggiungibili dall'all-atom (unfolding, aggregazione, sistemi grandi).
[stated] asked to remember (1/10/2026): proposito di introdurre il confronto con S(q) sperimentali (e poi usarle nel training) — S(q) da calcolare per trasformata di Fourier spaziale della densità nella simulazione; da riprendere solo dopo aver chiuso il lavoro in corso sul TEL26 nativo (piano nel doc di progetto claude/sviluppi-futuri.md).
[stated] Criterio di prosecuzione (ottobre 2026): il CG con ML ha senso solo se è almeno un ordine di grandezza più efficiente dell'AA (fattore 10–100); altrimenti è inutile continuare. Priorità attuale: valutare questa efficienza (fattore di scala temporale, γ/dt, ottimizzazione del plugin).
[stated] asked to remember (2/10/2026): manca ancora l'opzione C dell'ottimizzazione (prior Morse COM–COM e contatti dei cappucci come legami invece di marker N²); fatte A (grafo PaiNN sul device), D (log vettoriale) e min_dist con immagine minima (dettaglio in claude/efficienza-cg-aa.md).
[stated] (5/10/2026) Valutava se far funzionare il vecchio framework (MLCG_Framework, prior da IBI) oppure passare al modello unfoldable; ha creato la cartella MLCG_Framework_v2/tutorials/tel26_unfold per lavorare sull'unfoldable.
[stated] (6/10/2026) Decisione: passare al modello unfoldable (la sfida più interessante), mantenendo gli ioni impliciti; partire dal modello TEL26 nativo e lavorare in modo incrementale introducendo la possibilità di unfolding; dati di partenza le corse AA ad alta T di Giulia (piano nel doc di progetto claude/unfold-tel26.md).
[stated] (6/10/2026) Per l'unfoldable vuole lanciare in proprio simulazioni all-atom di TEL26 che si srotolino, in una cartella separata da quella di Giulia (per non interferire): /leonardo_work/IscrB_G4MES/cdemiche/AA_unfold.
[stated] (7/10/2026) Obiettivo finale dell'unfoldable: stimare la temperatura di transizione (Tm), anche in presenza di ligandi; inizialmente senza ligandi. Vuole comunque usare PaiNN nel modello unfoldable. La Tm sperimentale del TEL26 è grossomodo nota.
[stated] (7/10/2026) Per l'unfoldable va avanti con PaiNN a due teste (entalpica ed entropica, U = H − T·S) e ne ha chiesto l'implementazione.
[stated] (23/10/2026) Per l'unfoldable il riferimento sperimentale resta comunque l'articolo di Buscaglia, Gray, Chaires (Biopolymers 2013); ha accettato di definire gli stati del CG come quelli di Buscaglia (F+I1 / tripla elica I2 / U) invece del numero di tetradi intatte. Poi (23/10), visto che il CG non ha la tripla elica, ha approvato di vincolare solo la curva U(T) di Buscaglia con U definito come bacino (nessun tratto G appaiato), ripartendo da k6 (catena pop3).
