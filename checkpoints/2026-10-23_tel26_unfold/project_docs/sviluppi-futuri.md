# PaiNN-LT — sviluppi futuri annotati

Annotato il 2026-10-01. Decisione di Cristiano: **prima si chiude il lavoro in corso sul TEL26 nativo** (catena re1 da it31, 1 ns di re1 it08 e it30, g(r) su prod-2), poi si riprendono questi punti.

## 1. S(q) sperimentale nel training (proposito di Cristiano)

**Obiettivo.** Confrontare il modello CG con S(q) sperimentali, e poi usarle come bersaglio di allenamento.

**Calcolo dalla simulazione** (suggerimento di Cristiano): trasformata di Fourier spaziale della densità sui vettori del reticolo reciproco della scatola, invece della formula di Debye.

- ρ(q) = Σⱼ fⱼ(q) e^{i q·rⱼ} con q = 2πn/L; S(q) = ⟨|ρ(q)|²⟩/N, mediata su gusci in |q|.
- Costo O(N·N_q) per frame. Include da sé le correlazioni fra copie.

**Vincoli noti.**

- Con L = 11,9 nm, q_min = Δq ≈ 0,53 nm⁻¹: il basso q (Rg) richiede la Debye sulla copia singola oppure una scatola più grande.
- Sotto 2–3 nm⁻¹ domina il termine fra copie: serve la concentrazione sperimentale (ora circa 10 mM con 10 copie) e più statistica di quanta ne danno 10 copie.
- Il CG ha solvente implicito: servono fattori di forma in eccesso rispetto al solvente (raggi X) o lunghezze di scattering con contrasto (neutroni), per sito CG.
- Dall'AA si può calcolare S(q) con tutti gli atomi, acqua e K⁺ compresi. È il confronto più diretto con la misura, e dice se l'AA stesso è d'accordo con l'esperimento.

**Uso nel training: schema DiffTRe** (Thaler e Zavadlav, Nat. Commun. 2021).

- ⟨O⟩_θ = Σ wᵢ(θ) O(xᵢ), con wᵢ ∝ e^(−βΔUᵢ) e ∂⟨O⟩/∂θ = −β Cov(O, ∂U/∂θ). Stesso ripesaggio e stessa regione di fiducia (ESS/N ≥ 0,5) di train_relent.
- Loss: S_rel + λ Σ (⟨O⟩ − O_bersaglio)²/σ². Termine aggiunto al RE, non sostitutivo.
- Limite: come il RE, corregge solo configurazioni che il CG visita. Non sostituisce prior che rendano metastabile la struttura giusta (lezione di lp1 → lp2).

**Passi, quando si riprende.**

1. Script di S(q) per trasformata di Fourier su AA (tutti gli atomi), AA mappato e CG, prima con fattori di forma unitari, poi con fattori di forma per sito.
2. Confronto a tre: esperimento, AA, CG.
3. Solo dopo, un termine DiffTRe in train_relent, con l'osservabile per frame |ρ(q)|²/N.

**Da chiedere a Cristiano**: tipo di misura (SAXS, WAXS o SANS), costrutto (Tel26 in K⁺?), concentrazione, range di q, formato dei dati.

**Utile soprattutto** per il modello unfoldable, dove un riferimento AA completo non ci sarà, e per sistemi con dati sperimentali ma traiettorie AA corte.

## 2. Altri punti emersi

- **DiffTRe mirato sui loop del TEL26** (T2, T13, T14, A19–A21): solo se la continuazione della catena re1 non li sistema. È meno invasivo di un insieme lp3.
- **Automatizzare il passo "lp2"** per sistemi nuovi: individuare dal riferimento AA le basi rigide (RMSF a nucleo fermo) e dare loro cappucci con tre partner sulla faccia, usando lo script 46 come diagnostica per residuo.
- **Interazioni fra copie**: non giudicabili con 10 copie (tetto inter 0,84). Servirebbe un riferimento AA con più copie o più lungo.
