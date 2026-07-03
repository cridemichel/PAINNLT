TODO LIST 

1) normalizzazione dell'energia (aggiungere shift and scale energia) 

SchNetPack affronta il problema delle grandezze dell'energia con due meccanismi integrati che corrispondono al nostro "Scale and Shift":

schnetpack.atomistic.Atomref: Questo modulo aggiunge al termine della rete l'energia di riferimento dei singoli atomi isolati (lo shift di base).

schnetpack.transform.Standardize: Durante il preprocessing, SchNetPack calcola statisticamente la media e la deviazione standard delle energie o delle forze rimaste e standardizza l'output. Questo garantisce che la rete neurale lavori sempre con numeri prossimi allo zero, lasciando che il framework ri-moltiplichi e ri-sommi i valori reali solo al momento dell'output finale.

2) implementare cosine cutoff

Se guardi nella cartella schnetpack/nn/cutoff.py, troverai un'intera classe chiamata CosineCutoff.
Quando in SchNetPack dichiari il modello PaiNN o SchNet, il parametro cutoff_network viene inizializzato di default proprio con questa classe. Usano esattamente la funzione coseno scalata che ti ho suggerito per garantire che le derivate spaziali scendano a zero in modo continuo sul bordo della Neighbor List, prevenendo salti discontinui delle forze.

3) Nella versione 2.0, SchNetPack ha delegato l'intero loop di addestramento a PyTorch Lightning.
Se controlli i loro file di configurazione per il training (gestiti tramite il sistema Hydra nella cartella configs/trainer), troverai il parametro gradient_clip_val. È prassi comune nei loro tutorial impostare questo parametro proprio intorno a 0.5 o 1.0. Come abbiamo discusso, poiché le forze sono la derivata dell'energia, senza questo "guinzaglio" al gradiente, una singola repulsione a corto raggio sfortunata nel batch rovinerebbe per sempre i pesi dell'ottimizzatore AdamW.

4) mixed precision


GROMACS TO BIN

1) dato un gruppo di atomi va preparato uno script che rimpiazzi questo gruppo di atomi
con un gruppo di siti virtuali e una particella reale o con una singola particella reale.  
Ad ogni sito virtuale va associato
un tipo (possiamo usare il numero atomico Z) e un mol_id (molecola a cui appartiene)
(di questo devo parlare con Laura)
Se non si hanno siti virtuali la singola particella reale avrà comunque un suo mol_id

2) nella costruzione delle interazioni in PaiNN atomi con stesso mol_id non dovranno interagire

3) In espresso i siti virtuali tra loro non interagiscono, quindi non si dovrà fare nulla,
solo importare il modello ed usarlo come già faccio ora.

4) molto probabilmente nella loss andranno inclusi anche i prior per evitare overlaps (usando WCA)
 o potenziali armonici (o FENE) per gli atomi legati.
 Le forze nella loss saranno quelle predette dalla rete + quelle che derivano dai prior.
 Il tutto andrà implementato così: in espresso i vari siti virtuali avranno comunque un'interazione
 WCA e alcune coppie di atomi un'interazione armonica o fene. Nel calcolo della loss nel training
 dovremmo includere queste interazioni come spiegato prima.
