ENG
===

TODO LIST

1) Energy normalization (add energy shift and scale)

SchNetPack addresses the issue of energy magnitudes with two built-in mechanisms that correspond to our "Scale and Shift":

schnetpack.atomistic.Atomref: This module appends the reference energy of the individual isolated atoms (the baseline shift) to the end of the network.

schnetpack.transform.Standardize: During preprocessing, SchNetPack statistically computes the mean and standard deviation of the remaining energies or forces and standardizes the output. This ensures that the neural network always works with numbers close to zero, letting the framework re-multiply and re-add the real values only at the moment of the final output.

2) Implement cosine cutoff

If you look in the schnetpack/nn/cutoff.py folder, you will find an entire class called CosineCutoff.
When you declare the PaiNN or SchNet model in SchNetPack, the cutoff_network parameter is initialized by default with this exact class. They use exactly the scaled cosine function that I suggested to ensure that the spatial derivatives smoothly drop to zero at the edge of the Neighbor List, preventing discontinuous jumps in the forces.

3) In version 2.0, SchNetPack delegated the entire training loop to PyTorch Lightning.

If you check their training configuration files (managed via the Hydra system in the configs/trainer folder), you will find the gradient_clip_val parameter. It is common practice in their tutorials to set this parameter right around 0.5 or 1.0. As we discussed, since the forces are the derivative of the energy, without this "leash" on the gradient, a single unlucky short-range repulsion in the batch would permanently ruin the weights of the AdamW optimizer.

4) Mixed precision

GROMACS TO BIN

1) Given a group of atoms, a script must be prepared that replaces this group of atoms with a cluster of virtual sites and a real particle, or with a single real particle.
Each virtual site must be associated with a type (we can use the atomic number Z) and a mol_id (the molecule it belongs to) (I need to discuss this with Laura).
If there are no virtual sites, the single real particle will still have its own mol_id.

2) In the construction of the interactions in PaiNN, atoms with the same mol_id must not interact.

3) In ESPResSo, virtual sites do not interact with each other, so nothing will need to be done; just import the model and use it as I already do now.

4) Most likely, priors will also need to be included in the loss function to prevent overlaps (using WCA) or harmonic (or FENE) potentials for bonded atoms.
The forces in the loss will be those predicted by the network + those derived from the priors.
Everything will be implemented as follows: in ESPResSo, the various virtual sites will still have a WCA interaction, and some pairs of atoms will have a harmonic or FENE interaction. In the calculation of the loss during training, we should include these interactions as explained earlier.


ITA
====

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
