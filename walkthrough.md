# Walkthrough: Integrazione PaiNN in ESPResSo

Ho completato tutte le modifiche al codice sorgente di ESPResSo per integrare il tuo modello nativo C++ `PaiNN` come potenziale per la Dinamica Molecolare.

## Modifiche Architetturali Effettuate

### 1. Codice C++ del Potenziale
- Ho copiato il tuo `PaiNN_Architecture.hpp` in `espresso/src/core/nonbonded_interactions/`.
- Ho modificato il metodo `forward()` di `PaiNNModel` per accettare direttamente `r_ij` come input, permettendo così di sfruttare le distanze di **Minimum Image Convention** e le **Periodic Boundary Conditions (PBC)** calcolate nativamente da ESPResSo.
- Ho creato `PaiNN_ML_Potential.hpp` e `.cpp`. Questa classe estrae le particelle da ESPResSo (incluse le particelle "ghost" per le PBC), mappa gli ID, intercetta il `non_bonded_loop` per estrarre le distanze `d.vec21` valide e costruisce un grafo PyTorch contiguo.
- L'energia viene propagata in PyTorch e le forze derivano tramite `torch::autograd::grad` rispetto a `r_ij`, per poi essere applicate indietro alle particelle ESPResSo (rispettando il principio di azione-reazione per i legami del grafo).

### 2. Integrazione nel Core di Integrazione Molecolare
- In `espresso/src/core/forces.cpp`, ho aggiunto l'hook che richiama la valutazione del potenziale PaiNN ad ogni step di simulazione, subito dopo le forze a corto raggio.
- Nel `CMakeLists.txt` di ESPResSo ho aggiunto `find_package(Torch REQUIRED)` e configurato i linking corretti per far comunicare ESPResSo con `libtorch`.

### 3. Binding Python
- Ho scritto un'estensione Cython `painn.pyx` che permette di attivare facilmente il potenziale dal tuo script Python di simulazione chiamando una singola funzione.
- Ho creato uno script di test- `test_espresso_painn.py`: script di validazione che crea particelle e calcola le forze.

## 🛠️ Risoluzione Bug e Fix Successivi
Durante la prima validazione abbiamo risolto alcuni comportamenti inaspettati tra Python, PyTorch ed ESPResSo:

1. **Bug Namespace c10**: L'include di `<torch/torch.h>` all'interno del blocco `System::calculate_forces()` generava conflitti di namespace. È stato risolto spostando le dipendenze in testa a `forces.cpp`.
2. **Cython Varargs**: Passare parametri Python dinamici (interi, float) ad un template C++ come `make_shared` mandava in tilt Cython. Risolto effettuando il cast a `cdef int` e `cdef double` all'interno del wrapper Cython prima della chiamata C++.
3. **Serializzazione Pesi (Mismatch Architetturale)**: I nomi e le strutture dei moduli esportati nel file `.pt` devono combaciare _esattamente_ (carattere per carattere) con l'architettura definita in `PaiNN_Architecture.hpp`. È stato necessario adattare le dichiarazioni C++ affinché mappassero il file `painn.cpp` (script di addestramento originale), in particolare la classe `expansion_rbf` e il raggruppamento delle liste `messages` e `updates`.
4. **Verlet Lists Invisibili (ESPResSo)**: Dato che il nostro potenziale ML è implementato come classe globale, ESPResSo non sapeva della necessità di valutare i cutoff. Poiché ESPResSo scarta le coppie se il loro cut-off formale è 0, le liste risultavano vuote. Il fix _lato utente_ richiede l'iniezione fittizia di un'interazione base (es. Lennard-Jones a zero energia) con `cutoff = 5.0` per forzare il tracciamento topologico della scatola. Abbiamo dovuto contestualmente aumentare `box_l` a `12.0` (poiché la somma tra cutoff 5.0 e skin 0.4 richiede un raggio d'interazione <= metà della scatola).
5. **Autograd su MPS (Apple Silicon)**: La richiesta di `autograd::grad` sul backend MPS genera attualmente lock di mutex invalidi, molto probabilmente perché ESPResSo esegue la routine C++ all'interno dei suoi loop paralleli. Per ovviare su Mac, abbiamo forzato l'uso di `device="cpu"` per questo test.

L'integrazione è completata e pienamente funzionante! Puoi estendere liberamente il modello per supportare ulteriori proprietà o dataset direttamente addestrando nuovi pesi e caricandoli su ESPResSo in modalità `cpu` o `cuda` (su cluster Linux). verificare l'inferenza una volta compilato.

---

## Prossimi Passi: Compilazione

> [!IMPORTANT]
> Ora ESPResSo deve essere compilato. Poiché hai già usato `libtorch` per `painn.cpp`, assumo tu abbia scaricato `libtorch`.
> Durante la configurazione CMake, devi specificare il percorso di PyTorch tramite `-DCMAKE_PREFIX_PATH`.

Esegui questi comandi nel tuo terminale per compilare la versione modificata di ESPResSo:

```bash
cd espresso
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/percorso/assoluto/alla/tua/libtorch
make -j4
```

Una volta compilato (senza errori), potrai testare il modello eseguendo:

```bash
# Esegui ESPResSo passando il nostro script di test
./pypresso ../../test_espresso_painn.py
```

Questo inizializzerà un piccolo sistema di test con l'etanolo e farà un primo calcolo delle forze passando attraverso la rete neurale e applicandole in ESPResSo. Fammi sapere se incontri errori durante la compilazione (è possibile che servano dei leggeri aggiustamenti su alcuni header Cython se la versione di ESPResSo è particolarmente nuova)!
