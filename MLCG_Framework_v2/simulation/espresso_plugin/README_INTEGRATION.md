# Guida all'Integrazione di PaiNN in ESPResSo

Questa cartella contiene tutti i file sorgenti custom che permettono di far funzionare il tuo modello `PaiNN` (scritto in C++ con `libtorch`) come potenziale nativo in ESPResSo.

L'idea è che tu possa prendere una versione "pulita" dei sorgenti di ESPResSo e applicare questa patch per avere il tuo potenziale CG (Coarse-Grained) ad alte prestazioni.

## File Inclusi
* `PaiNN_Architecture.hpp`: Il tuo modello originale, con l'aggiunta di un metodo `forward_with_rij` per accettare direttamente i vettori distanza in modo da rispettare nativamente le Minimum Image Convention (PBC) calcolate da ESPResSo.
* `PaiNN_ML_Potential.hpp` e `.cpp`: Le classi "ponte" che agganciano il modello a ESPResSo. Raccolgono le particelle, valutano il vicinato, chiamano il forward per l'energia e restituiscono i gradienti (forze) sulle particelle.
* `painn.pyx`: L'interfaccia Cython per poter instanziare e chiamare il potenziale comodamente da Python, come ogni altro potenziale in ESPResSo.

---

## Passaggi per applicare la patch ai sorgenti di ESPResSo

Assumendo di aver clonato ESPResSo e di trovarti nella cartella root del progetto (`espresso/`), ecco i passi esatti da seguire:

### 1. Copia dei file
Copia i file da questa cartella (`espresso_plugin`) nei sorgenti di ESPResSo:
```bash
# Copia gli header e il cpp nel core
cp espresso_plugin/PaiNN_Architecture.hpp espresso/src/core/nonbonded_interactions/
cp espresso_plugin/PaiNN_ML_Potential.hpp espresso/src/core/nonbonded_interactions/
cp espresso_plugin/PaiNN_ML_Potential.cpp espresso/src/core/nonbonded_interactions/

# Copia l'interfaccia python (Cython)
cp espresso_plugin/painn.pyx espresso/src/python/espressomd/
```

### 2. Modifica di `espresso/src/core/CMakeLists.txt`
Apri `espresso/src/core/CMakeLists.txt` e fai due modifiche:
1. Subito prima di `target_link_libraries(espresso_core ...)` (solitamente intorno alla riga 77), aggiungi:
   ```cmake
   find_package(Torch REQUIRED)
   ```
2. Dentro il blocco `target_link_libraries(espresso_core PUBLIC ...)` aggiungi `"${TORCH_LIBRARIES}"`.
3. Sotto al blocco `target_link_libraries`, aggiungi la flag del compilatore:
   ```cmake
   target_compile_definitions(espresso_core PUBLIC ESPRESSO_PAINN)
   ```

### 3. Modifica di `espresso/src/core/nonbonded_interactions/CMakeLists.txt`
Apri `espresso/src/core/nonbonded_interactions/CMakeLists.txt` e aggiungi il file `PaiNN_ML_Potential.cpp` alla lista dei file sorgenti (dentro `target_sources(espresso_core PRIVATE ...)`).
```cmake
          ${CMAKE_CURRENT_SOURCE_DIR}/PaiNN_ML_Potential.cpp
```

### 4. Modifica di `espresso/src/core/forces.cpp`
Apri `espresso/src/core/forces.cpp`. Fai due modifiche:

**A.** In cima al file, assieme agli altri `#include`, aggiungi:
```cpp
#ifdef ESPRESSO_PAINN
#include "nonbonded_interactions/PaiNN_ML_Potential.hpp"
#endif
```

**B.** Cerca la funzione `System::calculate_forces()`.
Verso la fine della funzione, **subito dopo** la fine del blocco `cabana_short_range` (che calcola le forze accoppiate corte), aggiungi il richiamo al nostro potenziale globale.
Cerca queste righe (intorno alla riga 370-375):
```cpp
#ifdef ESPRESSO_CALIPER
  CALI_MARK_END("cabana_short_range");
#endif
```
E aggiungi **esattamente sotto** questo blocco:
```cpp
#ifdef ESPRESSO_PAINN
  if (global_painn_potential) {
    global_painn_potential->calculate_forces(*cell_structure, verlet_criterion);
  }
#endif
```

*(Nota: in ESPResSo i file Cython nella cartella `espressomd/` vengono "pescati" in automatico dal CMake tramite un glob `*.pyx`, per cui non serve aggiungere manualmente `painn.pyx` al CMake Python).*

---

## Compilazione
Una volta patchati i file, devi compilare ESPResSo indicando il percorso della tua `libtorch`.
Entra in `espresso`, crea la cartella build e lancia CMake:
```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/percorso/assoluto/alla/tua/libtorch
make -j4
```

## Esecuzione e Test
In questa cartella `espresso_plugin` troverai due script Python pronti all'uso per testare e validare la tua implementazione:

### 5.1 Test di Base (`test_espresso_painn.py`)
Lancia lo script base tramite l'eseguibile Python di ESPResSo:
```bash
./espresso/build/pypresso espresso_plugin/test_espresso_painn.py
```
Questo script posiziona 9 atomi a caso, attiva il potenziale C++ PaiNN, ed esegue 0 step per farsi restituire le forze predette.

### 5.2 Validazione Quantitativa (`validate_single_point.py`)
Usa questo script per verificare che le forze predette dentro ESPResSo combacino con le forze *reali* previste dal dataset originale (utile per calcolare il MAE). 
```bash
./espresso/build/pypresso espresso_plugin/validate_single_point.py
```
*(Nota: assicurati che il file `ethanol_val.bin` e i pesi `.pt` si trovino nella directory da cui lanci lo script).*

---

**ATTENZIONE - Verlet Lists (Importante!)**
ESPResSo calcola le liste dei vicini (Verlet Lists) *solo se* esistono interazioni `non_bonded` registrate. Il nostro potenziale è globale e in C++, quindi ESPResSo "non sa" che stiamo usando un cutoff di 5.0. 
Affinché ESPResSo passi i vicini corretti al modello PaiNN C++, **devi sempre definire un'interazione fittizia (dummy)** con cutoff pari o superiore a quello del tuo modello nello script Python:
```python
# Crea un'interazione a energia zero ma con cutoff = 5.0 per forzare ESPResSo a cercare i vicini
for i in range(10): # per ogni tipo di atomo
    for j in range(i, 10):
        system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=0.0, sigma=1.0, cutoff=5.0, shift=0.0)
```
Inoltre, ricorda che la dimensione minima della scatola `system.box_l` deve rispettare il criterio `box_l / 2 > cutoff + skin`. Se usi cutoff = 5.0 e skin = 0.4, il box deve essere almeno 10.9x10.9x10.9.

## Bond Morse analitico (necessario per NVE)

Il framework non rappresenta piu' i prior bonded Morse con `TabulatedDistance`.
La cartella contiene `morse_bond.hpp` e l'installer
`install_analytic_morse_bond.py`, che aggiunge un vero pair bond conservativo
al core, alla ScriptInterface e a `espressomd.interactions`.

La convenzione e' la stessa del preprocessing:

```text
U(r) = D * (1 - exp(-a * (r-r_0)))^2
```

`copy_plugin_files.sh` installa automaticamente anche questo bond. Se ESPResSo
non si trova in `<framework>/espresso`, indicare il source tree esplicitamente:

```bash
ESPRESSO_SRC=/path/to/espresso bash simulation/espresso_plugin/copy_plugin_files.sh
python3 simulation/espresso_plugin/install_analytic_morse_bond.py \
  --espresso-root /path/to/espresso --check
```

L'installer e' idempotente e usa anchor specifici per ESPResSo 5.0.x: se il
layout non corrisponde, termina con errore invece di applicare modifiche
ambigue. Dopo l'installazione ESPResSo deve essere ricompilato.

### Verifica del runtime ricompilato

Dopo la ricompilazione di ESPResSo, esegui lo smoke test con il launcher ricompilato:

```bash
/path/to/espresso/build/pypresso simulation/espresso_plugin/check_analytic_morse_bond.py
```

Il test crea un singolo Morse bond e verifica che energia bonded e forze sulle particelle coincidano con l'espressione analitica.

## Prior bonded conservative spline (IBI fase 2)

I prior IBI di bond e angle possono essere convertiti nella rappresentazione
`conservative_spline` a singola sorgente del framework. Installare e validare
l'estensione ESPResSo dal workflow TEL22 IBI con:

```bash
cd tutorials/tel22_IBI
bash ./20_install_conservative_spline.sh
```

Il comando non e' soltanto un controllo di build: dopo aver verificato che
`ConservativeSplineDistance` e `ConservativeSplineAngle` siano esposti a
Python, esegue `simulation/smoke_conservative_spline_runtime.py` con il
`pypresso` appena ricompilato. Lo smoke test crea spline sintetiche per bond e
angle e verifica che le forze runtime coincidano con il gradiente cartesiano
negativo, calcolato per differenze finite, dell'energia bonded restituita dallo
stesso ESPResSo. Il successivo `22_validate_conservative_spline.sh` esegue il
controllo di parita' corrispondente sulle tabelle IBI convertite reali.
