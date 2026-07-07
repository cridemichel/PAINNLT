# GUIDA RAPIDA: DEPLOY E SOTTOMISSIONE SU LEONARDO (CINECA)

Questa guida riassume i passaggi esatti per trasformare il tuo progetto locale in un container funzionante sul supercomputer Leonardo, utilizzando Docker e Apptainer.

---

## FASE 1: Preparazione sul Laptop

Tutti questi comandi vanno eseguiti sul tuo computer locale, all'interno della cartella del progetto (`PAINNLT`), dove si trova il file `Dockerfile`.

**1. Compila l'immagine Docker locale**
```bash
docker build -t painn_env .
```

**2. Esporta l'immagine in un file compresso (tar)**
*(L'esportazione può richiedere qualche minuto vista la grandezza dell'immagine PyTorch/CUDA)*
```bash
docker save painn_env -o painn_env.tar
```

**3. Trasferisci il file tar e i sorgenti su Leonardo**
*(Sostituisci `<username>` con il tuo vero utente CINECA)*
```bash
# Copia l'immagine Docker
scp painn_env.tar <username>@login.leonardo.cineca.it:/leonardo/home/userexternal/<username>/

# Copia l'intero progetto (escludendo eventuali build o dataset pesanti già presenti)
rsync -avz --exclude 'build' --exclude '*.trr' ./ <username>@login.leonardo.cineca.it:/leonardo/home/userexternal/<username>/PAINNLT/
```

---

## FASE 2: Conversione su Leonardo (Login Node)

Collegati a Leonardo tramite SSH. Questi comandi servono a convertire l'immagine nel formato nativo per supercomputer (`.sif`).

```bash
ssh <username>@login.leonardo.cineca.it

# Entra nella cartella dove hai copiato il tar
cd /leonardo/home/userexternal/<username>/

# Carica il modulo Apptainer
module load apptainer

# Converte l'immagine (Ci metterà un po' a scompattare e convertire)
apptainer build painn_env.sif docker-archive://painn_env.tar
```
*Nota: Ora hai il file `painn_env.sif` pronto all'uso.*

---

## FASE 3: Sottomissione del Job SLURM

Nel tuo progetto c'è già il file `leonardo_submit.slurm`. Prima di inviarlo:

**1. Personalizza lo script SLURM**
Apri `leonardo_submit.slurm` e assicurati di:
- Inserire il tuo account alla riga: `#SBATCH --account=<TUO_ACCOUNT_CINECA>`
- Sostituire `<username>` nei percorsi `IMAGE_PATH` e `WORK_DIR`.
- Scegliere (decommentare) quale comando eseguire (es. Compilazione ESPResSo, Conversione Dataset, o Training PaiNN). 
  *Nota: Il flag `--bind $WORK_DIR:/app` fa in modo che il container veda i file di Leonardo, e i risultati vengano salvati su Leonardo e non persi quando il container si chiude.*

**2. Invia il Job in coda**
```bash
cd /leonardo/home/userexternal/<username>/PAINNLT
sbatch leonardo_submit.slurm
```

**3. Monitora il Job**
Per controllare lo stato del tuo job in coda:
```bash
squeue -u <username>
```
Per leggere l'output del programma (o eventuali errori) in tempo reale mentre gira:
```bash
tail -f slurm-<JOB_ID>.out
tail -f slurm-<JOB_ID>.err
```
