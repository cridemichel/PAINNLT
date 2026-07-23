FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Evita prompt interattivi di apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Aggiornamento e installazione dipendenze essenziali
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    # Dipendenze tipiche di ESPResSo \
    libopenmpi-dev \
    openmpi-bin \
    libfftw3-dev \
    libboost-all-dev \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

# Installazione moduli Python essenziali (incluso MDAnalysis per il parsing di Gromacs)
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    MDAnalysis \
    h5py \
    Cython

# Variabili di ambiente utili per LibTorch / ESPResSo
ENV Torch_DIR=/opt/conda/lib/python3.10/site-packages/torch/share/cmake/Torch
ENV CPATH=/opt/conda/include:$CPATH
ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# Configura una directory di lavoro
WORKDIR /app

# NOTA: Per compilare su Leonardo con Apptainer, è consigliabile montare la cartella PAINNLT 
# come volume (tramite --bind) piuttosto che copiarla nell'immagine. In questo modo i binari 
# generati restano persistenti sul filesystem del supercomputer.
# Tuttavia, per portabilità, definiamo un entrypoint base.
CMD ["/bin/bash"]
