import csv
import os
import matplotlib.pyplot as plt

def plot_metrics(csv_filepath="training_log.csv"):
    if not os.path.exists(csv_filepath):
        print(f"[ERRORE] Il file {csv_filepath} non esiste. Controlla il percorso.")
        return

    # Inizializziamo le liste per salvare i dati
    epochs = []
    train_loss = []
    val_loss = []
    train_mae_forces = []
    val_mae_forces = []

    # 1. Lettura del file CSV usando il modulo nativo 'csv'
    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            for row in reader:
                # Salta righe vuote
                if not row:
                    continue
                
                # Se la riga contiene testo (es. un header scritto a mano), la saltiamo
                try:
                    epoch = int(row[0])
                    t_loss = float(row[1])
                    v_loss = float(row[2])
                    t_mae_f = float(row[3])
                    v_mae_f = float(row[5]) # La colonna 4 è train_mae_torques, la 5 è val_mae_forces
                except ValueError:
                    # Riga di intestazione (header) saltata correttamente
                    continue
                
                # Salviamo i dati convertiti nelle liste
                epochs.append(epoch)
                train_loss.append(t_loss)
                val_loss.append(v_loss)
                train_mae_forces.append(t_mae_f)
                val_mae_forces.append(v_mae_f)
                
    except Exception as e:
        print(f"[ERRORE] Impossibile leggere il file CSV: {e}")
        return

    if not epochs:
        print("[AVVISO] Nessun dato numerico trovato nel file CSV.")
        return

    print(f"[INFO] Caricate {len(epochs)} epoche da {csv_filepath}")

    # 2. Creazione del grafico con Matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # -----------------------------------------------------------------
    # GRAFICO 1: LOSS
    # -----------------------------------------------------------------
    ax1.plot(epochs, train_loss, label="Train Loss (L1)", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss (L1)", color="#ff7f0e", linestyle="--", linewidth=2)
    ax1.set_title("Andamento della Loss", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoca", fontsize=12)
    ax1.set_ylabel("Loss [kJ/(mol*nm)]", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=11)
    
    # Annotazione dell'ultimo valore di validazione
    ax1.annotate(f'Ultima: {val_loss[-1]:.2f}', 
                 xy=(epochs[-1], val_loss[-1]),
                 xytext=(10, 0), textcoords='offset points', color="#ff7f0e", fontweight='bold')

    # -----------------------------------------------------------------
    # GRAFICO 2: MAE FORZE
    # -----------------------------------------------------------------
    ax2.plot(epochs, train_mae_forces, label="Train MAE Forze", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, val_mae_forces, label="Val MAE Forze", color="#d62728", linestyle="--", linewidth=2)
    ax2.set_title("Mean Absolute Error (MAE) Forze", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoca", fontsize=12)
    ax2.set_ylabel("MAE [kJ/(mol*nm)]", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=11)
    
    # Annotazione dell'ultimo valore del MAE
    ax2.annotate(f'Ultimo: {val_mae_forces[-1]:.2f}', 
                 xy=(epochs[-1], val_mae_forces[-1]),
                 xytext=(10, 0), textcoords='offset points', color="#d62728", fontweight='bold')

    # Ottimizzazione degli spazi
    plt.tight_layout()
    
    # Salva il grafico come immagine
    output_img = "training_curves.png"
    plt.savefig(output_img, dpi=300)
    print(f"[INFO] Grafico salvato con successo in: {output_img}")
    
    # Mostra la finestra a schermo
    plt.show()

if __name__ == "__main__":
    plot_metrics("cg_training_log.csv")
