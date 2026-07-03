import csv
import matplotlib.pyplot as plt
import os
import argparse

def plot_training_metrics(csv_path="training_metrics.csv", output_dir="plots"):
    if not os.path.exists(csv_path):
        print(f"Errore: Il file {csv_path} non esiste.")
        return

    epochs = []
    train_loss = []
    val_loss = []
    mae_e = []
    mae_f = []

    # Leggi il CSV con la libreria standard
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                epochs.append(int(row['Epoch']))
                train_loss.append(float(row['TrainLoss']))
                val_loss.append(float(row['ValLoss']))
                mae_e.append(float(row['MaeE']))
                mae_f.append(float(row['MaeF']))
            except (ValueError, KeyError) as e:
                continue # Salta righe malformate

    # Controlla se ci sono dati
    if not epochs:
        print("Il file CSV è vuoto. Attendi che il training salvi almeno un'epoca.")
        return

    # Crea la cartella per i plot se non esiste
    os.makedirs(output_dir, exist_ok=True)

    print(f"File caricato con successo. Trovate {len(epochs)} epoche.")
    
    # 1. Plot della Loss (Train vs Val)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss (Combinata)', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss (Combinata)', color='orange', linewidth=2, linestyle='--')
    plt.yscale('log') # La loss spesso varia di ordini di grandezza
    plt.xlabel('Epoca', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Andamento Loss Combinata (Energia + Forze)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Salvato: {loss_plot_path}")

    # 2. Plot dell'Errore Assoluto Medio (MAE) per Energia e Forze
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mae_e, label='MAE Energia (kcal/mol)', color='green', linewidth=2)
    plt.plot(epochs, mae_f, label='MAE Forze (kcal/mol/Å)', color='red', linewidth=2)
    plt.xlabel('Epoca', fontsize=12)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
    plt.title('Accuratezza PaiNN (Validation Set)', fontsize=14)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    
    # Se il MAE ha valori molto distanti, potrebbe servire la scala logaritmica.
    # Scommentare la riga seguente se necessario:
    # plt.yscale('log') 
    
    mae_plot_path = os.path.join(output_dir, 'mae_plot.png')
    plt.savefig(mae_plot_path, dpi=300, bbox_inches='tight')
    print(f"Salvato: {mae_plot_path}")

    print("Grafici generati con successo! (Chiudi le finestre dei grafici per terminare lo script)")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotta le metriche di training di SchNet.")
    parser.add_argument("--csv", type=str, default="training_metrics.csv", help="Percorso al file CSV")
    parser.add_argument("--out", type=str, default="plots", help="Cartella in cui salvare i grafici")
    args = parser.parse_args()

    plot_training_metrics(args.csv, args.out)
