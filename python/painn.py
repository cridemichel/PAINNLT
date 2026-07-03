import torch
import pytorch_lightning as pl
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.representation import PaiNN
from schnetpack.nn.cutoff import CosineCutoff
from torch.optim import AdamW

# ==============================================================================
# 1. DEFINIZIONE DEL TASK LIGHTNING (TRAINING LOOP CUSTOM)
# ==============================================================================
class CGMDTask(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        
        # Pesi della loss per bilanciare grandezze con unità diverse
        self.force_weight = 1.0
        self.torque_weight = 0.5 

    def forward(self, batch):
        # Il modello predice l'energia
        # IMPORTANTE: abilitiamo il tracking dei gradienti sulle posizioni
        batch['_positions'].requires_grad_(True)
        results = self.model(batch)
        
        # Calcolo delle Forze: derivata dell'energia rispetto alle posizioni
        # create_graph=True è essenziale per il double backward (calcolo della derivata seconda)
        forces = -torch.autograd.grad(
            outputs=results['energy'],
            inputs=batch['_positions'],
            grad_outputs=torch.ones_like(results['energy']),
            create_graph=True,
            retain_graph=True
        )[0]
        
        results['predicted_forces'] = forces
        
        # [NOTA: Qui potresti aggiungere il calcolo vettoriale del momento torcente predetto
        # usando le forze calcolate (results['predicted_forces']) e le posizioni relative]
        
        return results

    def training_step(self, batch, batch_idx):
        results = self(batch)
        
        # Calcolo della Loss (Force Matching)
        loss_forces = self.loss_fn(results['predicted_forces'], batch['forces'])
        
        # Esempio di come sommeresti la loss per il momento torcente:
        # loss_torques = self.loss_fn(calcola_torque(results), batch['torques'])
        # total_loss = self.force_weight * loss_forces + self.torque_weight * loss_torques
        
        total_loss = self.force_weight * loss_forces
        
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        # Ottimizzatore AdamW
        return AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)


# ==============================================================================
# 2. FLUSSO PRINCIPALE (DATASET, MODELLO, TRAINING ED ESPORTAZIONE)
# ==============================================================================
if __name__ == '__main__':
    
    # --- SETUP DEL DATASET ---
    # Inserisci qui il percorso al tuo database All-Atom / CG
    dataset_path = './dataset_coarse_grained.db'
    cutoff_radius = 5.0  # Angstrom
    
    # Trasformazioni: calcolo neighbor list dinamica e mixed precision
    transforms = [
        trn.ASENeighborList(cutoff=cutoff_radius),
        trn.CastTo32() 
    ]
    
    data_module = spk.data.AtomsDataModule(
        datapath=dataset_path,
        batch_size=32,
        num_train=1000, # Adatta questi numeri al tuo dataset
        num_val=200,
        transforms=transforms,
        num_workers=4,
        pin_memory=True
    )
    data_module.prepare_data()
    data_module.setup()
    
    # --- DEFINIZIONE DEL MODELLO PAINN ---
    print("Inizializzazione del modello PaiNN...")
    painn_representation = PaiNN(
        n_atom_basis=128,
        n_interactions=3,
        shared_interactions=False,
        shared_filter=False,
        cutoff_fn=CosineCutoff(cutoff_radius) # Utilizzo del Cosine Cutoff
    )
    
    # Modulo di output con standardizzazione dell'energia (Scale and Shift)
    output_module = spk.atomistic.Atomwise(
        n_in=128,
        output_key='energy',
        aggregation_mode='sum',
        energy_standardization=trn.Standardize(
            mean=torch.tensor(0.0), 
            stddev=torch.tensor(1.0)
        )
    )
    
    # Assemblaggio finale del Modello Potenziale
    model = spk.model.NeuralNetworkPotential(
        representation=painn_representation,
        input_modules=[spk.atomistic.PairwiseDistances()], 
        output_modules=[output_module],
        postprocessors=[trn.AddOffsets(output_key='energy')] 
    )
    
    # --- TRAINING CON PYTORCH LIGHTNING ---
    lightning_task = CGMDTask(model)
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed',      # Mixed precision per ottimizzare VRAM e velocità
        gradient_clip_val=1.0,     # Gradient clipping contro le repulsioni estreme (NaN)
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='train_loss', save_top_k=1)
        ]
    )
    
    print("Inizio addestramento...")
    trainer.fit(lightning_task, datamodule=data_module)
    
    # --- ESPORTAZIONE DEL MODELLO IN TORCHSCRIPT (PER ESPRESSO) ---
    print("\nTraining completato. Esportazione del modello in TorchScript...")
    lightning_task.eval()
    
    # Estraiamo un batch fittizio dal DataLoader di training
    with torch.no_grad():
        example_batch = next(iter(data_module.train_dataloader()))
        
        # Manteniamo solo le chiavi (features) che ESPResSo fornirà in inferenza
        infer_batch = {
            '_positions': example_batch['_positions'],
            '_atomic_numbers': example_batch['_atomic_numbers'],
            '_cell': example_batch['_cell'],
            '_pbc': example_batch['_pbc']
        }
        
        # Tracciamo il modello per compilarlo (congela l'architettura in C++)
        traced_model = torch.jit.trace(lightning_task.model, (infer_batch,))
        
        # Salviamo il file da importare in libtorch/ESPResSo
        output_filename = "painn_cgmd_model.pt"
        traced_model.save(output_filename)
        print(f"Modello '{output_filename}' salvato con successo. Pronto per il C++!")
