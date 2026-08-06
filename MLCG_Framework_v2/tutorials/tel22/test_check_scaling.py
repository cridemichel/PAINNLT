import subprocess
import numpy as np

dts = [0.002, 0.004, 0.008, 0.012, 0.020]
total_time = 0.2

for dt in dts:
    steps = int(total_time / dt)
    
    cmd = [
        "../../espresso/build/pypresso", "../../simulation/run_cg_md.py",
        "--model", "tel22_model.pt",
        "--config", "tel22_training_config.json",
        "--priors", "cg_priors.json",
        "--rb_info", "rigid_bodies_info.json",
        "--dataset", "tel22_dataset.bin",
        "--checkpoint", "equilibrated.npz",
        "--nve",
        "--allow_nonconservative_tables",
        "--dt", str(dt),
        "--steps", str(steps),
        "--device", "cpu"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    energies = []
    for line in result.stdout.split('\n'):
        if "[INFO] Step " in line and "E_tot:" in line:
            parts = line.split("|")
            for p in parts:
                if "E_tot:" in p:
                    val = float(p.split("E_tot:")[1].strip())
                    energies.append(val)
                    
    if len(energies) > 0:
        std_e = np.std(energies)
        print(f"dt = {dt:.3f} -> Std(E) = {std_e:.5e}")
