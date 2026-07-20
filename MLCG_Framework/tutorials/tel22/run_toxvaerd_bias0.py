import subprocess
import numpy as np
import os

dts = [0.004, 0.006, 0.008, 0.01]
total_time = 0.5 # 0.5 ps

std_devs_toxvaerd_bias0 = []

for dt in dts:
    steps = int(total_time / dt)
    print(f"\n--- Running NVE for dt = {dt} ps, steps = {steps} ---")
    
    # Run the MD simulation with use_bias=false (omitting --use_bias) and apply_envelope
    cmd = [
        "../../../espresso/build/pypresso", "../../simulation/run_cg_md.py",
        "--model", "tel22_model_fixed.pt",
        "--config", "tel22_training_config.json",
        "--priors", "cg_priors.json",
        "--rb_info", "rigid_bodies_info.json",
        "--dataset", "tel22_dataset.bin",
        "--checkpoint", "equilibrated.npz",
        "--nve",
        "--dt", str(dt),
        "--steps", str(steps),
        "--device", "cpu",
        "--apply_envelope"
    ]
    
    # Capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output to extract E_tot from energy.csv
    energies = []
    if os.path.exists("energy.csv"):
        with open("energy.csv", "r") as f:
            lines = f.readlines()[1:] # skip header
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    energies.append(float(parts[1]))
    
    if len(energies) > 0:
        std_e = np.std(energies)
        std_devs_toxvaerd_bias0.append(std_e)
        print(f"-> Std(E) = {std_e:.5e} kJ/mol")
    else:
        print("-> Error: could not parse energies!")
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        std_devs_toxvaerd_bias0.append(np.nan)

print("Final std_devs_toxvaerd_bias0:", std_devs_toxvaerd_bias0)
