import numpy as np

# A trick to get ESPResSo to load the system is to just import run_cg_md logic
# but we can't because it runs integration.

with open('../../simulation/run_cg_md.py', 'r') as file:
    content = file.read()

# Remove the integration loop and replace with temperature calculation
content = content.split('if args.model:')[0]

content += """
chk = np.load(args.checkpoint)
pos = chk["pos"]
vel = chk["v"]
omega = chk.get("omega", None)

for i in range(len(system.part)):
    p = system.part.by_id(i)
    if not p.is_virtual:
        p.v = vel[i]
        if omega is not None:
            p.omega_body = omega[i]

# Compute exactly
e_kin = 0.0
dofs = 0
for p in system.part:
    if p.mass > 1e-4:
        v_sq = sum(v**2 for v in p.v)
        e_kin += 0.5 * p.mass * v_sq
        dofs += 3
        
        w_sq = sum(w**2 for w in p.omega_body)
        e_kin += 0.5 * sum(I * w**2 for I, w in zip(p.rinertia, p.omega_body))
        dofs += 3

print("Total E_kin:", e_kin)
print("Total DOFs:", dofs)
print("kT =", (2 * e_kin) / dofs)
print("T =", (2 * e_kin) / dofs / 0.00831446)
"""

with open('temp_run.py', 'w') as file:
    file.write(content)

