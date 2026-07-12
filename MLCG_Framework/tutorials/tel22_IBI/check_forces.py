import numpy as np

# We'll just run the _tmp_ibi_md.py but intercept the run
with open("_tmp_ibi_md.py", "r") as f:
    code = f.read()

# Replace the run block with our own force inspection
code = code.replace("system.integrator.run(1000)", "")
code = code.replace("system.force_cap = 2000.0", "system.force_cap = 0.0")

# Append our diagnostic code
diagnostic_code = """
system.integrator.run(0)
forces = system.part.all().f
for i, f in enumerate(forces):
    mag = np.linalg.norm(f)
    if mag > 500:
        print(f"Particle {i} has HUGE force: {mag:.2f}  vector: {f}")

f164 = system.part.by_id(164).f
f165 = system.part.by_id(165).f
print(f"Force on 164: {np.linalg.norm(f164):.2f}")
print(f"Force on 165: {np.linalg.norm(f165):.2f}")
"""

with open("check_forces_md.py", "w") as f:
    f.write(code + diagnostic_code)
