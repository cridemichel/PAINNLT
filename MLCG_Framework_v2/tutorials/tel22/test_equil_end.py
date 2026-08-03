import numpy as np
with open('../../simulation/equilibrate.py', 'r') as f:
    content = f.read()

content = content.split("np.savez(")[0]
content += """
for i in range(10):
    p = system.part.by_id(i)
    print(f"ID: {i} | Virtual: {p.is_virtual} | Mass: {p.mass} | v: {p.v} | pos: {p.pos}")
"""

with open('temp_equil.py', 'w') as f:
    f.write(content)
