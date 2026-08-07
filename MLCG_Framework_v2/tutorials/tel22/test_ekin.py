import subprocess
import sys

# Replace the loop in run_cg_md.py to print every step
with open("../../simulation/run_cg_md.py", "r") as f:
    code = f.read()

code = code.replace("for step in range(1, args.steps + 1):", "for step in range(0, 5):")
code = code.replace("if step % 10 == 0:", "if True:")

with open("test_md.py", "w") as f:
    f.write(code)

