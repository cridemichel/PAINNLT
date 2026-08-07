with open("debug_md.py", "r") as f:
    lines = f.readlines()

with open("debug_md.py", "w") as f:
    for line in lines:
        if "for step in range(" in line:
            f.write("for step in range(0, 11):\n")
        elif "system.integrator.run(1)" in line:
            f.write("        if step > 0:\n            system.integrator.run(1)\n")
        elif "if step % 10 == 0:" in line:
            f.write("        if True:\n")
        else:
            f.write(line)
