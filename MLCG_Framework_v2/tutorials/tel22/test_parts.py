with open('../../simulation/run_cg_md.py', 'r') as f:
    content = f.read()

content = content.split("if args.checkpoint:")[0]
content += """
for i in range(10):
    p = system.part.by_id(i)
    print(f"ID: {i} | Type: {p.type} | Virtual: {p.is_virtual} | Mass: {p.mass} | Rot: {p.rotation}")
"""
with open('temp_parts.py', 'w') as f:
    f.write(content)
