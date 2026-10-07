import pymol
pymol.finish_launching(["pymol","-cq"])
import json, sys, numpy as np
from pymol import cmd
sites=json.load(open('cg_sites.json'))
pri=json.load(open('cg_priors.lp2.json'))
pos={(s[0],s[2]):np.array(s[4])*10 for s in sites}
GS=['CG_DG_S','CG_DG_B1','CG_DG_B2','CG_DG_B3','CG_DG_B4','CG_DG_B5']
seq="TTAGGGTTAGGGTTAGGGTT"+"TTAGGG"[:0]+"AGGGTT"
seq="TTAGGGTTAGGGTTAGGGTTAGGGTT"
def sname(mol,site):
    b=seq[mol]
    return GS[site] if b=='G' else ('CG_DA' if b=='A' else 'CG_DT')
TET=[[4,12,16,22],[5,11,17,23],[6,10,18,24]]
# frame: y = stacking axis, origin at core center
cen=[np.mean([pos[(r,'CG_DG_B3')] for r in t],0) for t in TET]
y=cen[0]-cen[2]; y/=np.linalg.norm(y)
# x: from tetrad center towards G4 (tetrad 1) projected
v=pos[(4,'CG_DG_B3')]-cen[0]; x=v-np.dot(v,y)*y; x/=np.linalg.norm(x)
z=np.cross(x,y)
Rm=np.array([x,y,z]); O=np.mean(cen,0)
ROT=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
def tr(p): return Rm@(p-O)
# write rotated CG pdb and AA pdb
short={'CG_DA':'A','CG_DT':'T','CG_DG_S':'S','CG_DG_B1':'B1','CG_DG_B2':'B2','CG_DG_B3':'B3','CG_DG_B4':'B4','CG_DG_B5':'B5'}
with open('cg_rot.pdb','w') as f:
    for i,s in enumerate(sites):
        p=tr(np.array(s[4])*10)
        f.write("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00%6.2f           C\n"%(i+1,short[s[2]],s[1],s[0],*p,s[3]))
with open('aa_native.pdb') as fi, open('aa_rot.pdb','w') as fo:
    for l in fi:
        if l.startswith('ATOM'):
            p=tr(np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])]))
            l=l[:30]+"%8.3f%8.3f%8.3f"%tuple(p)+l[54:]
        fo.write(l)
panel=sys.argv[1]
cmd.reinitialize()
cmd.bg_color('white')
cmd.set('ray_opaque_background',1)
cmd.set('antialias',2); cmd.set('ray_trace_mode',1); cmd.set('ray_trace_gain',0.08)
cmd.set('ambient',0.45); cmd.set('specular',0.2); cmd.set('depth_cue',0); cmd.set('ray_shadows',0)
cmd.set('orthoscopic',1)
cmd.load('aa_rot.pdb','aa'); cmd.load('cg_rot.pdb','cg')
cmd.hide('everything')
COL={'S':'0x7f7f7f','A':'0xb5895a','T':'0x46b3c2','B1':'0x1f3b73','B2':'0x3366cc','B3':'0x6f93e6','B4':'0x8e6fd0','B5':'0xc2a8f2'}
for k,c in COL.items(): cmd.color(c,f'cg and name {k}')
cmd.show('spheres','cg')
cmd.set('sphere_scale',0.42,'cg'); cmd.set('sphere_scale',0.62,'cg and name S+A+T')
# AA context
cmd.set('cartoon_ring_mode',3); cmd.set('cartoon_transparency',0.6); cmd.set('cartoon_color','0xb0b0b0')
cmd.show('cartoon','aa'); cmd.set('cartoon_ring_transparency',0.75); cmd.set('cartoon_tube_radius',0.4)
cmd.set('cartoon_ring_color','0xb8b8b8')
def sel(r,sn): return f"cg and resi {r} and name {short[sn]}"
def bond(a,b,obj,col,rad=0.22):
    cmd.bond(a,b)
# rigid guanine skeleton and backbone chain as sticks
cmd.set('stick_radius',0.25)
for r in range(1,27):
    if seq[r-1]=='G':
        for a,b in [('S','B1'),('B1','B2'),('B2','B3'),('B3','B4'),('B4','B5'),('B5','B1')]:
            cmd.bond(f'cg and resi {r} and name {a}',f'cg and resi {r} and name {b}')
for e in pri['bonds']:
    if e['type']=='harmonic' and e['mol_i']<26 and e['mol_j']<26:
        cmd.bond(sel(e['mol_i']+1,sname(e['mol_i'],e['site_i'])),sel(e['mol_j']+1,sname(e['mol_j'],e['site_j'])))
cmd.show('sticks','cg'); cmd.set('stick_color','0x404040','cg'); cmd.set('stick_radius',0.2,'cg')
CC={'tet':'0xd62728','hoog':'0xff7f0e','stack':'0x2ca02c','loop':'0xd94fb0'}
groups={k:[] for k in CC}
for e in pri['bonds']:
    if e['type']!='morse' or e['mol_i']>=26: continue
    role=e.get('role'); g={'stacking':'stack','hoogsteen_n2n7':'hoog','loop':'loop'}.get(role,'tet')
    groups[g].append(e)
show=sys.argv[3].split(',') if len(sys.argv)>3 else list(CC)
for g,es in groups.items():
    if g not in show: continue
    for i,e in enumerate(es):
        cmd.distance(f'{g}_{i}',sel(e['mol_i']+1,sname(e['mol_i'],e['site_i'])),sel(e['mol_j']+1,sname(e['mol_j'],e['site_j'])))
    cmd.group(g,f'{g}_*')
    cmd.set('dash_color',CC[g],g)
cmd.hide('labels'); cmd.set('dash_radius',0.13); cmd.set('dash_gap',0.35); cmd.set('dash_length',0.55)
cmd.set('dash_round_ends',1)
if panel=='ml':
    c=sel(16,'CG_DG_B3')
    cmd.pseudoatom('cut', selection=c, vdw=12.616)
    cmd.show('spheres','cut'); cmd.set('sphere_transparency',0.9,'cut'); cmd.color('0xf3c623','cut'); cmd.set('sphere_scale',1.0,'cut')
    cmd.select('nbr', f'cg within 12.616 of ({c})')
    n=0
    for idx in cmd.identify('nbr and not ('+c+')'):
        cmd.distance(f'g_{n}', c, f'cg and id {idx}'); n+=1
    cmd.group('graph','g_*'); cmd.set('dash_color','0xc99a06','graph'); cmd.set('dash_gap',0.0,'graph'); cmd.set('dash_radius',0.07,'graph')
    cmd.set('sphere_scale',0.75, c); cmd.color('0xf3c623', c)
    print('vicini', n)
    for g in CC: 
        try: cmd.disable(g)
        except: pass
    cmd.hide('labels')
cmd.reset()
if panel=='top':
    cmd.turn('x',-90)
    cmd.hide('everything','aa')
    cmd.hide('everything',"cg and not resi 4+12+16+22")
    cmd.show('cartoon','aa and resi 4+12+16+22')
    for g in ('stack','loop'):
        cmd.disable(g)
    for i,e in enumerate(groups['tet']+groups['hoog']):
        pass
cmd.turn('y',ROT)
cmd.zoom('cg' if panel!='top' else 'cg and resi 4+12+16+22', 2.0 if panel!='ml' else 4.0)
if panel=='top':
    # keep only contacts inside tetrad 1
    for g in ('tet','hoog'):
        for i,e in enumerate(groups[g]):
            if not (e['mol_i']+1 in TET[0] and e['mol_j']+1 in TET[0]): cmd.disable(f'{g}_{i}')
cmd.hide('labels')
cmd.ray(1600,1600 if panel!='top' else 1400)
cmd.png(f'panel_{panel}.png',dpi=300)
print('ok',panel)
