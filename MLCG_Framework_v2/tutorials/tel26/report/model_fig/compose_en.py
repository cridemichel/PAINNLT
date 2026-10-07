import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
plt.rcParams['font.family']='DejaVu Sans'
def crop(p, pad=20):
    im=np.asarray(Image.open(p).convert('RGB')); m=(im<245).any(2)
    ys,xs=np.where(m); return im[max(ys.min()-pad,0):ys.max()+pad, max(xs.min()-pad,0):xs.max()+pad]
P=[crop('panel_side.png'),crop('panel_top.png'),crop('panel_ml.png')]
fig=plt.figure(figsize=(12,6.6))
gs=fig.add_gridspec(2,3,height_ratios=[5.2,1.25],hspace=0.0,wspace=0.04)
T=["a   mapping and priors, side view","b   tetrad 1, viewed along the axis","c   PaiNN graph, $r_c$ = 1.26 nm"]
axs=[]
for k in range(3):
    ax=fig.add_subplot(gs[0,k]); ax.imshow(P[k]); ax.axis('off'); ax.set_anchor('S'); axs.append(ax)
ax=fig.add_subplot(gs[1,:]); ax.axis('off')
COL={'S (sugar-phosphate, G)':'#7f7f7f','A (one site)':'#b5895a','T (one site)':'#46b3c2','B1 (N9, C4)':'#1f3b73','B2 (N3, C2, N2)':'#3366cc',
     'B3 (N1, C6, O6)':'#6f93e6','B4 (C5, N7)':'#8e6fd0','B5 (C8)':'#c2a8f2'}
h1=[Line2D([0],[0],marker='o',ls='',ms=10,mfc=c,mec='k',mew=0.6,label=l) for l,c in COL.items()]
CC=[('harmonic backbone bond / G rigid body','#404040','-'),('Morse B3–B3 within a tetrad (K$_4$, 18)','#d62728','--'),
    ('Morse Hoogsteen N2–N7, B2–B4 (12)','#ff7f0e','--'),('Morse stacking B5–B5 (8)','#2ca02c','--'),
    ('Morse loop caps (16)','#d94fb0','--'),('PaiNN graph edge','#c99a06','-')]
h2=[Line2D([0],[0],color=c,ls=s,lw=2.4 if s=='--' else 2.0,label=l) for l,c,s in CC]
l1=ax.legend(handles=h1,loc='upper left',ncol=4,fontsize=9,frameon=False,title='CG sites (8 types)',title_fontsize=9.5,alignment='left',bbox_to_anchor=(0.0,1.05))
ax.add_artist(l1)
ax.legend(handles=h2,loc='upper left',ncol=3,fontsize=9,frameon=False,title='terms',title_fontsize=9.5,alignment='left',bbox_to_anchor=(0.0,0.42))
fig.canvas.draw()
for k,ax in enumerate(axs):
    bb=ax.get_position(original=False)
    ytop=max(a.get_position(original=False).y1 for a in axs)
    fig.text(bb.x0+0.01, ytop+0.01, T[k], fontsize=11, ha='left', va='bottom')
fig.savefig('fig_model_en.png',dpi=200,bbox_inches='tight',facecolor='white')
print('ok')
