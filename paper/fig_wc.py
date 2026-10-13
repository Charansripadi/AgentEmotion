import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5})
cells=[("Qwen3-1.7B GSM8K",[.497,.527,.589,.544,.495,.448],(.345,.447,.576),.791),
 ("Qwen3-1.7B Social IQa",[.668,.663,.594,.573,.627,.497],(.187,.403,.810),.550),
 ("Qwen3-8B GSM8K",[.204,.326,.449,.294,.206,.143],(.181,.353,.659),.448),
 ("Qwen3-8B Social IQa",[.560,.641,.678,.435,.496,.487],(.429,.566,.654),.798),
 ("Llama GSM8K",[.182,.093,.231,.218,.153,.130],(.127,.170,.347),.570),
 ("Llama Social IQa",[.444,.468,.338,.617,.442,.456],(.057,.368,.540),.442)]
fig,axs=plt.subplots(2,3,figsize=(6.6,3.0),sharex=True)
for ax,(t,ev,(lo,mu,hi),c) in zip(axs.flat,cells):
    ax.hlines(1,lo,hi,color='#3b6ea8',lw=8,alpha=.35,label='neutral rewordings (min to max)')
    ax.plot([mu],[1],'D',color='#3b6ea8',ms=5,label='neutral mean')
    ax.scatter(ev[:-1],[0]*5,s=22,color='#e08a1e',label='emotion framings')
    ax.scatter([ev[-1]],[0],s=30,color='#c8553d',label='carefree')
    ax.axvline(c,color='k',ls='--',lw=0.8)
    ax.set_yticks([0,1]); ax.set_yticklabels(['emotion','neutral']); ax.set_ylim(-.6,1.6); ax.set_title(t,fontsize=8,loc='left')
    ax.set_xlim(0,.9)
    for s in ['top','right']: ax.spines[s].set_visible(False)
fig.supxlabel('Per-question Jaccard with the neutral reference',fontsize=7.5,y=0.1)
h,l=axs[0,0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h.append(Line2D([0],[0],color='k',ls='--',lw=0.8)); l.append('different-question control')
fig.legend(h,l,loc='lower center',ncol=5,fontsize=6.5,frameon=False)
plt.tight_layout(rect=(0,0.08,1,1)); plt.savefig('fig_wc.pdf'); plt.savefig('fig_wc.png',dpi=200)
