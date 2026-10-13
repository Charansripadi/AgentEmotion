import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
emos=['normal','career','tip','careful','expert','threat','carefree']
d={'Qwen3-1.7B':[83.66,76.59,75.77,77.70,69.27,75.25,48.48],
   'Qwen3-8B':[92.53,59.93,76.06,90.74,89.91,72.90,24.20],
   'Llama-3.2-3B':[79.45,62.25,41.50,56.54,75.05,62.47,8.84]}
fig,axs=plt.subplots(1,3,figsize=(6.6,2.3),sharey=True)
for ax,(m,v) in zip(axs,d.items()):
    cols=['#8a8a8a']+['#3b6ea8']*5+['#c8553d']
    b=ax.bar(range(7),v,color=cols,width=0.72)
    for i,x in enumerate(v): ax.text(i,x+1.5,f'{x:.0f}',ha='center',va='bottom',fontsize=7)
    ax.set_title(m,fontsize=8.5,loc='left'); ax.set_xticks(range(7)); ax.set_xticklabels(emos,rotation=55,ha='right',fontsize=7.5)
    ax.set_ylim(0,105)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.grid(axis='y',color='#dddddd',lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel('GSM8K accuracy (%)')
plt.tight_layout(); plt.savefig('fig_phase1_gsm8k.pdf'); plt.savefig('fig_phase1_gsm8k.png',dpi=200)
