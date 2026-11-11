"""Single-column, two-panel-column figures from the final integrated controller.
No smoothing, interpolation, or changes to the source experiment data.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from context import ROOT,OUT
DF=DR=OUT
F=OUT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'cm','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'legend.handlelength':1.6,'legend.labelspacing':.2,'legend.borderaxespad':.3,'lines.linewidth':1.1,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'axes.labelpad':2,'axes.titlepad':4,'xtick.major.pad':2,'ytick.major.pad':2})
methods=['conic','dual','mcl','cvo','mcl_pdo'];names=['Proposed','AOV','MCL','CVO','MCL/PDO'];styles=[('#0072B2','-'),('#D55E00','--'),('#009E73','-.'),('#AA4499',':'),('#444444',(0,(5,1,1,1,1,1)))]
def layout(rows,height,**kw):
 fig,axs=plt.subplots(rows,2,figsize=(3.5,height),layout='constrained',squeeze=False,**kw)
 fig.get_layout_engine().set(w_pad=.015,h_pad=.025,wspace=.035,hspace=.035)
 for ax in axs.flat:
  ax.grid(alpha=.2);ax.xaxis.set_major_locator(MaxNLocator(3));ax.tick_params(length=2.5,width=.6)
 return fig,axs


def cumulative(tr,J=.00774):
 t=tr[:,0];K=.5*J*np.sum(tr[:,1:4]**2,axis=1)
 return np.r_[0,np.cumsum((K[1:]+K[:-1])*.5*np.diff(t))]

def trace_path(folder,pattern,method):
 return folder/pattern.format(method=method)

def responses(cases,output,travel=False):
 fig,axs=layout(3,3.7,sharex='col')
 for col,(folder,pattern,title,jfactor,cfactor) in enumerate(cases):
  for method,name,(color,ls) in zip(methods,names,styles):
   source=trace_path(folder,pattern,method)
   tr=np.load(source)['trace'];t=tr[:,0]
   axs[0,col].plot(t,np.max(abs(tr[:,1:4]),axis=1),color=color,ls=ls,label=name)
   middle=np.max(np.vstack([np.zeros(3),np.cumsum(.5*(abs(tr[1:,1:4])+abs(tr[:-1,1:4]))*np.diff(t)[:,None],axis=0)]),axis=1)if travel else np.sqrt(2*tr[:,10]/(.00047*cfactor))
   axs[1,col].plot(t,middle,color=color,ls=ls)
   axs[2,col].plot(t,cumulative(tr,.00774*jfactor),color=color,ls=ls)
  axs[0,col].set_title(title)
  axs[0,col].axhline(1,color='.5',lw=.6,ls=':')
  if not travel:
   for lim in [650,840]:axs[1,col].axhline(lim,color='.5',lw=.6,ls=':')
   axs[1,col].set_ylim(640,850);axs[1,col].set_yticks([650,750,840])
  for row,label in enumerate([r'$\max_i|\omega_i|$ (rad/s)',r'$\theta_{\max}(t)$ (rad)'if travel else r'${\mathcal{V}}$ (V)',r'${\mathcal{J}}_{\mathcal{K}}(t)$ (J s)']):
   axs[row,col].set(ylabel=label if col==0 else '',xlim=(0,t[-1]))
  axs[2,col].set_xlabel('Time (s)')
 handles,_=axs[0,0].get_legend_handles_labels()
 axs[2,0].legend(handles,names,loc='lower right',frameon=False)
 fig.savefig(F/output);plt.close(fig)
responses([(DF/'friction','h0_p0_{method}_1.npz','(a) Sinusoidal, 0 W',1,1),(DF/'friction','h11_p100_{method}_2.npz','(b) Steps, 100 W',1,1)],'time_responses.pdf')
responses([(DF/'newstates','n0_p0_combined_pos_{method}.npz','(a) C+, 0 W',1.05,.95),(DF/'newstates','n7_p100_combined_neg_{method}.npz','(b) C−, 100 W',.95,1.05)],'new_state_responses.pdf',travel=True)
cases=json.loads((ROOT/'config/cases.json').read_text());case=next(x['case']for x in cases if x['id']=='h0_p0');H=np.array(case['heat_budget'])
fig,axs=layout(3,3.85,sharex=True)
for method,name,(color,ls) in zip(methods,names,styles):
 tr=np.load(trace_path(DF/'friction','h0_p0_{method}_1.npz',method))['trace'];t=tr[:,0]
 for i in range(3):
  axs[i,0].plot(t,tr[:,4+i],color=color,ls=ls,label=name)
  axs[i,1].plot(t,tr[:,11+i]/H[i],color=color,ls=ls,label=name)
for i in range(3):
 axs[i,0].set_title(f'Axis {i+1}'+(' (stationary)'if i==2 else ''));axs[i,1].set_title(f'Axis {i+1}')
 axs[i,1].axhline(1,color='.5',ls=':',lw=.6)
 axs[i,1].set(ylim=(0,1.06),yticks=[0,.5,1],ylabel=r'${\mathcal{H}}_i/\bar {\mathcal{H}}_i$')
 axs[i,0].set(ylim=(-18,1),yticks=[-15,-10,-5,0],ylabel=r'$i_{d,i}$ (A)')
 for ax in axs[i]:ax.set_xlim(0,t[-1])
for ax in axs[-1]:ax.set_xlabel('Time (s)')
axs[1,0].legend(names,frameon=False,loc='upper left')
fig.savefig(F/'axis_allocation.pdf');plt.close(fig)
fig,axs=layout(2,3.1)
for method,name,(color,style) in zip(methods,names,styles):
 source=trace_path(DR/'noisy','h0_p0_psi_plus_{method}.npz',method)
 r=np.load(source);tr=r['trace'];t=tr[:,0]
 axs[0,0].plot(t,np.max(abs(tr[:,1:4]),axis=1),label=name,color=color,ls=style)
 axs[0,1].plot(t,np.sqrt(2*tr[:,10]/.00047),color=color,ls=style)
 axs[1,0].plot(t,cumulative(tr),color=color,ls=style)
 if method=='conic':
  est=r['estimate'];tt=np.r_[0,est[:,0]];psi=np.r_[.32,est[:,3]]
  axs[1,1].plot(tt*1000,100*(psi/.3264-1),label='Estimated',color=color)
axs[1,1].axhline(100*(1/1.02-1),color='.45',ls='--',label='Nominal');axs[1,1].axhline(0,color='.65',ls=':',lw=.7)
axs[0,0].set(xlabel='Time (s)',ylabel=r'$\max_i|\omega_i|$ (rad/s)',title='(a) Braking');axs[0,0].legend(frameon=False,loc='upper right')
axs[0,1].set(xlabel='Time (s)',ylabel=r'${\mathcal{V}}$ (V)',title='(b) Bus voltage',ylim=(640,850),yticks=[650,750,840])
for vv in [650,840]:axs[0,1].axhline(vv,color='.6',ls=':',lw=.6)
axs[1,0].set(xlabel='Time (s)',ylabel=r'${\mathcal{J}}_{\mathcal{K}}(t)$ (J s)',title='(c) Motion cost')
axs[1,1].set(xlabel='Time (ms)',ylabel='Flux error (%)',title='(d) Identification',xlim=(0,10));axs[1,1].legend(frameon=False,loc='center right')
for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.set_xlim(0,t[-1])
fig.savefig(F/'adaptive_execution.pdf');plt.close(fig)
print('Four final-method result PDFs regenerated at native single-column width.')

# Component removal on the same first/last cases as the performance figures.
fig,axs=layout(3,3.7,sharex='col')
for col,(cid,title) in enumerate([('h0_p0','(a) First state, 0 W'),('h11_p100','(b) Last state, 100 W')]):
 for folder,label,color,ls in [('friction','Complete method','#0072B2','-'),('auxiliary','No auxiliary axis','#D55E00','--')]:
  tr=np.load(DF/f'{folder}/{cid}_conic_1.npz')['trace'];t=tr[:,0]
  axs[0,col].plot(t,np.max(abs(tr[:,1:4]),axis=1),label=label,color=color,ls=ls)
  axs[1,col].plot(t,cumulative(tr),color=color,ls=ls)
  axs[2,col].plot(t,tr[:,13],color=color,ls=ls)
 axs[0,col].set_title(title);axs[0,col].axhline(1,color='.45',lw=.6,ls=':')
 for row,label in enumerate([r'$\max_i|\omega_i|$ (rad/s)',r'${\mathcal{J}}_{\mathcal{K}}(t)$ (J s)',r'${\mathcal{H}}_3$ (J)']):axs[row,col].set(ylabel=label if col==0 else '',xlim=(0,t[-1]))
 axs[2,col].set_xlabel('Time (s)')
axs[0,0].legend(frameon=False,loc='upper right')
fig.savefig(F/'auxiliary_ablation.pdf');plt.close(fig)
print('Auxiliary-axis ablation PDF regenerated from complete saved trajectories.')

# Common energy schedule: only the inner current-control implementation changes.
fig,axs=layout(3,3.65,sharex='col')
for col,(cid,title,jf,cf,condition) in enumerate([('h0_p0','(a) C+, 0 W',1.05,.95,'combined_pos'),('h11_p100','(b) C−, 100 W',.95,1.05,'combined_neg')]):
 for method,label,color,ls in [('conic','Proposed','#0072B2','-'),('pdo_smpcc','Same plan / PDO','#D55E00','--'),('dpcc','Same plan / DPCC','#009E73','-.')]:
  data=np.load(OUT/'parameters'/f'{cid}_{condition}_{method}.npz');tr=data['trace'];t=tr[:,0]
  axs[0,col].plot(t,np.max(abs(tr[:,1:4]),axis=1),label=label,color=color,ls=ls)
  axs[1,col].plot(t,np.max(np.hypot(tr[:,4:7],tr[:,7:10]),axis=1),color=color,ls=ls)
  axs[2,col].plot(t,cumulative(tr,.00774*jf),color=color,ls=ls)
 axs[0,col].set_title(title)
 axs[1,col].axhline(16.5,color='.5',ls=':',lw=.7)
 for row,label in enumerate([r'$\max_i|\omega_i|$ (rad/s)',r'$\max_i\|\mathbf{i}_i\|$ (A)',r'${\mathcal{J}}_{\mathcal{K}}(t)$ (J s)']):
  axs[row,col].set(ylabel=label if col==0 else '',xlim=(0,t[-1]))
 axs[2,col].set_xlabel('Time (s)')
axs[2,0].legend(*axs[0,0].get_legend_handles_labels(),loc='lower right',frameon=False)
fig.savefig(F/'current_loop_comparison.pdf');plt.close(fig)
