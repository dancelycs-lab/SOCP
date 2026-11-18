"""Friction-bounded energy scheduling and feasibility audit."""
import time
import numpy as np
import cvxpy as cp

def solve(c,friction=.0089,brake_profile=None,prepared=None,prepare_only=False,friction_bounds=None):
 bounds=(friction,friction) if friction_bounds is None else tuple(friction_bounds)
 profile=tuple(np.full(c.N,c.brake_limit) if brake_profile is None else np.array(brake_profile))
 signature=(c.N,c.h,c.J,c.Rs,c.L,c.poles,c.flux,c.current_limit,c.capacitance,c.voltage_min,c.voltage_max,tuple(v>0 for v in c.omega),bounds,profile)
 if prepared is not None:
  if prepared['signature']!=signature:raise ValueError('Prepared SOCP configuration changed; prepare a new cache')
  return _execute(c,prepared)
 start=time.perf_counter();n=len(c.omega);N=c.N;s=bounds[0]*c.h/c.J;shi=bounds[1]*c.h/c.J;beta=2*s/3
 if not 0<=s<=shi<1:raise ValueError('Require 0 <= f_min*h/J <= f_max*h/J < 1')
 active=[i for i,v in enumerate(c.omega) if v>0];m=len(active)
 kf=cp.Variable((m,N-1),nonneg=True);gf=cp.Variable((m,N-1),nonneg=True)
 Kinit=cp.Parameter(n,nonneg=True);Hcap=cp.Parameter(n,nonneg=True);Einit=cp.Parameter()
 Kinit.value=c.initial_K;Hcap.value=np.array(c.heat_budget);Einit.value=c.E0
 K=cp.vstack([cp.hstack([Kinit[i],kf[active.index(i),:],0.]) if i in active else np.zeros(N+1)for i in range(n)])
 g=cp.vstack([cp.hstack([gf[active.index(i),:],0.]) if i in active else np.zeros(N)for i in range(n)])
 a=K[:,:-1];b=K[:,1:];ell=cp.Variable((n,N),nonneg=True);pb=cp.Variable(N,nonneg=True)
 Eu=cp.Variable(N+1);El=cp.Variable(N+1);H=cp.Variable((n,N+1),nonneg=True)
 coeff=2*c.Rc*c.J/(c.kt**2*c.h**2)
 cons=[b<=(1-shi)**2*a,ell>=coeff*((1-s)**2*a+b-2*(1-s)*g),ell>=coeff*(a+(1+s)**2*b-2*(1+s)*g),ell<=c.Rc*c.current_limit**2,
 ell>=coeff*((1-shi)**2*a+b-2*(1-shi)*g),ell>=coeff*(a+(1+shi)**2*b-2*(1+shi)*g),
 H[:,0]==0,H[:,1:]==H[:,:-1]+c.h*ell,H<=cp.reshape(Hcap,(n,1),order='C'),pb<=(c.brake_limit if brake_profile is None else np.array(brake_profile)),Eu[0]==Einit,El[0]==Einit]
 af=cp.reshape(K[active,:-2],(m*(N-1),),order='C');bf=cp.reshape(K[active,1:-1],(m*(N-1),),order='C');gg=cp.reshape(g[active,:-1],(m*(N-1),),order='C')
 cons +=[cp.SOC(af+bf,cp.vstack([2*gg,af-bf]),axis=0)]
 P=cp.sum(ell,axis=0)+pb
 cons +=[Eu[1:]==Eu[:-1]+cp.sum(a-b-beta*(a+b+g),axis=0)-c.h*P,
 El[1:]==El[:-1]+cp.sum(a-b-shi*(a+b),axis=0)-c.h*P,Eu[-1]<=c.Emax,El[-1]>=c.Emin]
 Us=Eu[:-1]-c.mag_alpha*cp.sum(ell,axis=0);Ls=El[:-1]-c.mag_alpha*cp.sum(ell,axis=0)
 u1=cp.sum(2*(1-s)*a-2*g,axis=0)-c.h*P
 u2=cp.sum((2*s-1)*a-b+2*(1-s)*g,axis=0)
 u3=-beta*cp.sum(a+b-2*g,axis=0)
 A0=cp.Variable(N,nonneg=True);C0=cp.Variable(N,nonneg=True);B0=cp.Variable(N)
 A1=cp.Variable(N,nonneg=True);C1=cp.Variable(N,nonneg=True);B1=cp.Variable(N)
 cons +=[A0==c.Emax-Us,2*B0-A0+A1==-u1,C0-2*B0+2*B1==-u2,C1-C0==-u3,
 cp.SOC(A0+C0,cp.vstack([2*B0,A0-C0]),axis=0),cp.SOC(A1+C1,cp.vstack([2*B1,A1-C1]),axis=0)]
 lm=Ls-c.Emin;l1=cp.sum((1-2*shi)*a-b,axis=0)-c.h*P;l2=shi*cp.sum(a-b,axis=0);lam=cp.Variable(N,nonneg=True)
 cons +=[cp.SOC(l2+lam+lm,cp.vstack([l2+lam-lm,l1-lam]),axis=0)]
 scale=max(1.,sum(c.initial_K));obj=cp.sum(K[:,1:])/(scale*N)+1e-4*c.h*cp.sum(ell)/scale
 problem=cp.Problem(cp.Minimize(1000*obj),cons)
 cache=dict(signature=signature,friction_bounds=bounds,problem=problem,Kinit=Kinit,Hcap=Hcap,Einit=Einit,variables=dict(K=K,g=g,loss=ell,brake=pb,Eu=Eu,El=El,heat=H),friction=friction,build_seconds=time.perf_counter()-start)
 if prepare_only:
  problem.get_problem_data('CLARABEL');cache['offline_seconds']=time.perf_counter()-start
  return cache
 return _execute(c,cache,cache['build_seconds'])

def _execute(c,cache,build=0.):
 start=time.perf_counter();problem=cache['problem'];cache['Kinit'].value=c.initial_K;cache['Hcap'].value=np.array(c.heat_budget);cache['Einit'].value=c.E0
 try:problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=300,max_threads=1,direct_solve_method='qdldl')
 except cp.error.SolverError as e:return dict(status='solver_error',error=str(e),build_seconds=build,wall_seconds=time.perf_counter()-start)
 out=dict(status=problem.status,objective=None,build_seconds=build,wall_seconds=time.perf_counter()-start,solver_seconds=problem.solver_stats.solve_time,iterations=problem.solver_stats.num_iters,friction=cache['friction'],friction_bounds=cache['friction_bounds'])
 values=cache['variables']
 if values['K'].value is not None:
  out.update({k:v.value for k,v in values.items()})
  out['objective']=float((np.sum(out['K'][:,1:])/c.N+1e-4*c.h*np.sum(out['loss']))/max(1.,sum(c.initial_K)))
 return out

def audit(c,r,friction=.0089):
 K=np.maximum(0,r['K']);om=np.sqrt(2*K/c.J);a=K[:,:-1];b=K[:,1:];x=np.sqrt(a*b);s=friction*c.h/c.J;beta=2*s/3
 slo=r.get('friction_bounds',(friction,friction))[0]*c.h/c.J;shi=r.get('friction_bounds',(friction,friction))[1]*c.h/c.J
 qacc=c.J*np.diff(om,axis=1)/(c.kt*c.h);q0=qacc+friction*om[:,:-1]/c.kt;q1=qacc+friction*om[:,1:]/c.kt
 # Increase ell only if numerical residual requires it, and recompute physical energy.
 ell=np.maximum(r['loss'],c.Rc*np.maximum(q0*q0,q1*q1));P=np.sum(ell,axis=0)+r['brake']
 E=np.r_[c.E0,c.E0+np.cumsum(np.sum(a-b-beta*(a+b+x),axis=0)-c.h*P)]
 hi=c.E0-c.initial_magnetic;lo=hi;env_error=0.
 for k in range(c.N):
  coef=[E[k]-c.mag_alpha*np.sum(ell[:,k]),np.sum(2*(1-s)*a[:,k]-2*x[:,k])-c.h*P[k],np.sum((2*s-1)*a[:,k]-b[:,k]+2*(1-s)*x[:,k]),-beta*np.sum(a[:,k]+b[:,k]-2*x[:,k])]
  roots=np.roots([3*coef[3],2*coef[2],coef[1]]) if abs(coef[3])+abs(coef[2])>1e-15 else []
  rr=[0.,1.]+[float(z.real)for z in roots if abs(z.imag)<1e-8 and 0<z.real<1]
  vals=[sum(coef[j]*t**j for j in range(4))for t in rr];hi=max(hi,max(vals));lo=min(lo,min(vals))
  for t in np.linspace(0,1,11):
   actual=sum(coef[j]*t**j for j in range(4));gg=r['g'][:,k]
   upper=r['Eu'][k]-c.mag_alpha*np.sum(r['loss'][:,k])+np.sum((2*(1-slo)*a[:,k]-2*gg)*t+((2*slo-1)*a[:,k]-b[:,k]+2*(1-slo)*gg)*t*t-2*slo/3*(a[:,k]+b[:,k]-2*gg)*t**3)-c.h*(np.sum(r['loss'][:,k])+r['brake'][k])*t
   lower=r['El'][k]-c.mag_alpha*np.sum(r['loss'][:,k])+np.sum(((1-2*shi)*a[:,k]-b[:,k])*t+shi*(a[:,k]-b[:,k])*t*t)-c.h*(np.sum(r['loss'][:,k])+r['brake'][k])*t
   env_error=max(env_error,actual-upper,lower-actual)
 hi=max(hi,E[-1]);lo=min(lo,E[-1]);heat=c.h*np.sum(ell,axis=1)
 out=dict(final_speed_max=float(np.max(om[:,-1])),initial_energy_error_J=float(np.max(abs(K[:,0]-c.initial_K))),max_geometric_cone_excess_J=float(np.max(r['g']-x)),max_voltage=float(np.sqrt(max(0,2*hi/c.capacitance))),min_voltage=float(np.sqrt(max(0,2*lo/c.capacitance))),max_current=float(np.sqrt(ell.max()/c.Rc)),max_heat_excess=float(np.max(heat-np.array(c.heat_budget))),max_positive_q=float(max(q0.max(),q1.max())),max_envelope_error_J=float(env_error),max_raw_gap_J=float(np.max(r['Eu']-r['El'])),canonical_endpoint_gap_J=float(np.sum(beta*((a+b)/2-x))),loss_repair_max=float(np.max(ell-r['loss'])))
 out['valid']=bool(out['final_speed_max']<.002 and out['initial_energy_error_J']<1e-4 and out['max_voltage']<=c.voltage_max+2e-4 and out['min_voltage']>=c.voltage_min-2e-4 and out['max_current']<=c.current_limit+1e-5 and out['max_heat_excess']<=1e-5 and out['max_positive_q']<=1e-5 and env_error<=1e-4)
 return out,dict(omega=om,qacc=qacc,loss=ell,E=E)
