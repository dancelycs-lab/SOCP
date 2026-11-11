"""Voltage-driven SPMSM experiment and He et al. (TIE 2025) PDO-SMPCC core.

Source DOI: 10.1109/TIE.2024.3423357, equations (16), (19), (38), (43), (44).
This implements the published PDO-SMPCC variant, WITHOUT its harmonic observer.
The shared-bus supervisor and nominal observer initialization are adaptations.
"""
import time
from dataclasses import replace
import numpy as np
from numba import njit
from scipy.integrate import simpson

from model import Case, metrics
from compiled_backup import CompiledReserveFilter
from estimator import ElectricalEstimator
from baselines import mcl, Dual


@njit(cache=True)
def voltage_rhs(t, y, v, b, act, mode, horizon):
    J, R, L, p, psi, C = act
    dy = np.zeros(15)
    for k in range(3):
        f = .0089
        if mode == 1:
            f *= 1 + .3 * np.sin(6*np.pi*t + 2*np.pi*k/3)
        elif mode == 2:
            f *= 1.3 if t < (k+1)*.2*horizon else .7
        w, d, q = y[k], y[3+k], y[6+k]
        vd, vq = v[k], v[3+k]
        dy[k] = (1.5*p*psi*q - f*w)/J
        dy[3+k] = (vd + p*w*L*q - R*d)/L
        dy[6+k] = (vq - p*w*(L*d + psi) - R*q)/L
        dy[9] -= 1.5*(vd*d + vq*q)
        dy[10+k] = 1.5*R*(d*d + q*q)
        dy[13] += f*w*w
    dy[9] -= b
    dy[14] = b
    return dy


@njit(cache=True)
def rk4(t, y, v, b, act, mode, horizon, step):
    k1 = voltage_rhs(t, y, v, b, act, mode, horizon)
    k2 = voltage_rhs(t+step/2, y+step*k1/2, v, b, act, mode, horizon)
    k3 = voltage_rhs(t+step/2, y+step*k2/2, v, b, act, mode, horizon)
    k4 = voltage_rhs(t+step, y+step*k3, v, b, act, mode, horizon)
    return y + step*(k1+2*k2+2*k3+k4)/6


@njit(cache=True)
def reaching(e, tau, exponent):
    return (np.abs(e)**(1-exponent) + 2*np.abs(e)
            + np.abs(e)**(1+exponent))*np.sign(e)/(tau*exponent)


@njit(cache=True)
def nominal_disturbance(y, nom):
    _, R, L, p, psi, _ = nom
    w, d, q = y[:3], y[3:6], y[6:9]
    return np.concatenate((-R/L*d + p*w*q, -R/L*q-p*w*d-p*w*psi/L))


@njit(cache=True)
def controller_voltage(y, ref, nom, method, state, gains, dt, reference_tau=0.):
    # method 0: paper's current regulator sampled at the common inner period.
    # method 1: PDO-SMPCC, 2: nominal DPCC diagnostic.
    cur = y[3:9]
    f = nominal_disturbance(y, nom)
    if method == 0:
        return nom[2]*(1000*(ref-cur)-f)
    if method == 2:
        return nom[2]*((ref-cur)/dt-f)
    tc, ell, t1, t2, rho = gains
    desired=state[18:24] if reference_tau>0 else ref
    derivative=(ref-desired)/reference_tau if reference_tau>0 else np.zeros(6)
    e = desired-cur
    s = e+state[12:18]
    # Supervisory commands are held. The shaped reference has the analytic
    # derivative above, without a differentiating impulse at command changes.
    return nom[2]*(derivative-state[6:12] + reaching(e,t1,rho)+reaching(s,t2,rho))


@njit(cache=True)
def advance(t, y, ref, b, nom, act, mode, horizon, method, state, gains,
            hold, inner_dt, plant_dt, voltage_reserve=0.,reference_tau=0.):
    n = int(np.ceil(hold/inner_dt-1e-9))
    dt = hold/n
    m = int(np.ceil(dt/plant_dt-1e-9))
    ds = dt/m
    rows = np.zeros((n*m, 17))
    volts = np.zeros((n*m, 6))
    for j in range(n):
        v = controller_voltage(y,ref,nom,method,state,gains,dt,reference_tau)
        limit = (1-voltage_reserve)*np.sqrt(max(1e-12,2*y[9]/act[5])/3)
        sat = 0.
        for k in range(3):
            vn = np.hypot(v[k],v[3+k])
            if vn > limit:
                v[k] *= limit/vn; v[3+k] *= limit/vn; sat = 1.
        if method == 1:
            tc, ell, t1, t2, rho = gains
            g = reaching(y[3:9]-state[:6],tc,.5)
            state[:6] += dt*(v/nom[2]+state[6:12]+g)
            state[6:12] += dt*ell*g
            desired=state[18:24].copy() if reference_tau>0 else ref
            state[12:18] += dt*reaching(desired-y[3:9],t1,rho)
            if reference_tau>0:state[18:24]=ref+(desired-ref)*np.exp(-dt/reference_tau)
        for k in range(m):
            y = rk4(t,y,v,b,act,mode,horizon,ds); t += ds
            idx = j*m+k
            rows[idx,0] = t; rows[idx,1:16] = y; rows[idx,16] = sat
            volts[idx] = v
    return y, rows, volts, state


def update_estimator(est, samples, volts, noise=0.):
    """Original integral regression, with exactly integrated held voltages."""
    tt=samples[:,0]
    if noise:
        samples=samples.copy()
        samples[:,1:4]+=noise*.1*np.sin(2*np.pi*17*tt[:,None]+np.arange(3)[None,:]*1.1)
        samples[:,4:10]+=noise*.01*np.sin(2*np.pi*31*tt[:,None]+np.arange(6)[None,:]*.7)
    w=samples[:,1:4]; d=samples[:,4:7]; q=samples[:,7:10]
    integ=lambda x:simpson(x,x=tt,axis=0)
    Id,Iq,Iw,Iwd,Iwq=map(integ,[d,q,w,w*d,w*q])
    V=np.sum(volts*np.diff(tt)[:,None],axis=0)
    if noise:
        V+=integ(noise*.1*np.sin(2*np.pi*23*tt[:,None]+np.arange(6)[None,:]*.9))
    A=np.zeros((6,3)); A[:3,0]=Id; A[:3,1]=d[-1]-d[0]-est.p*Iwq
    A[3:,0]=Iq; A[3:,1]=q[-1]-q[0]+est.p*Iwd; A[3:,2]=est.p*Iw
    A*=est.scale
    weights=np.maximum(np.linalg.norm(A,axis=1),est.weight_floor)
    a=A/weights[:,None]; z=V/weights
    Q,R=np.linalg.qr(np.vstack([est.qr,a]),mode='reduced')
    est.qt=Q.T@np.r_[est.qt,z]; est.qr=R
    sol,_,rank,svals=np.linalg.lstsq(R,est.qt,rcond=1e-9)
    est.rank=int(rank); est.residual=float(np.max(abs(a@sol-z)))
    if rank==3 and svals[-1]/svals[0]>1e-5 and np.all(sol>[.8,.8,.9]) and np.all(sol<[1.2,1.2,1.1]):
        est.value=sol*est.scale; est.accepted=True


def schedule_reference(obs,t,c,base,pl,h):
    if t<.025-1e-10:
        u,b=mcl(obs,t,c,.0089)
        return u,b,min(h,.025-t)
    tt=max(0,t-.025); k=min(pl['N'],int((tt+1e-10)/pl['h'])); local=tt-k*pl['h']
    if k<pl['N']:
        h=min(h,pl['h']-local)
        w0=np.asarray(pl['omega'])[:,k]; acc=np.asarray(pl['qacc'])[:,k]
        ell=np.asarray(pl['loss'])[:,k]; b=pl['brake'][k]
        wr=w0+base.kt*acc*local/base.J; acc=acc*base.kt/c.kt
    else:
        acc=np.zeros(3);ell=np.zeros(3);wr=np.zeros(3);b=0.
    q=np.clip(acc+.0089*obs[:3]/c.kt+50*c.J*(wr-obs[:3])/c.kt,-c.current_limit,c.current_limit)
    im=np.minimum(c.current_limit,np.sqrt(np.maximum(0,ell/c.Rc)))
    d=-np.sqrt(np.maximum(0,im*im-q*q))
    alpha=np.clip((obs[9]-c.Emin)/(.5*(c.Emax-c.Emin)),0,1)
    return np.r_[d*np.sqrt(alpha),q],b*alpha,h


def run(c,pl,method='sampled_proposed',scale=None,mode=1,gains=None,
        inner_dt=5e-5,plant_dt=1e-5,supervise=True,adapt=True,current_reserve=.05,voltage_reserve=.01,reference_tau=.001,outer_policy='conic', nominal_policy=None, execution_filter=None, id_noise=0., ablation='none', component_times=None):
    names={'sampled_proposed':0,'pdo_smpcc':1,'dpcc':2}
    code=names[method]; gains=np.asarray(gains or [.008,1500.,.008,.008,.5])
    nom=np.array([c.J,c.Rs,c.L,c.poles,c.flux,c.capacitance]); act=nom*np.asarray(scale or [1]*6)
    base=c; cloop=nom.copy(); estimator=ElectricalEstimator(c,.01)
    actual=replace(c,J=act[0],Rs=act[1],L=act[2],poles=int(act[3]),flux=act[4],capacitance=act[5])
    y=np.zeros(15);y[:3]=c.omega;y[9]=.5*act[5]*c.voltage0**2
    state=np.zeros(24);state[:6]=y[3:9];state[6:12]=nominal_disturbance(y,nom)
    filt=execution_filter if execution_filter is not None else CompiledReserveFilter((.00623,.01157))
    dual=Dual()
    t=0.;T=c.N*c.h;rows=[np.r_[0.,y,0.][None,:]];reference=[];estimates=[];failed=None
    err_integral=0.;sample_duration=0.;max_error=0.;inverter_excess=-np.inf
    while t<T-1e-10:
        clock=time.perf_counter_ns() if component_times is not None else 0
        c=replace(base,Rs=cloop[1],L=cloop[2],flux=cloop[4])
        obs=y[:13].copy();obs[9]=y[9]*c.capacitance/act[5]
        h=min(.001,T-t)
        if nominal_policy is not None:u,b=nominal_policy(obs,t,c,.00623,h)
        elif outer_policy=='dual':u,b=dual(obs,t,c,.00623,h)
        elif outer_policy=='mcl':
            u,b=mcl(obs,t,c,.00623)
            if t<.025-1e-10:h=min(h,.025-t)
        else:u,b,h=schedule_reference(obs,t,c,base,pl,h)
        if t<.025-1e-10:h=min(h,.025-t)
        if ablation=='no_auxiliary':u[2]=0.;u[5]=0.
        limit=base.current_limit*min(.9 if t<.005 else 1.,1-current_reserve)
        norms=np.hypot(u[:3],u[3:]);ss=np.minimum(1.,limit/np.maximum(norms,1e-30));u*=np.r_[ss,ss]
        if supervise:
            if hasattr(filt,'set_context'):filt.set_context(t,limit)
            u,g,_=filt(obs,None,u,h,c,b)
            if u is None:failed='supervisor_rejected';break
            b=g['applied_brake']
        control_ns=time.perf_counter_ns()-clock if component_times is not None else 0
        start=np.r_[t,y,0.]; reference.append(np.r_[t,u,b])
        # PDO receives the same nominal L as a parameter-independent disturbance
        # controller; its estimate is not replaced with the proposed estimator.
        inner_nom=nom if code==1 else cloop
        y,tr,v,state=advance(t,y,u,b,inner_nom,act,mode,T,code,state,gains,h,inner_dt,plant_dt,voltage_reserve,reference_tau)
        t+=h;rows.append(tr)
        inverter_excess=max(inverter_excess,float(np.max(np.hypot(v[:,:3],v[:,3:])-np.sqrt(np.maximum(0,2*tr[:,10]/act[5]))[:,None]/np.sqrt(3))))
        er=tr[:,4:10]-u
        err_integral+=float(np.sum(er*er))*h/len(tr);sample_duration+=h
        max_error=max(max_error,float(np.max(abs(er))))
        clock=time.perf_counter_ns() if component_times is not None else 0
        if nominal_policy is not None and hasattr(nominal_policy,'observe'):
            nominal_policy.observe(tr,v,b)
        if adapt:
            update_estimator(estimator,np.vstack([start,tr]),v,id_noise)
            if estimator.accepted:cloop[[1,2,4]]=estimator.value
        estimates.append(np.r_[t,estimator.value])
        if component_times is not None:component_times.append([control_ns,time.perf_counter_ns()-clock])
        if not np.all(np.isfinite(y)) or y[9]<=0 or np.max(abs(y))>1e9:
            failed='numerical_divergence';break
    tr=np.vstack(rows)
    m=metrics(actual,np.c_[tr[:,:14],tr[:,-1]])
    total=tr[:,10]+.5*act[0]*np.sum(tr[:,1:4]**2,axis=1)+.75*act[2]*np.sum(tr[:,4:10]**2,axis=1)+np.sum(tr[:,11:14],axis=1)+tr[:,14]+tr[:,15]
    m.update(method=method,completed=failed is None,status=failed or 'completed',duration=t,
             success=bool(m['success'] and failed is None),current_rmse=float(np.sqrt(err_integral/max(6*sample_duration,1e-20))),
             max_current_error=max_error,energy_balance_error=float(np.max(abs(total-total[0]))),
             max_inverter_excess=inverter_excess,filter_changes=filt.changed,inner_dt=inner_dt,plant_dt=plant_dt,
             id_noise=id_noise,ablation=ablation,adaptation=adapt,current_reserve=current_reserve,voltage_reserve=voltage_reserve,reference_tau=reference_tau if code==1 else 0.,outer_policy=outer_policy)
    m['physical_safe']=bool(m['physical_safe'] and inverter_excess<=1e-3)
    m['success']=bool(m['physical_safe'] and m['completed'] and m['deadline_speed']<1.)
    return m,tr,np.asarray(reference),np.asarray(estimates)


@njit(cache=True)
def tracking_trial(gains,method,amp,speed,scale,dt=5e-5,reference_tau=0.):
    """Separate stiff-bus, imposed-speed inner-loop calibration experiment."""
    nom=np.array([.00774,.25,.0048,4.,.32,.00047]);act=nom*scale
    y=np.zeros(15);y[:3]=speed;y[9]=.5*act[5]*800**2
    state=np.zeros(24);state[6:12]=nominal_disturbance(y,nom)
    T=.1;n=int(round(T/dt));result=np.zeros((n,6));err=0.;tail=0.;peak=0.
    for k in range(n):
        t=k*dt
        target=np.zeros(6)
        if t>=.01:
            target[:3]=-amp*.6;target[3:]=-amp*.8
        if t>=.05:
            target[:3]=-amp*.8;target[3:]=-amp*.4
        y[:3]=speed*(1. if t<.07 else 1.2);y[9]=.5*act[5]*800**2
        y,rows,_,state=advance(t,y,target,0.,nom,act,0,T,method,state,gains,dt,dt,dt/5,0.,reference_tau)
        e=y[3:9]-target;err+=np.sum(e*e)/6
        if t>.08:tail+=np.sum(e*e)/6
        peak=max(peak,np.max(np.abs(e)))
        result[k]=np.array([t,y[3],y[6],target[0],target[3],np.hypot(y[3],y[6])])
    return np.sqrt(err/n),np.sqrt(tail/(.02/dt)),peak,result
