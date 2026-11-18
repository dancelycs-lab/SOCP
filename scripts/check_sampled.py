"""Independent physical identities and published-law algebra checks."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from context import OUT
from sampled_drive import voltage_rhs,controller_voltage,nominal_disturbance,reaching
from plant import rhs

def main():
    rng=np.random.default_rng(5111117)
    nom=np.array([.00774,.25,.0048,4.,.32,.00047])
    derivative_error=0.;balance_error=0.;optimality_error=0.;reference_interface_error=0.
    gains=np.array([.016,1500.,.008,.008,.5])
    for _ in range(100):
        y=np.zeros(15);y[:3]=rng.uniform(0,160,3);y[3:9]=rng.uniform(-10,10,6);y[9]=.5*nom[5]*800**2
        ref=rng.uniform(-12,0,6);b=50.;t=.123
        v=controller_voltage(y,ref,nom,0,np.zeros(18),gains,5e-5)
        lim=800/np.sqrt(3);scale=np.minimum(1,lim/np.hypot(v[:3],v[3:]));v*=np.r_[scale,scale]
        d=voltage_rhs(t,y,v,b,nom,1,.6)
        expected,_=rhs(t,y,ref,b,nom,nom,1,.6,0)
        derivative_error=max(derivative_error,float(np.max(abs(d-expected))))
        # Independently differentiated mechanical + magnetic + capacitor + losses.
        e_dot=d[9]+nom[0]*np.dot(y[:3],d[:3])+1.5*nom[2]*np.dot(y[3:9],d[3:9])+d[10:13].sum()+d[13]+d[14]
        balance_error=max(balance_error,abs(e_dot))
        state=rng.normal(size=18);state[6:12]=nominal_disturbance(y,nom)
        v=controller_voltage(y,ref,nom,1,state,gains,5e-5)
        e=ref-y[3:9];s=e+state[12:]
        predicted_s_dot=-(v/nom[2]+state[6:12])+reaching(e,gains[2],gains[4])
        target_s_dot=-reaching(s,gains[3],gains[4])
        optimality_error=max(optimality_error,float(np.max(abs(predicted_s_dot-target_s_dot))))
        shaped=np.zeros(24);shaped[:6]=y[3:9]
        shaped[6:12]=nominal_disturbance(y,nom);shaped[18:24]=y[3:9]
        vs=controller_voltage(y,ref,nom,1,shaped,gains,5e-5,.001)
        va=controller_voltage(y,ref,nom,0,shaped,gains,5e-5)
        reference_interface_error=max(reference_interface_error,float(np.max(abs(vs-va))))
    # Observer driven by a measured analytic ramp with known constant disturbance.
    f=100.;tc=.016;ell=1500.
    def fun(t,z):
        e=f*t-z[0];g=float(reaching(np.array([e]),tc,.5)[0])
        return [z[1]+g,ell*g]
    sol=solve_ivp(fun,[0,.2],[0,0],rtol=1e-9,atol=1e-10,max_step=1e-4)
    observer_error=abs(sol.y[1,-1]-f)
    result=dict(random_states=100,plant_derivative_error=derivative_error,
                energy_derivative_residual=balance_error,pdo_cost_stationarity_residual=optimality_error,
                constant_disturbance_observer_error=observer_error,
                matched_reference_interface_error=reference_interface_error)
    assert derivative_error<1e-9 and balance_error<1e-9 and optimality_error<1e-9
    assert observer_error<.05
    assert reference_interface_error<1e-10
    print(json.dumps(result,indent=2))
    folder=OUT
    (folder/'implementation_checks.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
