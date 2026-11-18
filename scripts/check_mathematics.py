"""Numerical cross-checks of the manuscript's analytical identities and bounds.

Run with the experiment environment. These tests supplement, not replace, the
analytical derivations. Saved experimental results are untouched.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.integrate import solve_ivp, simpson

from context import OUT
from backup import reserves, speed_box, endpoint_box, guard

rng = np.random.default_rng(5110909)
report = {'seed': 5110909, 'checks': {}}


def close(x, y, tol=2e-10):
    assert np.allclose(x, y, atol=tol, rtol=tol), (x, y)


def polynomial_checks():
    r = Poly([0., 1.])
    for _ in range(2000):
        a, c = rng.uniform(0, 300, 2)
        sm, sp = sorted(rng.uniform(0, .95, 2))
        x = np.sqrt(a*c)
        g = rng.uniform(0, x)
        kz = lambda z: (1-r)**2*a+2*r*(1-r)*z+r**2*c
        ds = lambda s, z: 2*s*kz(z).integ()
        up = a-kz(g)-ds(sm, g)
        low = a-kz((a+c)/2)-ds(sp, (a+c)/2)
        close(up.coef, [0, 2*(1-sm)*a-2*g,
                       (2*sm-1)*a-c+2*(1-sm)*g,
                       -2*sm/3*(a+c-2*g)])
        close(low(np.linspace(0, 1, 21)),
              ((1-2*sp)*a-c)*np.linspace(0, 1, 21)
              +sp*(a-c)*np.linspace(0, 1, 21)**2)
        for s in [sm, (sm+sp)/2, sp]:
            exact = a-kz(x)-ds(s, x)
            grid = np.linspace(0, 1, 51)
            assert np.min(up(grid)-exact(grid)) > -1e-9
            assert np.min(exact(grid)-low(grid)) > -1e-9
        close(up(1), a-c-2*sm/3*(a+c+g))
        close(low(1), a-c-sp*(a+c))
        # General envelope width includes the slack of the geometric lift.
        close(up(1)-low(1), 2*(sp-sm)/3*(a+x+c)
              +sp/3*(a+c-2*x)+2*sm/3*(x-g))
        close((a+c)/2-kz(x).integ()(1), (a+c-2*x)/6)
        for s in [sm, sp]:
            q = np.sqrt(c)-(1-s)*np.sqrt(a)+s*grid*(np.sqrt(c)-np.sqrt(a))
            cap = max((1-s)**2*a+c-2*(1-s)*g,
                      a+(1+s)**2*c-2*(1+s)*g)
            assert np.max(q*q) <= cap+1e-9
        B = rng.normal(size=(2, 2, 2))
        Q = np.array([b.T@b for b in B])
        A0, B0, C0 = Q[0, 0, 0], Q[0, 0, 1], Q[0, 1, 1]
        A1, B1, C1 = Q[1, 0, 0], Q[1, 0, 1], Q[1, 1, 1]
        p = (1-r)*Poly([A0, 2*B0, C0])+r*Poly([A1, 2*B1, C1])
        close(p.coef, [A0, 2*B0-A0+A1, C0-2*B0+2*B1, C1-C0])
        nu = rng.uniform(0, 3)
        pq = Poly([A0, 2*B0, C0])+nu*r*(1-r)
        close([[pq.coef[0], (pq.coef[1]-nu)/2],
               [(pq.coef[1]-nu)/2, pq.coef[2]+nu]], Q[0])
        assert min(p(grid)) >= -1e-10 and min(pq(grid)) >= -1e-10
        assert np.hypot(2*B0, A0-C0) <= A0+C0+1e-10
    report['checks']['polynomial_envelopes_cones_and_objective'] = 2000


def backup_checks():
    # Include both signs of rho/(2 lambda)-gamma and nonsmooth zero-speed cases.
    for j in range(4000):
        J, R, L, psi = 10**rng.uniform([-3, -1, -4, -1], [-1, 1, -1, 0])
        lam, f, p = 10**rng.uniform(0, 4), rng.uniform(0, .1), 4
        k, rho, gamma = 1.5*p*psi, 1.5*R, .75*L
        w, d, q = rng.uniform(-200, 200), *rng.uniform(-20, 20, 2)
        if j % 20 == 0:
            w = 0.
        if j % 23 == 0:
            q = 0.
        s = w*q
        sd = k*q*q/J-(f/J+lam)*s
        wd = (k*q-f*w)/J
        A, D = max(rho/(2*lam)-gamma, 0), max(gamma-rho/(2*lam), 0)
        pp = lambda v, dv: dv if v > 0 else (0. if v < 0 else max(dv, 0))
        r2 = d*d+q*q
        Ed = (2*lam*gamma-rho)*r2-k*s
        Bmd = -2*lam*A*r2+k/lam*pp(s, sd)-k*k*q*q/(J*lam)
        Bpd = -2*lam*D*r2+k/lam*pp(-s, -sd)
        assert Ed-Bmd >= -1e-7 and Ed+Bpd <= 1e-7
        Mp = pp(w, wd)+k/(J*lam)*pp(q, -lam*q)
        Mm = pp(-w, -wd)+k/(J*lam)*pp(-q, lam*q)
        assert max(Mp, Mm) <= 1e-7
        close(rho*r2+rho/(2*lam)*(-2*lam*r2), 0)
        # Electrical/mechanical energy cancellation for arbitrary input voltage.
        vd, vq = rng.uniform(-600, 600, 2)
        dd = (vd-R*d+p*w*L*q)/L
        qd = (vq-R*q-p*w*(L*d+psi))/L
        close(-1.5*(vd*d+vq*q)+J*w*wd+1.5*L*(d*dd+q*qd),
              -rho*r2-f*w*w, tol=1e-7)
    report['checks']['backup_derivatives_and_energy_balance'] = 4000


def hold_checks():
    c = SimpleNamespace(J=.00774, Rs=.25, L=.0048, poles=4, flux=.32,
                        capacitance=.00047, Emin=.5*.00047*650**2,
                        Emax=.5*.00047*840**2, current_limit=16.5,
                        heat_budget=(20., 40., 70.), kt=1.92, Rc=.375,
                        omega=(0., 0., 0.))
    fmin, fmax = .00623, .01157
    max_violation = 0.
    max_heat_error = 0.
    for j in range(240):
        y = np.zeros(13)
        y[:3] = rng.uniform(-220, 220, 3)
        y[3:9] = rng.uniform(-14, 14, 6)
        y[9] = rng.uniform(c.Emin, c.Emax)
        u = rng.uniform(-14, 14, 6)
        h = rng.uniform(.00005, .001)
        b = rng.uniform(0, 100)
        phase = rng.uniform(0, 2*np.pi, 3)
        def rhs(t, state):
            w, cur = state[:3], state[3:9]
            d, q = cur[:3], cur[3:]
            cd = 1000*(u-cur)
            f = (fmin+fmax)/2+(fmax-fmin)/2*np.sin(2*np.pi*71*t+phase)
            wd = (c.kt*q-f*w)/c.J
            # Continuous electrical energy balance, independent of endpoint code.
            ed = -c.kt*np.sum(w*q)-c.Rc*np.sum(cur*cur)-1.5*c.L*np.dot(cur, cd)-b
            return np.r_[wd, cd, ed, c.Rc*(d*d+q*q)]
        ts = np.linspace(0, h, 81)
        sol = solve_ivp(rhs, (0, h), y, t_eval=ts, method='DOP853',
                        rtol=2e-12, atol=2e-12)
        assert sol.success
        tr = sol.y.T
        wl, wh = speed_box(y, u, h, c, (fmin, fmax))
        elw, ehw, elo, ehi, ie, dh = endpoint_box(y, u, h, c, (fmin, fmax))
        g = guard(y, u, b, h, c, (fmin, fmax))
        violation = max(0., np.max(wl-tr[:, :3]), np.max(tr[:, :3]-wh),
                        g['Elo']-min(tr[:, 9]), max(tr[:, 9])-g['Ehi'],
                        np.max(elw-tr[-1, :3]), np.max(tr[-1, :3]-ehw),
                        elo-b*h-tr[-1, 9], tr[-1, 9]-ehi+b*h)
        assert violation < 2e-8, (j, violation)
        close(ie, tr[-1, 3:9], tol=2e-9)
        close(dh, tr[-1, 10:], tol=2e-9)
        max_violation = max(max_violation, violation)
        max_heat_error = max(max_heat_error, float(np.max(abs(dh-tr[-1, 10:]))))
    report['checks']['signed_speed_time_varying_friction_holds'] = {
        'cases': 240, 'maximum_enclosure_violation': max_violation,
        'maximum_copper_integral_error': max_heat_error}


def identification_checks():
    for _ in range(1000):
        Astar = rng.normal(size=(18, 3))
        z = rng.uniform(.9, 1.1, 3)
        da, db = rng.normal(size=(18, 3))*.001, rng.normal(size=18)*.001
        A = Astar+da
        b = Astar@z+db
        e = np.linalg.lstsq(A, b, rcond=None)[0]-z
        close(e, np.linalg.pinv(A)@(db-da@z))
        bound = (np.linalg.norm(db)+np.linalg.norm(da, 2)*np.linalg.norm(z))/np.linalg.svd(A, compute_uv=False)[-1]
        assert np.linalg.norm(e) <= bound+1e-12
        # Direct substitution into the physical dq equations checks both signs.
        R, L, psi = .25, .0048, .32
        Rh, Lh, ph = np.array([R, L, psi])*rng.uniform(.9, 1.1, 3)
        p, lam, w = 4, 1000, rng.uniform(-200, 200)
        cur, u = rng.uniform(-15, 15, (2, 2))
        S = np.array([[0., -1.], [1., 0.]])
        eq = np.array([0., 1.])
        v = Rh*cur+Lh*lam*(u-cur)+p*w*(Lh*S@cur+ph*eq)
        actual = (v-R*cur-p*w*(L*S@cur+psi*eq))/L
        disturbance = (Lh/L-1)*lam*(u-cur)+(Rh-R)/L*cur+p*w/L*((Lh-L)*S@cur+(ph-psi)*eq)
        close(actual, lam*(u-cur)+disturbance)
    # An independent smooth trajectory gives voltage integrals for regression.
    t = np.linspace(0, .02, 2001)
    d, q, w = 2*np.cos(17*t), 3*np.sin(23*t), 100+5*np.sin(11*t)
    dd, qd = -34*np.sin(17*t), 69*np.cos(23*t)
    R, L, psi, p = .25, .0048, .32, 4
    integ = lambda x: simpson(x, x=t)
    phi = np.array([[integ(d), d[-1]-d[0]-p*integ(w*q), 0],
                    [integ(q), q[-1]-q[0]+p*integ(w*d), p*integ(w)]])
    vb = [integ(R*d+L*dd-p*w*L*q), integ(R*q+L*qd+p*w*(L*d+psi))]
    close(phi@np.array([R, L, psi]), vb)
    report['checks']['identification_and_mismatch_identities'] = 1000


if __name__ == '__main__':
    polynomial_checks()
    backup_checks()
    hold_checks()
    identification_checks()
    report['status'] = 'passed'
    report['scope'] = 'Numerical identity and enclosure checks; not a formal proof or an experimental rerun.'
    (OUT/'mathematical_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
