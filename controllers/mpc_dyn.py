"""MPC with the DYNAMIC bicycle model (tire slip) vs the kinematic one."""
import warnings, os, time; warnings.filterwarnings("ignore")
import numpy as np, cvxpy as cp
from f110_gym.envs.f110_env import F110Env
import f110_gym

EX="EX=os.path.expanduser("~/dev/f1tenth_gym/examples")"
W=np.loadtxt(os.path.join(EX,"example_waypoints.csv"),delimiter=";",skiprows=3)
PATH=W[:,1:3]; PSI=W[:,3]+np.pi/2; KAPPA=W[:,4]; VREF=W[:,5]
SEG=np.linalg.norm(np.roll(PATH,-1,axis=0)-PATH,axis=1); NP=len(PATH)

# --- vehicle parameters (simulator defaults) ---
MU=1.0489; CSF=4.718; CSR=5.4562
LF=0.15875; LR=0.17145; L=LF+LR; M=3.74; IZ=0.04712; G=9.81
FZF=M*G*LR/L; FZR=M*G*LF/L          # static axle normal loads
CF=CSF*FZF; CR=CSR*FZR               # axle cornering stiffness, N/rad
STEER_LIM=0.4
print("axle cornering stiffness: Cf=%.1f N/rad  Cr=%.1f N/rad"%(CF,CR))
# understeer gradient K = m/L * (lr/Cf - lf/Cr)   [rad per m/s^2]
K_us=M/L*(LR/CF - LF/CR)
print("understeer gradient K = %+.5f rad/(m/s^2)  -> %s"%(
    K_us, "UNDERSTEER" if K_us>0 else "OVERSTEER"))

def wrap(a): return np.arctan2(np.sin(a),np.cos(a))
def cte(p):
    a=PATH;b=np.roll(PATH,-1,axis=0);ab=b-a;ap=p-a
    t=np.clip((ap*ab).sum(1)/np.maximum((ab*ab).sum(1),1e-12),0,1)
    return float(np.linalg.norm(a+t[:,None]*ab-p,axis=1).min())

class MPCDyn:
    """
    Dynamic bicycle model in error coordinates (Rajamani Ch.3).
    State: [e_y, e_y_dot, e_psi, e_psi_dot]
    Tire lateral force = C * slip_angle -> the car does NOT go where it points.
    """
    name="mpc_dyn"
    def __init__(s,N=10,DT=0.05,vgain=0.6,q_y=60.0,q_yd=1.0,q_psi=5.0,q_psid=1.0,
                 r=1.0,rd=20.0,solve_every=5):
        s.N,s.DT,s.vgain,s.solve_every=N,DT,vgain,solve_every
        s.prev=0.0; s.hold=(0.0,1.0); s.k=0; s.solves=0; s.tsolve=0.0
        s.last_e=np.zeros(4); s.prev_ey=0.0; s.prev_epsi=0.0
        X=cp.Variable((4,N+1)); U=cp.Variable(N)
        s.x0=cp.Parameter(4)
        s.A=[cp.Parameter((4,4)) for _ in range(N)]
        s.B=[cp.Parameter(4) for _ in range(N)]
        s.C=[cp.Parameter(4) for _ in range(N)]
        s.dprev=cp.Parameter()
        cost=0; cons=[X[:,0]==s.x0]
        Q=np.diag([q_y,q_yd,q_psi,q_psid])
        for k in range(N):
            cons+=[X[:,k+1]==s.A[k]@X[:,k]+s.B[k]*U[k]+s.C[k], cp.abs(U[k])<=STEER_LIM]
            cost+=cp.quad_form(X[:,k+1],Q)+r*cp.square(U[k])
        for k in range(N-1): cost+=rd*cp.square(U[k+1]-U[k])
        cost+=rd*cp.square(U[0]-s.dprev)
        s.X,s.U=X,U; s.prob=cp.Problem(cp.Minimize(cost),cons)

    def _mats(s,v,kap):
        v=max(v,0.8); dt=s.DT
        Ac=np.zeros((4,4))
        Ac[0,1]=1.0
        Ac[1,1]=-(CF+CR)/(M*v); Ac[1,2]=(CF+CR)/M; Ac[1,3]=(-CF*LF+CR*LR)/(M*v)
        Ac[2,3]=1.0
        Ac[3,1]=-(CF*LF-CR*LR)/(IZ*v); Ac[3,2]=(CF*LF-CR*LR)/IZ
        Ac[3,3]=-(CF*LF**2+CR*LR**2)/(IZ*v)
        Bc=np.array([0.0, CF/M, 0.0, CF*LF/IZ])
        psid=v*kap                                     # desired yaw rate
        Cc=np.array([0.0, -((CF*LF-CR*LR)/(M*v)+v)*psid, 0.0,
                     -((CF*LF**2+CR*LR**2)/(IZ*v))*psid])
        return np.eye(4)+Ac*dt, Bc*dt, Cc*dt

    def __call__(s,x,y,th,v):
        if s.k % s.solve_every: s.k+=1; return s.hold
        s.k+=1
        p=np.array([x,y]); i0=int(np.argmin(np.linalg.norm(PATH-p,axis=1)))
        n=np.array([-np.sin(PSI[i0]),np.cos(PSI[i0])])
        e_y=float(np.dot(p-PATH[i0],n)); e_psi=float(wrap(th-PSI[i0]))
        dt_meas=s.DT*s.solve_every/5*0.05 if False else s.solve_every*0.01
        e_yd=(e_y-s.prev_ey)/dt_meas; e_psid=(e_psi-s.prev_epsi)/dt_meas
        s.prev_ey, s.prev_epsi = e_y, e_psi
        idx=[]; i=i0
        for k in range(s.N):
            idx.append(i); step=VREF[i]*s.vgain*s.DT; d=0.0
            while d<step: d+=SEG[i]; i=(i+1)%NP
        idx=np.array(idx); vv=np.maximum(VREF[idx]*s.vgain,0.8)
        for k in range(s.N):
            Ad,Bd,Cd=s._mats(vv[k],KAPPA[idx[k]])
            s.A[k].value=Ad; s.B[k].value=Bd; s.C[k].value=Cd
        s.x0.value=np.array([e_y,np.clip(e_yd,-8,8),e_psi,np.clip(e_psid,-15,15)])
        s.dprev.value=s.prev
        t0=time.time()
        try:
            s.prob.solve(solver=cp.OSQP,warm_start=True,eps_abs=1e-4,eps_rel=1e-4,max_iter=8000)
            d=float(s.U[0].value)
        except Exception: d=s.prev
        if not np.isfinite(d): d=s.prev
        s.tsolve+=time.time()-t0; s.solves+=1
        d=float(np.clip(d,-STEER_LIM,STEER_LIM)); s.prev=d
        s.hold=(d,float(VREF[i0]*s.vgain)); return s.hold

env=F110Env(map=os.path.join(EX,"example_map"),map_ext=".png",num_agents=1)
def trial(ctrl):
    obs,_,_,_=env.reset(np.array([[0.7,0.0,1.37079632679]])); e=[];st=[]
    for i in range(9000):
        d,vv=ctrl(obs["poses_x"][0],obs["poses_y"][0],obs["poses_theta"][0],obs["linear_vels_x"][0])
        obs,_,_,_=env.step(np.array([[d,vv]]))
        e.append(cte(np.array([obs["poses_x"][0],obs["poses_y"][0]]))); st.append(d)
        if obs["collisions"][0]: r="crash";break
        if obs["lap_counts"][0]>=1: r="lap";break
    else: r="timeout"
    e=np.array(e[100:]); sr=np.abs(np.diff(np.array(st[100:])))/0.01
    return r,obs["lap_times"][0],e.mean()*100,e.max()*100,sr.mean(),1000*ctrl.tsolve/max(ctrl.solves,1)

print(f"\n{'vgain':>6} {'result':>7} {'lap_t':>7} {'mean_cm':>8} {'max_cm':>7} {'srate':>6} {'ms':>5}")
for vg in (0.6,0.7,0.8):
    r,t,me,mx,sr,ms=trial(MPCDyn(vgain=vg))
    print(f"{vg:>6.1f} {r:>7} {t:>7.2f} {me:>8.2f} {mx:>7.2f} {sr:>6.2f} {ms:>5.1f}")
