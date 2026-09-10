"""Friction sweep: how does each controller degrade as grip disappears?"""
import warnings, os, time; warnings.filterwarnings("ignore")
import numpy as np, cvxpy as cp
from f110_gym.envs.f110_env import F110Env
import f110_gym

EX=os.path.expanduser("~/dev/f1tenth_gym/examples")
W=np.loadtxt(os.path.join(EX,"example_waypoints.csv"),delimiter=";",skiprows=3)
PATH=W[:,1:3]; PSI=W[:,3]+np.pi/2; KAPPA=W[:,4]; VREF=W[:,5]
SEG=np.linalg.norm(np.roll(PATH,-1,axis=0)-PATH,axis=1); NP=len(PATH)
LF,LR=0.15875,0.17145; L=LF+LR; M=3.74; IZ=0.04712; G=9.81
CSF,CSR=4.718,5.4562; FZF=M*G*LR/L; FZR=M*G*LF/L
STEER_LIM=0.4
BASE=dict(mu=1.0489,C_Sf=CSF,C_Sr=CSR,lf=LF,lr=LR,h=0.074,m=M,I=IZ,
          s_min=-0.4189,s_max=0.4189,sv_min=-3.2,sv_max=3.2,v_switch=7.319,
          a_max=9.51,v_min=-5.0,v_max=20.0,width=0.31,length=0.58)

def wrap(a): return np.arctan2(np.sin(a),np.cos(a))
def cte(p):
    a=PATH;b=np.roll(PATH,-1,axis=0);ab=b-a;ap=p-a
    t=np.clip((ap*ab).sum(1)/np.maximum((ab*ab).sum(1),1e-12),0,1)
    return float(np.linalg.norm(a+t[:,None]*ab-p,axis=1).min())
def nearest(p): return int(np.argmin(np.linalg.norm(PATH-p,axis=1)))

class PurePursuit:
    name="pure_pursuit"; solves=1; tsolve=0.0
    def __init__(s,Ld=1.0,vgain=0.6): s.Ld,s.vgain=Ld,vgain
    def __call__(s,x,y,th,v):
        p=np.array([x,y]); i=nearest(p); j=i
        for _ in range(NP):
            if np.linalg.norm(PATH[j]-p)>=s.Ld: break
            j=(j+1)%NP
        dx,dy=PATH[j,0]-x,PATH[j,1]-y
        xr=np.cos(-th)*dx-np.sin(-th)*dy; yr=np.sin(-th)*dx+np.cos(-th)*dy
        d=np.arctan2(2*L*yr,np.hypot(xr,yr)**2)
        return float(np.clip(d,-STEER_LIM,STEER_LIM)), float(VREF[i]*s.vgain)

class Stanley:
    name="stanley"; solves=1; tsolve=0.0
    def __init__(s,k=3.5,ks=1.0,vgain=0.6): s.k,s.ks,s.vgain=k,ks,vgain
    def __call__(s,x,y,th,v):
        fx,fy=x+L*np.cos(th),y+L*np.sin(th); p=np.array([fx,fy]); i=nearest(p)
        dx,dy=fx-PATH[i,0],fy-PATH[i,1]; e=np.hypot(dx,dy)
        if np.dot([dx,dy],[-np.sin(PSI[i]),np.cos(PSI[i])])>0: e=-e
        d=wrap(PSI[i]-th)+np.arctan2(s.k*e,v+s.ks)
        return float(np.clip(d,-STEER_LIM,STEER_LIM)), float(VREF[i]*s.vgain)

class MPCDyn:
    name="mpc_dyn"
    def __init__(s,N=10,DT=0.05,vgain=0.6,mu_model=1.0489,
                 q_y=60.,q_yd=1.,q_psi=5.,q_psid=1.,r=1.,rd=20.,solve_every=5):
        s.N,s.DT,s.vgain,s.solve_every=N,DT,vgain,solve_every
        s.Cf=CSF*FZF*(mu_model/1.0489); s.Cr=CSR*FZR*(mu_model/1.0489)
        s.prev=0.; s.hold=(0.,1.); s.k=0; s.solves=0; s.tsolve=0.
        s.pey=0.; s.pep=0.
        X=cp.Variable((4,N+1)); U=cp.Variable(N)
        s.x0=cp.Parameter(4); s.A=[cp.Parameter((4,4)) for _ in range(N)]
        s.B=[cp.Parameter(4) for _ in range(N)]; s.C=[cp.Parameter(4) for _ in range(N)]
        s.dprev=cp.Parameter(); Q=np.diag([q_y,q_yd,q_psi,q_psid]); cost=0; cons=[X[:,0]==s.x0]
        for k in range(N):
            cons+=[X[:,k+1]==s.A[k]@X[:,k]+s.B[k]*U[k]+s.C[k], cp.abs(U[k])<=STEER_LIM]
            cost+=cp.quad_form(X[:,k+1],Q)+r*cp.square(U[k])
        for k in range(N-1): cost+=rd*cp.square(U[k+1]-U[k])
        cost+=rd*cp.square(U[0]-s.dprev)
        s.X,s.U=X,U; s.prob=cp.Problem(cp.Minimize(cost),cons)
    def _m(s,v,kap):
        v=max(v,0.8); dt=s.DT; Cf,Cr=s.Cf,s.Cr; Ac=np.zeros((4,4))
        Ac[0,1]=1.; Ac[1,1]=-(Cf+Cr)/(M*v); Ac[1,2]=(Cf+Cr)/M; Ac[1,3]=(-Cf*LF+Cr*LR)/(M*v)
        Ac[2,3]=1.; Ac[3,1]=-(Cf*LF-Cr*LR)/(IZ*v); Ac[3,2]=(Cf*LF-Cr*LR)/IZ
        Ac[3,3]=-(Cf*LF**2+Cr*LR**2)/(IZ*v)
        Bc=np.array([0.,Cf/M,0.,Cf*LF/IZ]); pd=v*kap
        Cc=np.array([0.,-((Cf*LF-Cr*LR)/(M*v)+v)*pd,0.,-((Cf*LF**2+Cr*LR**2)/(IZ*v))*pd])
        return np.eye(4)+Ac*dt, Bc*dt, Cc*dt
    def __call__(s,x,y,th,v):
        if s.k%s.solve_every: s.k+=1; return s.hold
        s.k+=1; p=np.array([x,y]); i0=nearest(p)
        n=np.array([-np.sin(PSI[i0]),np.cos(PSI[i0])])
        ey=float(np.dot(p-PATH[i0],n)); ep=float(wrap(th-PSI[i0]))
        dtm=s.solve_every*0.01
        eyd=(ey-s.pey)/dtm; epd=(ep-s.pep)/dtm; s.pey,s.pep=ey,ep
        idx=[]; i=i0
        for k in range(s.N):
            idx.append(i); st=VREF[i]*s.vgain*s.DT; d=0.
            while d<st: d+=SEG[i]; i=(i+1)%NP
        idx=np.array(idx); vv=np.maximum(VREF[idx]*s.vgain,0.8)
        for k in range(s.N):
            A,B,C=s._m(vv[k],KAPPA[idx[k]]); s.A[k].value=A; s.B[k].value=B; s.C[k].value=C
        s.x0.value=np.array([ey,np.clip(eyd,-8,8),ep,np.clip(epd,-15,15)])
        s.dprev.value=s.prev; t0=time.time()
        try:
            s.prob.solve(solver=cp.OSQP,warm_start=True,eps_abs=1e-4,eps_rel=1e-4,max_iter=8000)
            d=float(s.U[0].value)
        except Exception: d=s.prev
        if not np.isfinite(d): d=s.prev
        s.tsolve+=time.time()-t0; s.solves+=1
        d=float(np.clip(d,-STEER_LIM,STEER_LIM)); s.prev=d
        s.hold=(d,float(VREF[i0]*s.vgain)); return s.hold

def trial(env,ctrl):
    obs,_,_,_=env.reset(np.array([[0.7,0.0,1.37079632679]])); e=[]
    for i in range(9000):
        d,vv=ctrl(obs["poses_x"][0],obs["poses_y"][0],obs["poses_theta"][0],obs["linear_vels_x"][0])
        obs,_,_,_=env.step(np.array([[d,vv]]))
        e.append(cte(np.array([obs["poses_x"][0],obs["poses_y"][0]])))
        if obs["collisions"][0]: return "crash",obs["lap_times"][0],np.mean(e[100:])*100,np.max(e[100:])*100
        if obs["lap_counts"][0]>=1: return "lap",obs["lap_times"][0],np.mean(e[100:])*100,np.max(e[100:])*100
    return "timeout",obs["lap_times"][0],np.mean(e[100:])*100,np.max(e[100:])*100


class MPCDynBal(MPCDyn):
    """MPC whose model is told the true rear cornering stiffness."""
    def __init__(s,csr=5.4562,**kw):
        super().__init__(**kw)
        s.Cr=csr*FZR
CSRS=[7.0,6.0,5.4562,4.718,4.2,3.6,3.0]
VG=0.7
G_=9.81
print("Handling-balance sweep at vgain=%.1f  (default C_Sr=5.4562)\n"%VG)
print(f"{'C_Sr':>7} {'K':>9} {'balance':>11} {'v_crit':>8} | {'pure pursuit':>20} | {'stanley':>20} | {'mpc (default model)':>20} | {'mpc (true balance)':>20}")
for csr in CSRS:
    K=(1/G_)*(1/CSF-1/csr)
    if K>1e-6: tag="under"; vc=np.sqrt(L/K)
    elif K<-1e-6: tag="OVER"; vc=np.sqrt(-L/K)
    else: tag="neutral"; vc=float('nan')
    P=dict(BASE); P["C_Sr"]=csr
    env=F110Env(map=os.path.join(EX,"example_map"),map_ext=".png",num_agents=1,params=P)
    row=f"{csr:>7.3f} {K:>+9.5f} {tag:>11} {vc:>8.1f} |"
    for c in (PurePursuit(vgain=VG), Stanley(vgain=VG),
              MPCDyn(vgain=VG), MPCDynBal(vgain=VG,csr=csr)):
        r,t,me,mx=trial(env,c)
        row+=f" {r:>5} {me:>6.2f} {mx:>6.2f} |"
    print(row)
