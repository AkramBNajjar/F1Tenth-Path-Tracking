import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
g=9.81; LF=0.15875; LR=0.17145; L=LF+LR; CSF,CSR=4.718,5.4562
def K_of(ax,h):
    fzf=g*LR-ax*h; fzr=g*LF+ax*h
    return np.where((fzf>0)&(fzr>0), LR/(CSF*np.maximum(fzf,1e-6))-LF/(CSR*np.maximum(fzr,1e-6)), np.nan)

fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.6))
ax=np.linspace(-6,4,200)
for h,c in zip((0.030,0.074,0.140,0.180),("#9ecae1","#4292c6","#e08214","#c0392b")):
    a1.plot(ax,K_of(ax,h)*1000,lw=2.2,color=c,label=f"CG height {h:.3f} m")
a1.axhline(0,color="#444",ls="--",lw=1.2)
a1.axvspan(-3.76,-1.0,color="#888",alpha=0.15,zorder=0)
a1.text(-2.4,17,"raceline\nbraking range",ha="center",fontsize=8.5,color="#444")
a1.text(3.4,4,"understeer",fontsize=8.5,color="#2166ac",ha="right")
a1.text(3.4,-6,"oversteer",fontsize=8.5,color="#c0392b",ha="right")
a1.set_xlabel("Longitudinal acceleration (m/s$^2$), negative is braking")
a1.set_ylabel("Understeer gradient $K$ ($\\times 10^{-3}$)")
a1.set_title("Predicted: braking flips the car into oversteer",fontsize=11)
a1.legend(frameon=False,fontsize=8.5,loc="upper left"); a1.grid(alpha=0.25)
a1.set_ylim(-25,25)

H=[0.030,0.050,0.074,0.100,0.140,0.180]
PP_ALL=[3.21,3.03,3.09,2.98,2.90,2.88]; PP_BR=[4.78,4.47,4.59,4.18,3.93,4.10]
ST_ALL=[2.13,2.03,2.14,2.00,1.98,1.89]; ST_BR=[5.22,4.75,4.79,4.67,4.51,4.37]
MP_ALL=[0.70,0.75,0.83,0.82,0.94,0.91]; MP_BR=[0.60,0.83,0.68,0.68,1.07,0.97]
a2.plot(H,PP_BR,"o-",color="#888780",lw=2,ms=6,label="Pure pursuit, braking zones")
a2.plot(H,PP_ALL,"o--",color="#888780",lw=1.4,ms=5,alpha=0.6,label="Pure pursuit, whole lap")
a2.plot(H,ST_BR,"o-",color="#c0392b",lw=2,ms=6,label="Stanley, braking zones")
a2.plot(H,ST_ALL,"o--",color="#c0392b",lw=1.4,ms=5,alpha=0.6,label="Stanley, whole lap")
a2.plot(H,MP_BR,"D-",color="#2166ac",lw=2,ms=6,label="MPC, braking zones")
a2.plot(H,MP_ALL,"D--",color="#2166ac",lw=1.4,ms=5,alpha=0.6,label="MPC, whole lap")
a2.set_xlabel("CG height $h$ (m)")
a2.set_ylabel("Mean cross-track error (cm)")
a2.set_title("Measured: raising the CG changes almost nothing",fontsize=11)
a2.legend(frameon=False,fontsize=7.5,ncol=1); a2.grid(alpha=0.25); a2.set_ylim(0,6)

fig.suptitle("Longitudinal load transfer: a strong prediction that the simulation does not confirm",
             fontsize=12.5,y=1.02)
fig.tight_layout(); fig.savefig("loadtransfer_sweep.png",dpi=170,bbox_inches="tight")
print("wrote loadtransfer_sweep.png")
