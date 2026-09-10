import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
K   =[0.00704,0.00462,0.00292,0.0,-0.00266,-0.00671,-0.01237]
CSR =[7.0,6.0,5.456,4.718,4.2,3.6,3.0]
PP  =[3.67,3.30,3.09,2.56,2.48,1.70,1.03]
ST  =[1.83,2.05,2.14,2.19,2.53,2.84,3.50]
MD  =[1.05,0.92,0.83,1.16,1.88,2.99,4.40]
MK  =[0.94,0.78,0.83,0.67,0.67,0.71,0.64]
C={"pp":"#888780","st":"#c0392b","md":"#e08214","mk":"#2166ac"}
x=np.array(K)*1000

fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={"width_ratios":[1.35,1]})
a1.axvspan(-14,0,color="#c0392b",alpha=0.06,zorder=0)
a1.axvspan(0,8,color="#2166ac",alpha=0.06,zorder=0)
a1.axvline(0,color="#666",ls="--",lw=1,zorder=1)
a1.text(-7,4.35,"OVERSTEER",ha="center",fontsize=9,color="#c0392b",fontweight="bold")
a1.text(4,4.35,"UNDERSTEER",ha="center",fontsize=9,color="#2166ac",fontweight="bold")
a1.plot(x,PP,"o-",color=C["pp"],lw=2,ms=7,label="Pure pursuit",zorder=3)
a1.plot(x,ST,"o-",color=C["st"],lw=2,ms=7,label="Stanley",zorder=3)
a1.plot(x,MD,"s-",color=C["md"],lw=2,ms=7,label="MPC — default model",zorder=3)
a1.plot(x,MK,"D-",color=C["mk"],lw=2.6,ms=7,label="MPC — knows balance",zorder=4)
a1.scatter([x[-1]],[ST[-1]],s=230,facecolors="none",edgecolors=C["st"],lw=2.4,zorder=5)
a1.annotate("spun out",(x[-1],ST[-1]),xytext=(8,8),textcoords="offset points",
            fontsize=8.5,color=C["st"],fontweight="bold")
a1.set_xlabel("Understeer gradient $K$  ($\\times 10^{-3}$ rad per m/s$^2$)")
a1.set_ylabel("Mean cross-track error (cm)")
a1.set_title("Handling balance vs tracking accuracy",fontsize=11)
a1.legend(frameon=False,fontsize=8.5,loc="upper center"); a1.grid(alpha=0.25)
a1.set_ylim(0,4.8); a1.set_xlim(-14,8)

vcrit=[np.sqrt(0.3302/abs(k)) if k<0 else np.nan for k in K]
vchar=[np.sqrt(0.3302/k) if k>0 else np.nan for k in K]
a2.plot(x,vchar,"o-",color="#2166ac",lw=2,ms=7,label="Characteristic speed")
a2.plot(x,vcrit,"o-",color="#c0392b",lw=2,ms=7,label="Critical speed (unstable above)")
a2.axhspan(3.8,5.6,color="#666",alpha=0.15,zorder=0)
a2.text(-6.5,4.7,"speeds actually driven",fontsize=8.5,ha="center",color="#444")
a2.axvline(0,color="#666",ls="--",lw=1)
a2.set_xlabel("Understeer gradient $K$  ($\\times 10^{-3}$)")
a2.set_ylabel("Speed (m/s)")
a2.set_title("Where the vehicle becomes unstable",fontsize=11)
a2.legend(frameon=False,fontsize=8.5); a2.grid(alpha=0.25); a2.set_xlim(-14,8)

fig.suptitle("Handling-balance sweep (vgain 0.7) — rear cornering stiffness varied, everything else fixed",
             fontsize=12.5,y=1.02)
fig.tight_layout(); fig.savefig("balance_sweep.png",dpi=170,bbox_inches="tight")
print("wrote balance_sweep.png")
