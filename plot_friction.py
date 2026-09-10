import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
MU=[1.05,0.90,0.80,0.70,0.60,0.50,0.40]
PP  =[3.09,3.82,4.43,5.30,6.57,8.11,10.77]
ST  =[2.14,3.08,4.17,5.75,8.01,10.83,11.72]
MPD =[0.83,1.07,1.52,2.37,3.62,5.37,8.04]
MPK =[0.83,0.68,0.58,0.50,0.53,0.58,0.71]
ST_CRASH=[False]*6+[True]
C={"pp":"#888780","st":"#c0392b","md":"#e08214","mk":"#2166ac"}

fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.5))
a1.plot(MU,PP,"o-",color=C["pp"],lw=2,ms=7,label="Pure pursuit")
a1.plot(MU,ST,"o-",color=C["st"],lw=2,ms=7,label="Stanley")
a1.plot(MU,MPD,"s-",color=C["md"],lw=2,ms=7,label="MPC — model assumes dry")
a1.plot(MU,MPK,"D-",color=C["mk"],lw=2.6,ms=7,label="MPC — model knows $\\mu$")
a1.scatter([MU[-1]],[ST[-1]],s=230,facecolors="none",edgecolors=C["st"],lw=2.4,zorder=5)
a1.annotate("crashed",(MU[-1],ST[-1]),xytext=(14,-4),textcoords="offset points",
            fontsize=8.5,color=C["st"],fontweight="bold")
a1.invert_xaxis()
a1.set_xlabel("Surface friction coefficient $\\mu$   (dry $\\rightarrow$ icy)")
a1.set_ylabel("Mean cross-track error (cm)")
a1.set_title("Tracking degradation as grip disappears",fontsize=11)
a1.legend(frameon=False,fontsize=8.5); a1.grid(alpha=0.25)

deg=[PP[-1]/PP[0], ST[-1]/ST[0], MPD[-1]/MPD[0], MPK[-1]/MPK[0]]
names=["Pure\npursuit","Stanley","MPC\nassumes dry","MPC\nknows $\\mu$"]
cols=[C["pp"],C["st"],C["md"],C["mk"]]
b=a2.bar(names,deg,color=cols,zorder=3)
for r,d in zip(b,deg):
    a2.annotate(f"{d:.1f}x",(r.get_x()+r.get_width()/2,d),xytext=(0,4),
                textcoords="offset points",ha="center",fontsize=10,fontweight="bold")
a2.axhline(1.0,color="k",ls=":",lw=1)
a2.set_ylabel("Error growth, dry $\\rightarrow$ $\\mu$=0.4")
a2.set_title("How much worse on ice than on dry pavement",fontsize=11)
a2.grid(alpha=0.25,axis="y",zorder=0); a2.set_ylim(0,11)

fig.suptitle("Friction sweep at fixed speed (vgain 0.7) — same controllers, varying surface grip",
             fontsize=12.5,y=1.02)
fig.tight_layout(); fig.savefig("friction_sweep.png",dpi=170,bbox_inches="tight")
print("wrote friction_sweep.png")
