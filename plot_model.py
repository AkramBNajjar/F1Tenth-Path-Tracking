import warnings, os; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

VG=[0.6,0.7,0.8]
KIN_MEAN=[1.33,2.33,3.71];  KIN_MAX=[4.82,6.75,10.15]
DYN_MEAN=[0.79,0.83,0.80];  DYN_MAX=[3.17,3.00,2.87]
LAP=[37.18,31.99,28.11]

K,D="#c0392b","#2166ac"
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.5,4.3))

a1.plot(LAP,KIN_MEAN,"o-",color=K,lw=2.2,ms=8,label="Kinematic model")
a1.plot(LAP,DYN_MEAN,"s-",color=D,lw=2.2,ms=8,label="Dynamic model (tire slip)")
for x,y in zip(LAP,KIN_MEAN): a1.annotate(f"{y:.2f}",(x,y),xytext=(0,9),textcoords="offset points",ha="center",fontsize=8,color=K)
for x,y in zip(LAP,DYN_MEAN): a1.annotate(f"{y:.2f}",(x,y),xytext=(0,-15),textcoords="offset points",ha="center",fontsize=8,color=D)
a1.invert_xaxis()
a1.set_xlabel("Lap time (s)  —  faster to the right")
a1.set_ylabel("Mean cross-track error (cm)")
a1.set_title("Kinematic error grows with speed.\nDynamic error does not.",fontsize=11)
a1.legend(frameon=False,fontsize=9); a1.grid(alpha=0.25); a1.set_ylim(0,4.3)

w=0.35; xs=np.arange(3)
a2.bar(xs-w/2,KIN_MAX,w,color=K,label="Kinematic",zorder=3)
a2.bar(xs+w/2,DYN_MAX,w,color=D,label="Dynamic",zorder=3)
a2.set_xticks(xs); a2.set_xticklabels([f"{l:.1f} s" for l in LAP])
a2.set_xlabel("Lap time"); a2.set_ylabel("Max cross-track error (cm)")
a2.set_title("Worst-case error, by model",fontsize=11)
a2.legend(frameon=False,fontsize=9); a2.grid(alpha=0.25,axis="y",zorder=0)

fig.suptitle("MPC prediction model vs tracking accuracy — same controller, same weights",fontsize=12.5,y=1.02)
fig.tight_layout(); fig.savefig("model_comparison.png",dpi=170,bbox_inches="tight")
print("wrote model_comparison.png")
