"""Transcribe the measurements shown in the executed notebook's outputs into tidy CSVs.

Run A = the earlier, partial pass of the fused sweep (11 sizes, stopped before N=32,760).
Run B = the pass whose log is printed under Sweep A in the notebook (all 14 sizes).
Latencies are the notebook's lat_med (median of 5 timed forwards after 2 warm-ups), in seconds.
"""
import csv, os
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
shape = {104:(1,16,26),390:(1,30,52),1560:(1,60,104),3120:(2,60,104),4680:(3,60,104),6240:(4,60,104),
         9360:(6,60,104),12480:(8,60,104),17160:(11,60,104),21840:(14,60,104),26520:(17,60,104),
         32760:(21,60,104),43680:(28,60,104),56160:(36,60,104)}
mem = {104:0.0133,390:0.0279,1560:0.0830,3120:0.1617,4680:0.2382,6240:0.3136,9360:0.4699,12480:0.6238,
       17160:0.8537,21840:1.0840,26520:1.3159,32760:1.6209,43680:2.1602,56160:2.7763}
runB = {104:0.0810,390:0.0973,1560:0.3405,3120:0.7839,4680:1.2144,6240:1.8215,9360:3.1011,12480:4.8603,
        17160:8.1325,21840:12.4396,26520:17.8332,32760:28.5999,43680:57.0699,56160:100.1684}
runA = {104:0.0519,390:0.0857,1560:0.2986,3120:0.6600,4680:1.0708,6240:1.6667,9360:2.9008,12480:4.5816,
        17160:7.9180,21840:12.3562,26520:17.8531}
math = [(104,0.0701,0.02),(390,0.1383,0.05),(1560,0.6186,0.33),(3120,1.8258,1.19),(4680,3.3727,2.57),
        (6240,5.6142,4.47),(9360,12.8252,9.86),(12480,None,None)]
sameN = [((4,60,104),1.823261,0.313563),((8,30,104),1.822288,0.313563),((16,30,52),1.826118,0.313563)]
rows=[]
for run,d in (("A",runA),("B",runB)):
    for N,l in d.items():
        F,H,W=shape[N]; rows.append(dict(sweep="main",backend="default",run=run,F=F,H=H,W=W,N=N,oom=False,lat_med_s=l,peak_act_gb=mem[N]))
for N,l,m in math:
    F,H,W=shape[N]; rows.append(dict(sweep="main",backend="math",run="B",F=F,H=H,W=W,N=N,oom=l is None,lat_med_s=l if l else "",peak_act_gb=m if m else ""))
for (F,H,W),l,m in sameN:
    rows.append(dict(sweep="sameN",backend="default",run="B",F=F,H=H,W=W,N=F*H*W//4,oom=False,lat_med_s=l,peak_act_gb=m))
with open(os.path.join(OUT,"results.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows),"rows")
