"""Render the Faraz Shayari AWS DEPLOYMENT / infrastructure diagram to a PNG."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK="#0f172a"; MUTE="#475569"
CODE=("#dbeafe","#2563eb"); DATA=("#fef3c7","#d97706"); AWS=("#ffedd5","#ea580c")
USER=("#ede9fe","#7c3aed"); OUT=("#d1fae5","#059669"); NEU=("#eef2f7","#64748b")

fig, ax = plt.subplots(figsize=(17, 11), dpi=200)
ax.set_xlim(0,17); ax.set_ylim(0,11); ax.axis("off")

def box(x,y,w,h,title,sub,colors,tsize=11,ssize=8.3,talign="center"):
    fc,ec=colors
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.09",
        linewidth=1.6,edgecolor=ec,facecolor=fc,zorder=4))
    cx=x+w/2
    ax.text(cx,y+h-0.26,title,ha="center",va="center",fontsize=tsize,fontweight="bold",color=INK,zorder=6)
    if sub: ax.text(cx,y+0.26 if "\n" not in sub else y+0.34,sub,ha="center",va="center",fontsize=ssize,color=MUTE,zorder=6)
    return {"x":x,"y":y,"w":w,"h":h,"cx":cx,"cy":y+h/2}

def zone(x,y,w,h,label,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.5,edgecolor=ec,facecolor=fc,zorder=1))
    ax.text(x+0.22,y+h-0.28,label,ha="left",va="center",fontsize=12,fontweight="bold",color=ec,zorder=2)

def arrow(a,sa,b,sb,color="#334155",ls="-",lw=1.9,rad=0.0,label=None,dy=0.16):
    pa={"R":(a["x"]+a["w"],a["cy"]),"L":(a["x"],a["cy"]),"T":(a["cx"],a["y"]+a["h"]),"B":(a["cx"],a["y"])}
    pb={"R":(b["x"]+b["w"],b["cy"]),"L":(b["x"],b["cy"]),"T":(b["cx"],b["y"]+b["h"]),"B":(b["cx"],b["y"])}
    x1,y1=pa[sa]; x2,y2=pb[sb]
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",mutation_scale=15,linewidth=lw,color=color,linestyle=ls,zorder=3))
    if label: ax.text((x1+x2)/2,(y1+y2)/2+dy,label,ha="center",fontsize=7.8,color=MUTE,style="italic",zorder=7)

# ---- title ----
ax.text(8.5,10.62,"Faraz Shayari — AWS Deployment Architecture",ha="center",fontsize=20,fontweight="bold",color=INK)
ax.text(8.5,10.2,"how the app is built on a laptop, shipped as a container, and run on ECS Fargate",
        ha="center",fontsize=10.5,color=MUTE)

# ================= DEPLOYER (laptop) =================
zone(0.4,1.6,4.2,8.0,"①  DEPLOYER · your laptop","#faf7ff","#7c3aed")
l1=box(0.7,8.1,3.6,0.95,"Source + data","app.py · generate.py · index files",CODE,10.5)
l2=box(0.7,6.95,3.6,0.95,"docker build","Dockerfile → image",CODE,10.5)
l3=box(0.7,5.8,3.6,0.95,"faraz-shayari:local","the built container image",DATA,10.5)
l4=box(0.7,4.35,3.6,0.95,"deploy.py (boto3)","creates all AWS resources",CODE,10.5)
l5=box(0.7,2.9,3.6,0.95,".env — IAM user keys","the DEPLOYER identity",USER,10.5)
arrow(l1,"B",l2,"T"); arrow(l2,"B",l3,"T")

# ================= AWS CLOUD =================
zone(5.0,0.55,11.7,9.05,"②  AWS Cloud · region eu-north-1","#fffaf4","#ea580c")

ecr =box(5.35,7.95,3.3,1.05,"Amazon ECR","registry · faraz-shayari:latest",AWS,10.8)
iam =box(8.95,7.95,3.4,1.05,"IAM roles","farazTaskRole · farazExecutionRole",AWS,10.8)
bed =box(12.7,6.55,3.7,2.45,"Amazon Bedrock","(managed AI service)",AWS,11.5)
ax.text(bed["cx"],bed["cy"]+0.15,"Titan Embed v2  (embeddings)\nNova Pro  (generation)",ha="center",va="center",fontsize=9,color=INK,zorder=6)
cw  =box(12.7,4.6,3.7,1.15,"CloudWatch Logs","/ecs/faraz-shayari",AWS,10.8)

# ECS Fargate nested: cluster -> service -> task(container)
zone(5.35,1.15,6.9,5.75,"ECS Fargate — cluster: faraz-cluster","#f2f9ff","#2563eb")
ax.add_patch(FancyBboxPatch((5.7,1.5),6.2,4.55,boxstyle="round,pad=0.02,rounding_size=0.05",
    linewidth=1.3,edgecolor="#38bdf8",facecolor="#eaf6ff",zorder=2))
ax.text(5.9,5.82,"Service: faraz-shayari-svc  (desired = 1, keeps it alive)",ha="left",fontsize=9.5,fontweight="bold",color="#0369a1",zorder=3)
# security group frame
ax.add_patch(FancyBboxPatch((6.0,1.9),5.6,3.55,boxstyle="round,pad=0.02,rounding_size=0.04",
    linewidth=1.4,edgecolor="#16a34a",facecolor="#f2fdf6",linestyle=(0,(4,2)),zorder=2))
ax.text(6.15,5.24,"Security Group faraz-sg · inbound TCP 8000",ha="left",fontsize=8.6,color="#15803d",zorder=3)
task=box(6.3,2.2,5.0,2.85,"Task (Fargate) — the running container","",NEU,11)
ax.text(task["cx"],4.25,"python:3.11-slim",ha="center",fontsize=8.5,color=MUTE,style="italic",zorder=6)
ax.text(task["cx"],3.75,"uvicorn + FastAPI  ·  :8000",ha="center",fontsize=10,fontweight="bold",color="#2563eb",zorder=6)
ax.text(task["cx"],3.30,"app.py  →  retrieve  +  generate",ha="center",fontsize=9.2,color=INK,zorder=6)
ax.text(task["cx"],2.88,"baked-in index: couplets.jsonl + embeddings.npy",ha="center",fontsize=8.6,color=MUTE,zorder=6)
ax.text(task["cx"],2.48,"NO AWS keys — creds come from the task role",ha="center",fontsize=8.4,color="#b91c1c",zorder=6)

# ================= USER =================
usr=box(6.55,9.05,4.5,0.95,"Internet · User / Browser","types a life situation",USER,11)

# ---- arrows ----
arrow(l3,"R",ecr,"L",color="#d97706",label="docker push",rad=-0.05)          # image -> ECR
arrow(l4,"R",task,"L",color="#7c3aed",ls=(0,(4,3)),rad=-0.18,label="provisions (boto3)")  # deploy -> ecs
arrow(ecr,"B",task,"T",color="#334155",rad=0.12,label="execution role pulls image")       # ECR -> task
arrow(task,"R",cw,"L",color="#334155",rad=0.10,label="stdout → logs")                      # task -> logs
arrow(task,"R",bed,"L",color="#ea580c",lw=2.2,rad=-0.05,label="task role → InvokeModel")   # task -> bedrock
arrow(usr,"B",task,"T",color="#16a34a",lw=2.1,rad=0.0,label="HTTP :8000")                   # user -> task
arrow(iam,"B",task,"T",color="#94a3b8",ls=(0,(2,2)),rad=-0.15,label="task assumes roles")   # iam -> task

# ---- legend ----
leg=[("Our code/build",CODE),("Image / data",DATA),("AWS managed",AWS),("Identity/User",USER),("Container",NEU)]
lx=0.6
for name,(fc,ec) in leg:
    ax.add_patch(FancyBboxPatch((lx,0.2),0.32,0.26,boxstyle="round,pad=0.01,rounding_size=0.05",
        linewidth=1.4,edgecolor=ec,facecolor=fc,zorder=3))
    ax.text(lx+0.42,0.33,name,ha="left",va="center",fontsize=9,color=INK)
    lx+=0.42+len(name)*0.108+0.55

OUTP=Path(__file__).parent/"faraz_shayari_deployment.png"
fig.savefig(OUTP,bbox_inches="tight",facecolor="white",dpi=200)
print(f"saved {OUTP}")
