import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
P_sizes = [8,16,32,64]
img_sizes = np.linspace(128, 1024, 40)
d = 768; L = 12
plt.figure(figsize=(5.6,4.6))
for P in P_sizes:
    N = (img_sizes/P)**2
    flops = L * (N**2 * d)
    plt.plot(img_sizes, flops/1e9, label=f"patch size P={P}")
plt.yscale("log")
plt.xlabel("Input image side length (pixels)")
plt.ylabel("Analytic self-attention FLOPs / image (x1e9, log scale)")
plt.title("Analytic HGAViT self-attention cost vs. image size\n(closed-form from Sec. 3.11 Eq. 39; not a measured runtime)")
plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figs/p1_new_complexity.png"); plt.close()
print("complexity done")
