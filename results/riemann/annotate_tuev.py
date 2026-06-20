"""Add a 'decodable != clustered' banner under the TUEV manifold figure (local, no cluster)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(__file__)
src = os.path.join(HERE, "riemann_latent_tuev.png")
dst = os.path.join(HERE, "riemann_latent_tuev_annot.png")
img = mpimg.imread(src)
h, w = img.shape[0], img.shape[1]
dpi = 150
band = 96  # px reserved for the banner
fig = plt.figure(figsize=(w / dpi, (h + band) / dpi), dpi=dpi)
ax = fig.add_axes([0, band / (h + band), 1, h / (h + band)])
ax.imshow(img); ax.axis("off")
fig.text(0.5, 0.5 * band / (h + band),
         "DECODABLE (frozen probe BA ~0.40, >> chance 0.17; ours > random) but NOT CLUSTERED "
         "(silhouette ~0)\n"
         "transient sub-second events diluted by the 10 s window -- silhouette measures "
         "clustering, not decodability",
         ha="center", va="center", fontsize=8.5, color="#c0392b", weight="bold")
fig.savefig(dst, dpi=dpi)
print("saved", dst)
