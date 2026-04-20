import numpy as np
import matplotlib.pyplot as plt

FR_system = "TransFace"

methods = np.array(['-', 'FPN + ResNet50', 'SegFormer-B0', 'BiSeNetv2',
                    'DANet', 'Fast SCNN', 'FCN + MobileNetv2', 'SAM'])

th10_MADPromptS = np.array([0.53, 14.05, 8.02, 11.7, 25.91, 12.42, 8.01, 10.23])
th10_SPL = np.array([81.83, 17.07, 23.93, 21.72, 18.96, 18.36, 9.01, 14.01])
th10_MixFaceNet = np.array([74.29, 14.39, 21.71, 21.49, 10.56, 17.48, 5.31, 14.09])

x = np.arange(len(methods))

fig, ax1 = plt.subplots(figsize=(14, 7))

ax2 = ax1.twinx()
ax3 = ax1.twinx()

ax3.spines["right"].set_position(("outward", 60))

colors = ['tab:blue', 'tab:orange', 'tab:green',
          'tab:red', 'tab:purple', 'tab:brown']

sc1 = ax1.scatter(x, th10_MADPromptS, s=120, color=colors[3], label='MADPromptS BPCER@t10%')
ax1.plot(x, th10_MADPromptS, color=colors[3], linestyle='--')
ax1.set_ylabel('MADPromptS BPCER@t10%', color=colors[3])
ax1.tick_params(axis='y', labelcolor=colors[3])

sc2 = ax2.scatter(x, th10_SPL, s=120, color=colors[4], label='SPL BPCER@t10%')
ax2.plot(x, th10_SPL, color=colors[4], linestyle='--')
ax2.set_ylabel('SPL BPCER@t10%', color=colors[4])
ax2.tick_params(axis='y', labelcolor=colors[4])

sc3 = ax3.scatter(x, th10_MixFaceNet, s=120, color=colors[5], label='MixFaceNet-MAD BPCER@t10%')
ax3.plot(x, th10_MixFaceNet, color=colors[5], linestyle='--')
ax3.set_ylabel('MixFaceNet-MAD BPCER@t10%', color=colors[5])
ax3.tick_params(axis='y', labelcolor=colors[5])

ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=30, ha='right')

ax1.grid(True, linestyle='--', alpha=0.3)

handles = [sc1, sc2, sc3]
labels = [h.get_label() for h in handles]

plt.tight_layout()
plt.savefig('/data/mcaldeir/exit_entry/IJBC/MAD_plot.png')