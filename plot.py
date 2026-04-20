import numpy as np
import matplotlib.pyplot as plt

FR_system = "TransFace"

methods = np.array(['-', 'FPN + ResNet50', 'SegFormer-B0', 'BiSeNetv2', 'DANet', 'Fast SCNN', 'FCN + MobileNetv2', 'SAM'])

if FR_system == "ElasticFace":
    FR_performance = np.array([96.47, 66.89, 95.82, 93.44, 0.08, 94.69, 7.64, 96.23])
    gen_imp_avg_diff = np.array([0.7127, 0.6778, 0.7010, 0.6795, 0.4974, 0.6863, 0.5889, 0.7041])
elif FR_system == "ArcFace":
    FR_performance = np.array([96.71, 50.47, 96.1, 94.11, 0.05, 95.37, 0.37, 96.49])
    gen_imp_avg_diff = np.array([0.7548, 0.7187, 0.7431, 0.7238, 0.5310, 0.7299, 0.6304, 0.7460])
elif FR_system == "SwinFace":
    FR_performance = np.array([96.71, 85.21, 96.11, 94.67, 3.37, 95.4, 45.46, 96.5])
    gen_imp_avg_diff = np.array([0.7442, 0.7077, 0.7335, 0.7138, 0.5330, 0.7204, 0.6230, 0.7360])
elif FR_system == "TransFace":
    FR_performance = np.array([97.61, 45.98, 97.04, 94, 0.08, 96.37, 1.41, 97.27])
    gen_imp_avg_diff = np.array([0.7341, 0.6847, 0.7182, 0.6989, 0.4838, 0.7053, 0.5944, 0.7202])

FIQA_score = np.array([2.025, 1.903, 2.003, 1.953, 1.399, 1.967, 1.731, 2.014])

metric_names = ['FR Performance', 'Avg Gen-Imp Diff', 'mIoU', 'FIQA Score']
data = {
    'FR Performance': FR_performance,
    'Avg Gen-Imp Diff': gen_imp_avg_diff,
    'FIQA Score': FIQA_score
}

x = np.arange(len(methods))

fig, ax1 = plt.subplots(figsize=(12, 6))

# create additional y-axes
ax2 = ax1.twinx()
ax3 = ax1.twinx()

# move extra axes outward
ax3.spines["right"].set_position(("outward", 60))

# colors
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# plot each metric on its own axis
sc1 = ax1.scatter(x, FR_performance, s=140, color=colors[0], label='FR Performance')
ax1.plot(x, FR_performance, color=colors[0], linewidth=2, linestyle='--')
ax1.set_ylabel('FR Performance', color=colors[0])
ax1.tick_params(axis='y', labelcolor=colors[0])

sc2 = ax2.scatter(x, gen_imp_avg_diff, s=140, color=colors[1], label='$\Delta$')
ax2.plot(x, gen_imp_avg_diff, color=colors[1], linewidth=2, linestyle='--')
ax2.set_ylabel('$\Delta$', color=colors[1])
ax2.tick_params(axis='y', labelcolor=colors[1])

sc3 = ax3.scatter(x, FIQA_score, s=140, color=colors[2], label='FIQA Score')
ax3.plot(x, FIQA_score, color=colors[2], linewidth=2, linestyle='--')
ax3.set_ylabel('FIQA Score', color=colors[2])
ax3.tick_params(axis='y', labelcolor=colors[2])

# x-axis
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=30, ha='right')

# title & grid
ax1.grid(True, linestyle='--', alpha=0.3)

# combined legend
handles = [sc1, sc2, sc3]
labels = [h.get_label() for h in handles]

plt.tight_layout()
plt.savefig('/data/mcaldeir/exit_entry/IJBC/' + FR_system + '.png')