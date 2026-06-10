import os
import matplotlib.pyplot as plt
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))

labels = ['ADJP', 'ADVP', 'NP', 'VP']

precision = [0.80, 0.78, 0.80, 0.87]
recall    = [0.75, 0.74, 0.81, 0.84]
f1        = [0.77, 0.76, 0.80, 0.86]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x,          recall,    width, label='Recall',    color='#4CAF50', alpha=0.85)
bars3 = ax.bar(x + width,  f1,        width, label='F1-Score',  color='#FF9800', alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Chunk Sınıfı', fontsize=12)
ax.set_ylabel('Skor', fontsize=12)
ax.set_title('Chunking Modeli - Sınıf Bazlı Performans (Test Seti)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
output_path = os.path.join(base_path, 'performance_chart.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Grafik kaydedildi → {output_path}")
plt.show()

fig2, ax2 = plt.subplots(figsize=(6, 4))

genel_labels = ['Micro F1', 'Macro F1', 'Weighted F1']
genel_degerler = [0.81, 0.80, 0.81]
renkler = ['#3F51B5', '#E91E63', '#009688']

bars = ax2.bar(genel_labels, genel_degerler, color=renkler, alpha=0.85, width=0.4)

for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=11)

ax2.set_ylim(0, 1.05)
ax2.set_ylabel('F1 Skoru', fontsize=12)
ax2.set_title('Genel F1 Skorları (Test Seti)', fontsize=13)
ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
ax2.set_axisbelow(True)

plt.tight_layout()
output_path2 = os.path.join(base_path, 'genel_f1.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Grafik kaydedildi → {output_path2}")
plt.show()
