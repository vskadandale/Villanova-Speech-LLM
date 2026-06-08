import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import EventCollection
"""
# # # RESULTS FOR SER x IEMOCAP x FROZEN WHISPER # # #
# CTC
uwa_ctc_test = [61.80, 59.46, 60.05, 62.28, 63.44, 68.74]
thr_pruned_ctc_test = [5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Cosine Distance
uwa_cos_test = [50.40, 55.31, 59.77, 62.07, 63.51, 61.57, 63.92, 65.48, 65.80, 68.74]
thr_pruned_cos_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Downsampling Rate (Fixed: added 8.34 to match uwa_ds_test length)
uwa_ds_test = [53.69, 57.35, 57.86, 61.85, 65.50, 66.17, 67.79, 64.79, 64.96, 68.74]
pruned_ds_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Random
uwa_random_test = [63.21, 64.66, 67.32, 67.58, 68.43, 69.45, 69.94, 70.56, 69.18, 70.11]
pruned_random_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# AvgPool
uwa_avgpool_test = [67.37, 68.84, 68.01, 68.83, 70.42, 68.99, 67.66, 70.34, 68.55, 68.85]
pruned_avgpool_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6)) # Increased width for better readability
ax.plot(thr_pruned_ctc_test, uwa_ctc_test, color='red', marker='o', label='CTC')
ax.plot(thr_pruned_cos_test, uwa_cos_test, color='blue', marker='x', linestyle='--', label='COSD')
ax.plot(pruned_ds_test, uwa_ds_test, color='green', marker='^', linestyle='-.', label='DS')
ax.plot(pruned_random_test, uwa_random_test, color='orange', marker='s', linestyle=':', label='Random')
ax.plot(pruned_avgpool_test, uwa_avgpool_test, color='purple', marker='d', linestyle='-', label='AvgPool')

# Formatting
ax.set_xlabel('Token Rate (TPS)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(r'$\mathbf{Unweighted\ Accuracy\ (UWA\%)}$', fontsize=15, labelpad=10)
ax.set_title('%UWA ($\\uparrow$) for ER with Whisper CTC (frozen backbone)', fontsize=16, fontweight='bold', pad=20)

ticks = np.arange(0, 11, 1.0) # From 0 to 10 inclusive with 0.5 step
ax.set_xticks(ticks)

ax.legend(fontsize=12, loc='best')
ax.grid(True, linestyle='--', alpha=0.7)


plt.tight_layout()
plt.show()

# # # RESULTS FOR ASR x VoxPopuli Test x FT WHISPER # # #

# CTC
wer_ctc_test = [12.33, 9.82, 8.74, 8.92, 8.58, 8.48, 8.53, 8.33, 8.63]
thr_pruned_ctc_test = [1.7229, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Cosine Distance
wer_cos_test = [47.82, 22.27, 11.71, 10.03, 9.68, 8.61, 8.79, 8.93, 8.46, 8.63]
thr_pruned_cos_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Downsampling Rate (Fixed: added 8.34 to match wer_ds_test length)
wer_ds_test = [62.53, 19.42, 11.23, 9.75, 10.16, 8.81, 8.82, 8.79, 8.73, 8.63]
pruned_ds_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Random
wer_random_test = [105.29, 65.96, 30.06, 23.43, 16.29, 14.78, 14.07, 12.31, 11.61, 10.63]
pruned_random_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# AvgPool
wer_avgpool_test = [67.46, 22.08, 10.50, 9.45, 8.79, 8.75, 9.03, 8.65, 8.43, 8.21]
pruned_avgpool_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Map methods to visual styles
methods = {
    'CTC': (thr_pruned_ctc_test, wer_ctc_test, 'red', 'o', '-'),
    'COSD': (thr_pruned_cos_test, wer_cos_test, 'blue', 'x', '--'),
    'DS': (pruned_ds_test, wer_ds_test, 'green', '^', '-.'),
    'Random': (pruned_random_test, wer_random_test, 'orange', 's', ':'),
    'AvgPool': (pruned_avgpool_test, wer_avgpool_test, 'purple', 'd', '-')
}

# --- Plotting ---
#fig, ax = plt.subplots(figsize=(14, 9))
fig, ax = plt.subplots(figsize=(8, 6))

# 1. Main Plot (Global context)
for name, (x, y, color, marker, style) in methods.items():
    ax.plot(x, y, label=name, marker=marker, color=color, linestyle=style,
            linewidth=2, markersize=8, alpha=0.9)

# Configure Main Axis ticks and limits
ax.set_xlim(0, 10.5)
ax.xaxis.set_major_locator(MultipleLocator(1)) # frequency of 1 on global plot

ax.set_xlabel('Token Rate (TPS)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(r'$\mathbf{Word\ Error\ Rate\ (WER\%)}$', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('%WER ($\\downarrow$) with fine-tuned Whisper CTC', fontsize=18, fontweight='bold', pad=20)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)

# 2. Zoomed Inset (Specs: x=0-10.5, y=8-15, xtick_freq=1)
ax_inset = ax.inset_axes([0.3, 0.42, 0.65, 0.53])

for name, (x, y, color, marker, style) in methods.items():
    ax_inset.plot(x, y, marker=marker, color=color, linestyle=style,
                  linewidth=2, markersize=6, alpha=0.9)

# Configure Inset axes
ax_inset.set_xlim(0.9, 5.5)
ax_inset.set_ylim(6.5, 25)
ax_inset.xaxis.set_major_locator(MultipleLocator(1)) # Ticks every 1 unit

ax_inset.grid(which='major', axis='both', color='#999999', linestyle='-', alpha=0.4)
#ax_inset.set_title('Zoom: $WER$ 8-15% (Full Token Range)', fontsize=11, fontweight='bold', backgroundcolor='white')

# 3. Add indicators (lines/box) to connect the main plot and the inset
ax.indicate_inset_zoom(ax_inset, edgecolor="black", alpha=0.3)

plt.tight_layout()
plt.show()
"""
# # # RESULTS FOR TopicID x SIBFLEURS x FROZEN WHISPER # # #

# CTC
uwa_ctc_test = [71.42, 75.22, 73.70, 76.42, 75.51]
thr_pruned_ctc_test = [5.56, 6.25, 7.143, 8.34, 10.0]

# Cosine Distance
uwa_cos_test = [33.86, 64.11, 66.90, 68.65, 69.37, 68.12, 75.51]
thr_pruned_cos_test = [0.5, 1.0, 2.5, 6.25, 7.143, 8.34, 10.0]

# Downsampling Rate (Fixed: added 8.34 to match wer_ds_test length)
uwa_ds_test = [58.04, 37.14, 68.13, 65.49, 67.83, 68.71, 70.55, 69.63, 72.34, 75.51]
pruned_ds_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Random
uwa_random_test = [24.64, 52.48, 68.30, 72.24, 74.27, 74.25, 71.81, 76.47, 76.88, 74.63]
pruned_random_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# AvgPool
uwa_avgpool_test = [62.21, 66.84, 67.91, 66.29, 75.62, 71.31, 71.83, 76.32, 76.16, 77.89]
pruned_avgpool_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6)) # Increased width for better readability

ax.plot(thr_pruned_ctc_test, uwa_ctc_test, color='red', marker='o', label='CTC')
ax.plot(thr_pruned_cos_test, uwa_cos_test, color='blue', marker='x', linestyle='--', label='COSD')
ax.plot(pruned_ds_test, uwa_ds_test, color='green', marker='^', linestyle='-.', label='DS')
ax.plot(pruned_random_test, uwa_random_test, color='orange', marker='s', linestyle=':', label='Random')
ax.plot(pruned_avgpool_test, uwa_avgpool_test, color='purple', marker='d', linestyle='-', label='AvgPool')

# Formatting
ax.set_xlabel('Token Rate (TPS)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(r'$\mathbf{Unweighted\ Accuracy\ (UWA\%)}$', fontsize=15, labelpad=10)
ax.set_title('%UWA ($\\uparrow$) for TD with Whisper CTC (frozen backbone)', fontsize=16, fontweight='bold', pad=20)

ticks = np.arange(0, 11, 1.0) # From 0 to 10 inclusive with 0.5 step
ax.set_xticks(ticks)

ax.legend(fontsize=12, loc='best')
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
"""
# # # RESULTS FOR SER x IEMOCAP x FINETUNED WHISPER # # #
# CTC
uwa_ctc_test = [37.90, 36.26, 41.14, 45.60, 39.28, 44.84, 44.04, 46.90, 46.04, 50.97]
thr_pruned_ctc_test = [0.638, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Cosine Distance
uwa_cos_test = [31.06, 37.37, 43.01, 43.88, 41.87, 47.49, 47.51, 45.17, 54.70, 50.97]
thr_pruned_cos_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Downsampling Rate (Fixed: added 8.34 to match uwa_ds_test length)
uwa_ds_test = [45.68, 46.76, 43.01, 42.04, 45.51, 48.94, 48.99, 44.80, 48.93, 50.97]
pruned_ds_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Random
uwa_random_test = [33.63, 33.34, 40.74, 48.16, 48.55, 44.45, 44.52, 49.73, 49.43, 49.89]
pruned_random_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# AvgPool
uwa_avgpool_test = [53.07, 49.41, 51.49, 49.56, 53.54, 52.05, 49.84, 51.82, 51.84, 53.64]
pruned_avgpool_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6)) # Increased width for better readability
ax.grid(True, linestyle='--', alpha=0.7)

ax.plot(thr_pruned_ctc_test, uwa_ctc_test, color='red', marker='o', label='CTC')
ax.plot(thr_pruned_cos_test, uwa_cos_test, color='blue', marker='x', linestyle='--', label='COSD')
ax.plot(pruned_ds_test, uwa_ds_test, color='green', marker='^', linestyle='-.', label='DS')
ax.plot(pruned_random_test, uwa_random_test, color='orange', marker='s', linestyle=':', label='Random')
ax.plot(pruned_avgpool_test, uwa_avgpool_test, color='purple', marker='d', linestyle='-', label='AvgPool')

# Formatting
ax.set_xlabel('Token Rate (TPS)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(r'$\mathbf{Unweighted\ Accuracy\ (UWA\%)}$', fontsize=15, labelpad=10)
ax.set_title('%UWA ($\\uparrow$) for ER with fine-tuned Whisper CTC', fontsize=16, fontweight='bold', pad=20)

ticks = np.arange(0, 11, 1.0) # From 0 to 10 inclusive with 0.5 step
ax.set_xticks(ticks)

ax.legend(fontsize=12, loc='best')
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# # # RESULTS FOR ST x EUROPARL x FT WHISPER # # #
# CTC
bleu_ctc_test = [21.62, 27.05, 28.31, 29.68, 27.29, 28.85, 29.23, 29.67, 27.72]
thr_pruned_ctc_test = [1.4936, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Cosine Distance
bleu_cos_test = [1.70, 1.70, 14.59, 22.86, 26.68, 26.58, 28.52, 28.01, 28.51, 27.72]
thr_pruned_cos_test = [0.5, 1.0, 2.5, 3.34, 5.0, 5.56, 6.25, 7.143, 8.34, 10.0]

# Downsampling Rate (Fixed: added 8.34 to match uwa_ds_test length)
bleu_ds_test = [1.58, 1.43, 1.45, 13.17, 21.20, 21.43, 24.71 , 25.09, 26.20, 27.72]
pruned_ds_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# Random
bleu_random_test = [1.31, 1.35, 1.07, 1.31, 5.86, 10.17, 8.13, 15.05, 13.78, 23.02]
pruned_random_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# AvgPool
bleu_avgpool_test = [1.53, 1.60, 13.95, 20.44, 22.27, 25.99, 27.63, 28.37, 28.16, 29.71]
pruned_avgpool_test = [0.5, 1.0, 2.5, 3.34, 5, 5.56, 6.25, 7.143, 8.34, 10.0]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6)) # Increased width for better readability
ax.grid(True, linestyle='--', alpha=0.7)

ax.plot(thr_pruned_ctc_test, bleu_ctc_test, color='red', marker='o', label='CTC')
ax.plot(thr_pruned_cos_test, bleu_cos_test, color='blue', marker='x', linestyle='--', label='COSD')
ax.plot(pruned_ds_test, bleu_ds_test, color='green', marker='^', linestyle='-.', label='DS')
ax.plot(pruned_random_test, bleu_random_test, color='orange', marker='s', linestyle=':', label='Random')
ax.plot(pruned_avgpool_test, bleu_avgpool_test, color='purple', marker='d', linestyle='-', label='AvgPool')

# Formatting
ax.set_xlabel('Token Rate (TPS)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('% Normalized BLEU', fontsize=15, fontweight='bold', labelpad=10)
ax.set_title('% Normalized BLEU ($\\uparrow$) with fine-tuned Whisper CTC', fontsize=17, fontweight='bold', pad=20)

ticks = np.arange(0, 11, 1.0) # From 0 to 10 inclusive with 0.5 step
ax.set_xticks(ticks)

ax.legend(fontsize=12, loc='best')

plt.tight_layout()
plt.show()


# # # RESULTS FOR ASR x voxpopuli-test x wav2vec2 # # #

fig, ax = plt.subplots(figsize=(8, 6)) # Increased width for better readability
ax.grid(True, linestyle='--', alpha=0.7)

wer_ctc_test=[12.84,12.59,11.66,11.14,10.94,10.88,11.46,10.65,10.52,10.54]
thr_pruned_ctc_test=[3.5,4.1,4.6,5.2,5.7,6.1,6.9,8.3,9.4,10.0]

wer_cos_test=[39.78,22.10,12.80, 11.02, 10.75, 10.74, 10.24, 10.54]  
thr_pruned_cos_test=[4.0, 4.6, 5.5, 6.4, 7.0, 8.3, 9.5, 10.0]

wer_avgpool_test=[28.16,18.14,13.92,11.14,10.22,9.44]
pruned_avgpool_test=[2.0,2.5,3.3,5.0,6.25,10.0]

wer_ds_test=[10.54,11.59, 12.81,12.87,20.28,34.36,65.68]
pruned_ds_test=[10.0,7.2,5.6,5.00,3.3,2.5,2.0]

wer_avgpool_test=[46.37, 25.87, 17.33, 12.91, 11.45, 10.83]
pruned_avgpool_test=[2.0,2.5,3.3,5.0,6.25,10.0]

wer_random_test=[100.84, 60.01, 46.46, 27.51]
pruned_random_test=[2.5, 5.0, 6.25, 10.0]


ax.plot(thr_pruned_ctc_test, wer_ctc_test, color='red', marker='o', label='CTC')
ax.plot(thr_pruned_cos_test, wer_cos_test, color='blue', marker='x', linestyle='--', label='COSD')
ax.plot(pruned_ds_test, wer_ds_test, color='green', marker='^', linestyle='-.', label='DS')
ax.plot(pruned_random_test, wer_random_test, color='orange', marker='s', linestyle=':', label='Random')
ax.plot(pruned_avgpool_test, wer_avgpool_test, color='purple', marker='d', linestyle='-', label='AvgPool')

# Formatting
ax.set_xlabel('Token Rate (TPS)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel(r'$\mathbf{Word\ Error\ Rate\ (WER\%)}$', fontsize=15, labelpad=10)
ax.set_title('%WER ($\\downarrow$) with Wav2Vec2', fontsize=18, fontweight='bold', pad=20)

ticks = np.arange(2, 11, 1.0) # From 0 to 10 inclusive with 0.5 step
ax.set_xticks(ticks)

ax.legend(fontsize=12, loc='best')

plt.tight_layout()
plt.show()
"""
