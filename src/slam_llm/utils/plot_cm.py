import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#"""
# 1. Define the holdback rates (indexes and columns)
rates = ["10", "8.34", "7.14", "6.25", "5.56", "5", "3.34", "2.5", "1.72"]

# 2. Input data for the main matrix
matrix_data = [
    [8.63, 29.95, 31.02, 45.87, 70.66, 99.11, 115.32, 140.89, 212.5],
    [8.7, 8.33, 8.47, 9.99, 12.21, 18.18, 85.45, 96.18, 165.01],
    [9.42, 8.47, 8.53, 8.51, 9.55, 10.88, 26.77, 44.34, 86.18],
    [9.11, 8.56, 8.86, 8.48, 8.44, 8.7, 16.19, 207.33, 503.57],
    [8.98, 8.71, 8.64, 8.5, 8.58, 8.84, 9.54, 22.28, 65.68],
    [11.0, 9.75, 8.8, 8.65, 8.65, 8.92, 9.72, 18.05, 46.21],
    [22.66, 9.87, 9.27, 9.18, 9.08, 9.01, 8.74, 11.06, 35.87],
    [105.48, 56.46, 28.93, 18.34, 12.44, 10.17, 8.89, 9.82, 16.36],
    [107.82, 255.16, 161.25, 85.96, 53.88, 43.16, 26.32, 20.79, 12.33],
]

# 3. Input data for the DYNAMIC baseline row
dynamic_data = [[8.24, 8.28, 8.16, 8.26, 8.27, 8.36, 8.26, 8.69, 10.91]]

# Create DataFrames for text annotations
df_matrix = pd.DataFrame(matrix_data, index=rates, columns=rates)
df_dynamic = pd.DataFrame(dynamic_data, index=["DYNAMIC"], columns=rates)

# 4. Apply the Non-Linear Exponential Decay Transformation for background coloring
# This forces everything except the lowest values to drop off quickly into darkness
vmin_val = 8.16
decay_scale = 3.0

df_matrix_trans = np.exp(-(df_matrix - vmin_val) / decay_scale)
df_dynamic_trans = np.exp(-(df_dynamic - vmin_val) / decay_scale)

# 5. Set up a 2x2 grid layout to align column widths exactly
fig, axs = plt.subplots(
    2, 2, 
    figsize=(10, 8), 
    gridspec_kw={'width_ratios': [15, 1], 'height_ratios': [9, 1]}
)

ax1 = axs[0, 0]      # Main matrix axis
cbar_ax = axs[0, 1]  # Colorbar axis
ax2 = axs[1, 0]      # Dynamic row axis
axs[1, 1].axis('off') # Hide the empty cell below the colorbar

# 6. Plot using a sequential colormap ('viridis' or 'magma') where 1.0 is bright and 0.0 is dark
cmap_choice = "viridis" 

sns.heatmap(
    df_matrix_trans,
    annot=df_matrix,  # Pass the original untransformed values as labels!
    fmt=".2f",
    cmap=cmap_choice,
    vmin=0.0,
    vmax=1.0,
    linewidths=0.5,
    ax=ax1,
    cbar_ax=cbar_ax,
    annot_kws={"size": 14, "weight": "medium"}
)

sns.heatmap(
    df_dynamic_trans,
    annot=df_dynamic,  # Pass the original untransformed values as labels!
    fmt=".2f",
    cmap=cmap_choice,
    vmin=0.0,
    vmax=1.0,
    linewidths=0.5,
    ax=ax2,
    cbar=False,
    annot_kws={"size": 14, "weight": "medium"}
)

# 7. Customize Title and Labels
ax1.set_title("WER ($\\downarrow$) in mismatched conditions", fontsize=18, fontweight="bold", pad=15)
ax1.set_ylabel("Training Token Rate (TPS)", fontweight='bold', fontsize=18)
ax2.set_xlabel("Inference Token Rate (TPS)", fontweight='bold',fontsize=18)
#ax2.set_ylabel("DYNAMIC", fontweight='bold', fontsize=14)
# 3. CHANGED: Increased the font size of numbers next to x-ticks and y-ticks to 15
ax1.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='x', labelsize=16)

# 8. Map colorbar ticks back to display realistic WER labels instead of 0.0 - 1.0
cbar_ticks = [1.0, 0.75, 0.5, 0.25, 0.05]
cbar_labels = [f"{vmin_val - decay_scale * np.log(y):.1f}" if y > 0 else ">20" for y in cbar_ticks]
cbar_ax.set_yticks(cbar_ticks)
cbar_ax.set_yticklabels(cbar_labels, fontsize=16)
# Added arrow explicitly into the side label text
#cbar_ax.set_ylabel("Normalized BLEU (Higher $\\uparrow$ is Better)", fontsize=12, labelpad=12)
ax2.set_yticklabels(["DYNAMIC"], va="center", fontweight='bold', fontsize=12)

# Added arrow explicitly into the side label text
#cbar_ax.set_ylabel("WER (the lower $\\downarrow$ the better)", fontsize=12, labelpad=12)


plt.tight_layout()
plt.show()


"""
# 1. Define the Token Rates 
rates = ["10", "8.34", "7.14", "6.25", "5.56", "5", "3.34", "2.5", "1.49"]

# 2. Input data for the main matrix
matrix_data = [
    [27.72, 24.49, 12.94,  6.44,   4.69,  4.15,  2.23,  1.14,  0.04],
    [29.00, 29.67, 29.17, 27.22,  22.84,  17.72,   5.58,   8.21,   3.35],
    [28.67, 28.87, 29.23, 29.07,  28.59,  28.22,  17.93,  10.08,  2.56],
    [27.55, 28.48, 28.76, 28.85,  29.03,  28.59,  27.70,  23.25,  6.18],
    [25.43, 27.02, 27.15, 27.23,  27.29,  27.04,  26.75,  24.57,  9.98],
    [27.89, 28.22, 28.91, 29.15,  29.42,  29.68,  28.92,  26.80,  8.26],
    [24.90, 25.31, 26.21, 26.79,  27.15,  27.46,  28.31,  25.95,  4.44],
    [ 5.89, 17.01, 21.25, 24.45,  25.48,  26.30,  27.13,  27.05,  16.14],
    [8.38, 11.36, 12.85, 14.37, 14.79, 15.84, 17.95, 19.84,  21.62]
]

# 3. Input data for the DYNAMIC baseline row
dynamic_data = [[30.34, 30.35, 30.26, 30.35, 30.30, 30.31, 30.02, 29.55, 21.33]]

df_matrix = pd.DataFrame(matrix_data, index=rates, columns=rates)
df_dynamic = pd.DataFrame(dynamic_data, index=["DYNAMIC"], columns=rates)

# 4. Non-Linear Exponential Mapping
vmax_val = 30.35
decay_scale = 3.0

df_matrix_trans = np.exp(-(vmax_val - df_matrix) / decay_scale).fillna(0.0)
df_dynamic_trans = np.exp(-(vmax_val - df_dynamic) / decay_scale)

# 5. Set up 2x2 layout for seamless column alignment
fig, axs = plt.subplots(
    2, 2, 
    figsize=(10, 8), 
    gridspec_kw={'width_ratios': [15, 1], 'height_ratios': [9, 1]}
)

ax1 = axs[0, 0]      
cbar_ax = axs[0, 1]  
ax2 = axs[1, 0]      
axs[1, 1].axis('off') 

# 6. Plot heatmaps
sns.heatmap(
    df_matrix_trans, annot=df_matrix, fmt=".2f", cmap="viridis",
    vmin=0.0, vmax=1.0, linewidths=0.5, ax=ax1, cbar_ax=cbar_ax,
    annot_kws={"size": 14, "weight": "medium"}
)

sns.heatmap(
    df_dynamic_trans, annot=df_dynamic, fmt=".2f", cmap="viridis",
    vmin=0.0, vmax=1.0, linewidths=0.5, ax=ax2, cbar=False,
    annot_kws={"size": 14, "weight": "medium"}
)

# --- CHANGES: Added upward arrows (\uparrow) here ---
ax1.set_title("Normalized BLEU ($\\uparrow$) in mismatched conditions", fontsize=18, fontweight="bold", pad=15)
ax1.set_ylabel("Training Token Rate (TPS)", fontweight='bold', fontsize=18)
ax2.set_xlabel("Inference Token Rate (TPS)", fontweight='bold',fontsize=18)
#ax2.set_ylabel("DYNAMIC", fontweight='bold', fontsize=14)
# 3. CHANGED: Increased the font size of numbers next to x-ticks and y-ticks to 15
ax1.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='x', labelsize=16)

# 7. Format Colorbar and append instruction to the label
cbar_ticks = [1.0, 0.75, 0.5, 0.25, 0.05]
cbar_labels = [f"{vmax_val + decay_scale * np.log(y):.1f}" if y > 0 else "<20" for y in cbar_ticks]
cbar_ax.set_yticks(cbar_ticks)
cbar_ax.set_yticklabels(cbar_labels, fontsize=16)
# Added arrow explicitly into the side label text
#cbar_ax.set_ylabel("Normalized BLEU (Higher $\\uparrow$ is Better)", fontsize=12, labelpad=12)
ax2.set_yticklabels(["DYNAMIC"], va="center", fontweight='bold', fontsize=12)

plt.tight_layout()
plt.show()
"""
