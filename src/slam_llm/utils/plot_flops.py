import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# token_rate = 10 * (1 - prune_rate/100), so:
# prune 0%    → token_rate 10
# prune 16.67%→ token_rate 8.33
# prune 28.57%→ token_rate 7.14
# prune 37.5% → token_rate 6.25
# prune 44.44%→ token_rate 5.56
# prune 50%   → token_rate 5.0
# prune 66.67%→ token_rate 3.33
# prune 75%   → token_rate 2.5
# prune 82.77%→ token_rate 1.72

prune_pct   = [0, 16.67, 28.57, 37.5, 44.44, 50, 66.67, 75, 82.77]
token_rate  = [10*(1-p/100) for p in prune_pct]  # continuous x values
total_flops = [16538.6, 13793.4, 11887.6, 10426.4, 9295.2, 8382.9, 5717.7, 4395.0, 3184.1]

x = np.array(token_rate)
y = np.array(total_flops)

BG    = 'white'
GRID  = '#cccccc'
COLOR = '#e05c00'

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
ax.set_facecolor(BG)
fig.subplots_adjust(left=0.11, right=0.95, top=0.93, bottom=0.12)

ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle='--', zorder=0)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, linestyle='--', zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor('#aaaaaa')
    spine.set_linewidth(0.8)

# ── Line ──────────────────────────────────────────────────────────────────────
ax.plot(x, y,
    color=COLOR, linewidth=2.5,
    marker='o', markersize=9,
    markerfacecolor=COLOR,
    markeredgecolor='white',
    markeredgewidth=2.0,
    linestyle='-', zorder=3,
    label='Total FLOPs')

# ── Annotations ───────────────────────────────────────────────────────────────
# first point (token_rate=10, no pruning)
ax.annotate(f'{y[0]:,.0f}G',
    xy=(x[0], y[0]), xytext=(x[0]-1.0, y[0] + 1400),
    fontsize=16, fontweight='bold', color=COLOR,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor=COLOR, linewidth=1.2))

# last point (token_rate~1.72, most pruning)
ax.annotate(f'{y[-1]:,.0f}G\n(5.2x ↓)',
    xy=(x[-1], y[-1]), xytext=(x[-1]-0.8, y[-1] + 3000),
    fontsize=16, fontweight='bold', color=COLOR,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor=COLOR, linewidth=1.2),
    arrowprops=dict(arrowstyle='->', color=COLOR, lw=1.2,
                    connectionstyle='arc3,rad=-0.2'))

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0, 11)
ax.set_ylim(0, 20000)
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], fontsize=14)
ax.set_yticks([0, 5000, 10000, 15000, 20000])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax.tick_params(axis='both', colors='#444444', labelsize=14, length=4)
ax.set_xlabel('Token Rate (TPS)', fontsize=14, fontweight='bold', labelpad=8)
ax.set_ylabel('GFLOPs', fontsize=15, fontweight='bold', labelpad=8)

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(fontsize=16, frameon=True, facecolor='white',
          edgecolor='#cccccc', loc='upper left',
          handlelength=2.0, borderpad=0.6)

# ── Save ──────────────────────────────────────────────────────────────────────
#plt.savefig('/mnt/user-data/outputs/total_flops_tokenrate.pdf',
#    format='pdf', bbox_inches='tight', facecolor=BG)
#plt.savefig('/mnt/user-data/outputs/total_flops_tokenrate.png',
#    dpi=180, bbox_inches='tight', facecolor=BG)
#print("Token rates:", [round(t,2) for t in token_rate])
#print("Saved.")
