import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
import warnings

warnings.filterwarnings('ignore')
set_matplotlib_formats('pdf', 'png')

plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14

plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = "DejaVu Serif"
plt.rcParams['font.family'] = "serif"

# Additional configurations for grey background and grid
plt.rcParams['axes.facecolor'] = '#E5E5E5'  # Light grey background
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#FFFFFF'  # White grid lines
plt.rcParams['grid.linestyle'] = '-'    # Solid grid lines
plt.rcParams['grid.linewidth'] = 0.5    # Thin grid lines
