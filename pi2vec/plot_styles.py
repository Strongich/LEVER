"""
Publication-quality plotting configuration for academic papers.

This module sets up matplotlib with paper-ready defaults including:
- High-DPI vector and raster output
- Colorblind-friendly palette (Okabe-Ito scheme)
- Clean, minimal styling optimized for single-column figures
- Type 42 fonts for Illustrator compatibility

Usage:
    import plot_metrics  # Automatically applies configuration
    # or
    from plot_metrics import PAPER_STYLE, COLORBLIND_COLORS
"""

import matplotlib as mpl

# Publication-ready matplotlib configuration
PAPER_STYLE = {
    # Figure dimensions optimized for single-column papers
    "figure.figsize": (6.0, 3.6),  # Single-column width
    "figure.dpi": 150,  # Display resolution
    # High-quality output settings
    "savefig.dpi": 300,  # Publication DPI
    "savefig.bbox": "tight",  # Tight bounding box
    "savefig.pad_inches": 0.1,  # Minimal padding
    "savefig.format": "pdf",  # Default vector format
    # Clean, minimal styling
    "axes.grid": True,  # Enable grid
    "grid.alpha": 0.25,  # Subtle grid
    "axes.spines.top": False,  # Remove top spine
    "axes.spines.right": False,  # Remove right spine
    "axes.linewidth": 0.8,  # Thin axes
    "axes.edgecolor": "#333333",  # Dark gray axes
    # Typography optimized for papers
    "axes.labelsize": 11,  # Axis label size
    "axes.titlesize": 12,  # Title size
    "xtick.labelsize": 10,  # X-tick size
    "ytick.labelsize": 10,  # Y-tick size
    "legend.fontsize": 9,  # Legend size
    "legend.frameon": False,  # No legend frame
    "font.family": "sans-serif",  # Font family
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    # Line and marker styling
    "lines.linewidth": 2.0,  # Line thickness
    "lines.markersize": 4,  # Marker size
    "lines.markeredgewidth": 0.5,  # Marker edge thickness
    # Vector output compatibility
    "pdf.fonttype": 42,  # Type 42 fonts (editable in Illustrator)
    "ps.fonttype": 42,  # Type 42 fonts for PostScript
    "svg.fonttype": "none",  # Embed fonts in SVG
}

# Colorblind-friendly palette (Okabe-Ito scheme)
COLORBLIND_COLORS = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion (red-orange)
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
    "#F0E442",  # Yellow
    "#56B4E9",  # Sky blue
    "#E69F00",  # Orange
    "#000000",  # Black
]

# Color cycle for automatic plot coloring
COLOR_CYCLE = mpl.cycler(color=COLORBLIND_COLORS)

# Complete style dictionary with color cycle
COMPLETE_STYLE = {
    **PAPER_STYLE,
    "axes.prop_cycle": COLOR_CYCLE,
}

# Additional plot configurations
PLOT_CONFIGS = {
    "loss": {
        "xlabel": "Training Step",
        "ylabel": "Loss",
        "title": "Training and Validation Loss",
        "yscale": "linear",  # or "log" for log scale
        "grid": True,
        "legend_ncol": 2,
    },
    "accuracy": {
        "xlabel": "Epoch",
        "ylabel": "Accuracy (%)",
        "title": "Training and Validation Accuracy",
        "yscale": "linear",
        "ylim": (0, 100),  # For percentage accuracy
        "grid": True,
        "legend_ncol": 2,
    },
    "learning_rate": {
        "xlabel": "Training Step",
        "ylabel": "Learning Rate",
        "title": "Learning Rate Schedule",
        "yscale": "log",
        "grid": True,
        "legend_ncol": 1,
    },
}

# EMA smoothing configuration
EMA_CONFIG = {
    "alpha": 0.10,  # Smoothing factor (0 < alpha < 1, smaller = more smoothing)
    "apply_to": ["loss"],  # Metrics to smooth
}

# Export settings
EXPORT_CONFIG = {
    "formats": ["pdf", "png"],  # Export formats
    "dpi": 300,  # DPI for raster formats
    "transparent": False,  # Transparent background
    "facecolor": "white",  # Background color
    "edgecolor": "none",  # Edge color
}


def apply_paper_style():
    """Apply the paper-ready matplotlib style."""
    mpl.rcParams.update(COMPLETE_STYLE)


def get_color(index: int) -> str:
    """
    Get a colorblind-friendly color by index.

    Args:
        index: Color index (cycles through available colors)

    Returns:
        Hex color string
    """
    return COLORBLIND_COLORS[index % len(COLORBLIND_COLORS)]


def get_plot_config(metric_name: str) -> dict:
    """
    Get plot configuration for a specific metric.

    Args:
        metric_name: Name of the metric

    Returns:
        Dictionary with plot configuration
    """
    return PLOT_CONFIGS.get(
        metric_name,
        {
            "xlabel": "X",
            "ylabel": "Y",
            "title": f"{metric_name.capitalize()}",
            "grid": True,
            "legend_ncol": 2,
        },
    )


# Automatically apply paper style when module is imported
apply_paper_style()
