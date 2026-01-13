"""
Publication-quality plotting configuration for the project.

This module provides matplotlib settings for paper-ready plots including:
- High-DPI vector and raster output
- Colorblind-friendly palette (Okabe-Ito scheme)
- Clean, minimal styling optimized for single-column figures
- Type 42 fonts for Illustrator compatibility
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


def apply_paper_style(font_scale: float = 1.0):
    """Apply the paper-ready matplotlib style."""
    style = dict(COMPLETE_STYLE)
    if font_scale != 1.0:
        for key in (
            "axes.labelsize",
            "axes.titlesize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
        ):
            style[key] = style[key] * font_scale
    mpl.rcParams.update(style)


def get_color(index: int) -> str:
    """
    Get a colorblind-friendly color by index.

    Args:
        index: Color index (cycles through available colors)

    Returns:
        Hex color string
    """
    return COLORBLIND_COLORS[index % len(COLORBLIND_COLORS)]


# Grid generation specs used by data_generation scripts.
# Add entries here to run additional grid sizes/seeds in batch.
GRID_SPECS = [
    {
        "name": "X1",
        "grid_size": 32,
        "num_golds": 50,
        "num_blocks": 25,
        "hazards": 30,
        "episodes": 30_000,
        "save_every": 1_000,
        "seeds": 5,
        "epsilon_decay": 0.99995,
    },
    {
        "name": "X5",
        "grid_size": 32,
        "num_golds": 50,
        "num_blocks": 25,
        "hazards": 30,
        "episodes": 150_000,
        "save_every": 1_000,
        "seeds": 5,
        "epsilon_decay": 0.99997,  # matches X1 final epsilon over 150k steps
    },
    {
        "name": "X10",
        "grid_size": 32,
        "num_golds": 50,
        "num_blocks": 25,
        "hazards": 30,
        "episodes": 300_000,
        "save_every": 1_000,
        "seeds": 5,
        "epsilon_decay": 0.99999,  # ~same final epsilon as X1 over 300k steps
    },
]

# 6x6 specs scaled from 16x16 densities (50/25/30 over 16x16 area).
GRID_SPECS_6 = [
    {
        "name": "X1",
        "grid_size": 6,
        "num_golds": 7,
        "num_blocks": 4,
        "hazards": 4,
        "episodes": 30_000,
        "save_every": 1_000,
        "seeds": 5,
        "epsilon_decay": 0.99995,
    },
    {
        "name": "X5",
        "grid_size": 6,
        "num_golds": 7,
        "num_blocks": 4,
        "hazards": 4,
        "episodes": 150_000,
        "save_every": 1_000,
        "seeds": 5,
        "epsilon_decay": 0.99997,  # matches X1 final epsilon over 150k steps
    },
    {
        "name": "X10",
        "grid_size": 6,
        "num_golds": 7,
        "num_blocks": 4,
        "hazards": 4,
        "episodes": 300_000,
        "save_every": 1_000,
        "seeds": 5,
        "epsilon_decay": 0.99999,  # ~same final epsilon as X1 over 300k steps
    },
]

# RL query decomposition configuration
GRIDWORLD_AVAILABLE_ACTIONS = [
    "move right",
    "move down",
    "move twice right",
    "move twice down",
]

# Policy sets for query decomposition
TRIVIAL_POLICIES = [
    "Find the shortest path to the exit.",
    "Collect as much gold as possible before exiting.",
    "Activate the lever before reaching the exit.",
    "Reach the exit while avoiding hazards and staying away from them.",
]

DOUBLE_POLICIES = [
    "Find the fastest exit and collect as much gold as possible.",
    "Reach the exit while avoiding hazards.",
    "Activate the lever and avoid hazards while reaching the exit.",
]

TRIPLE_POLICIES = [
    "Find the fastest exit and collect as much gold as possible, while avoiding hazards.",
    "Activate the lever and reach the exit as quickly as possible.",
]

QUERY_DECOMPOSITION_PROMPT = """You are a query decomposition assistant for a Reinforcement Learning application.

Application Context:
- Environment: {application_name}
- Available Actions: {actions}

Available policies in the system:
{policy_list}

Your task is to decompose the user's query about a new and novel task into TRIVIAL tasks, where each sub-query describes only ONE single objective.

Important Guidelines:
- Break down complex tasks into the simplest possible single-objective components
- Each sub-query must represent ONE atomic objective only
- Avoid combining multiple objectives in a single sub-query
- Consider the RL environment context and available actions
- Each sub-query should be actionable for retrieving policies with a single clear goal

Return exactly {expected_count} sub-queries. Each sub-query must be selected verbatim from the policy list.
Ensure all objectives in the user query are covered.
Do not introduce objectives that are not mentioned in the user query.

IMPORTANT: Do not return any other text than the list of sub-queries.
""".strip()
