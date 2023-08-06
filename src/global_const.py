from pathlib import Path

# PATHS
CODE_PATH = Path(__file__).parents[1]

CACHE_PATH = CODE_PATH / '_cached_dir'
DATA_PATH = CODE_PATH / 'data'
FINANCE_DATA_PATH = DATA_PATH / 'finance_data'
SYNTHETIC_DATA = DATA_PATH / 'synthetic'

OUTPUT_PATH = CODE_PATH / 'output'
FIGURE_PATH = CODE_PATH / 'figures'
