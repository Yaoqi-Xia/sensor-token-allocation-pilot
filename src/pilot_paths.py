"""Where the pilot reads data and writes results.  Override with environment variables."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get('PILOT_RESULTS', os.path.join(ROOT, 'results'))
MMFI = os.environ.get('MMFI_ROOT', os.path.join(ROOT, 'data', 'MMFi_Dataset'))
