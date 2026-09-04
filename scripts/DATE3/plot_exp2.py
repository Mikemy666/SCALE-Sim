import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.DATE3.plot_experiments import plot_exp2
if __name__ == "__main__": plot_exp2()
