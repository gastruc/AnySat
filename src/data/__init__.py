import sys
from pathlib import Path

# Use absolute path instead of relative import
repo_root = Path(__file__).parent.parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.append(str(repo_root / "src"))