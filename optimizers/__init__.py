"""
optimizers package

Importing this package automatically registers all optimizers
with the optimizer_registry. To add a new optimizer:
    1. Create optimizers/new_optimizer.py
    2. Add an import line below

No other framework component needs modification.
"""

from optimizers.gwo import GWO           # noqa: F401
from optimizers.bbgwo import BBGWO       # noqa: F401
from optimizers.regwo import REGWO       # noqa: F401
from optimizers.mengwo import MENGWO     # noqa: F401
from optimizers.mgwo import MGWO        # noqa: F401
from optimizers.rwgwo import RWGWO       # noqa: F401
from optimizers.obgwo import OBGWO       # noqa: F401
from optimizers.modgwo import modGWO     # noqa: F401
from optimizers.ebgwo import EBGWO       # noqa: F401
from optimizers.igwo_ms import IGWO_MS   # noqa: F401
from optimizers.agwo import AGWO         # noqa: F401
