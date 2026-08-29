"""imbalance_seq -- the imbalance sequence delta(p,q) = (q-p)/(q+p), 1 <= p < q.

See core.py for the mathematics.  Everything in core's __all__ is re-exported
here, so `from imbalance_seq import entry_at, novel_index, ...` works.
"""

from .core import *          # noqa: F401,F403
from .core import __all__ as _core_all

__all__ = list(_core_all)
__version__ = "0.1.0"
