"""cg2all.lib — internal library modules.

Alias `cg2all.lib.residue_constants_base` as the top-level name
`residue_constants_base` so `residue_constants.pkl` (which was pickled when
the module lived at the bare name) can still be unpickled.
"""

import sys as _sys

from cg2all.lib import residue_constants_base as _rcb

_sys.modules.setdefault("residue_constants_base", _rcb)
