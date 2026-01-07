"""Database backend selection helper.

Attempts to import the Postgres-backed implementation; on failure it falls back
to the in-memory `local` implementation. This module exposes a common interface
(`init`, `getUser`, `addMessages`, etc.) for use by the rest of the application.
"""

from traceback import print_exc
try:
    from .pg import *
except:
    print_exc()
    from .local import *
