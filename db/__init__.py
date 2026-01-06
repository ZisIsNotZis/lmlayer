from traceback import print_exc
try:
    from .pg import *
except:
    print_exc()
    from .local import *
