from importlib import import_module
from os.path import basename, dirname, isfile, join
import glob


modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = sorted([
    basename(f)[:-3]
    for f in modules
    if isfile(f) and not f.endswith("__init__.py")
])
for module_name in __all__:
    import_module(f"{__name__}.{module_name}")
