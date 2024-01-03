#credit: (https://stackoverflow.com/questions/15506971/recursive-version-of-reload)

import sys
import importlib
from types import ModuleType

def deep_reload(m):
    # print(m,ModuleType)
    name = m.__name__  # get the name that is used in sys.modules
    name_ext = name + '.'  # support finding sub modules or packages

    def compare(loaded: str):
        return (loaded == name) or loaded.startswith(name_ext)

    all_mods = tuple(sys.modules)  # prevent changing iterable while iterating over it
    sub_mods = filter(compare, all_mods)
    for pkg in sorted(sub_mods, key=lambda item: item.count('.'), reverse=True):
        importlib.reload(sys.modules[pkg])  # reload packages, beginning with the most deeply nested