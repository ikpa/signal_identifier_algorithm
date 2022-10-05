# -*- coding: utf-8 -*-

_use_disk_cache = 'coils_cache' # subdir name prefix or False for no caching
def get_cache_dir():
    import tempfile
    import pathlib
    import os

    return (pathlib.Path(tempfile.gettempdir()) /
            (_use_disk_cache + '_' + os.environ.get('USER', 'anonymous')))

verbose = 1  # 1 is not full verbosity
