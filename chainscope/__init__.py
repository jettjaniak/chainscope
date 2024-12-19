from importlib.resources import files
from importlib.resources.readers import MultiplexedPath

from beartype.claw import beartype_this_package

beartype_this_package()
_data_dir = files("chainscope.data")
assert isinstance(_data_dir, MultiplexedPath)
DATA_DIR = _data_dir._paths[0]

__version__ = "0.1.0"
