from importlib.resources import files

from beartype.claw import beartype_this_package

beartype_this_package()
_data_dir = files("chainscope.data")
DATA_DIR = _data_dir._paths[0]  # type: ignore

__version__ = "0.1.0"
