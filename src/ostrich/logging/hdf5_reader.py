import os
from typing import Any
from typing import List
from typing import Union

import h5py
import numpy as np


class HDF5Reader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file = None
        self.open()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Log file not found: {self.filepath}")
        if self._file is None:
            self._file = h5py.File(self.filepath, "r")

    def close(self):
        if self._file is not None:
            self._file.close()

    def get_dataset(self, path: str) -> np.ndarray:
        try:
            return self._file[path][()]
        except KeyError:
            raise KeyError(f"Dataset not found at path: {path}")

    def get_scalar(self, path: str) -> Union[int, float, str]:
        """Retrieves a scalar dataset and returns it as a native Python type."""
        scalar = self.get_dataset(path)
        if isinstance(scalar, np.ndarray):
            if scalar.size != 1:
                raise ValueError(f"Dataset at '{path}' is not a scalar (size={scalar.size}).")
            return scalar.item()
        return scalar

    def get_attribute(self, path: str, attr_name: str) -> Any:
        """Retrieves an attribute from a group or dataset."""
        try:
            return self._file[path].attrs[attr_name]
        except KeyError:
            raise KeyError(f"Object '{path}' or attribute '{attr_name}' not found.")

    def list_groups(self, path: str = "/") -> List[str]:
        return [key for key, val in self._file[path].items() if isinstance(val, h5py.Group)]

    def list_datasets(self, path: str = "/") -> List[str]:
        return [key for key, val in self._file[path].items() if isinstance(val, h5py.Dataset)]

    def list_attributes(self, path: str = "/") -> List[str]:
        """Lists all attribute names on a given group or dataset."""
        return list(self._file[path].attrs.keys())

    def print_tree(self):
        print(f"File Tree for {self.filepath}:")
        self._file.visititems(self._print_node)

    def _print_node(self, name: str, obj: h5py.HLObject):
        depth = name.count("/")
        indent = "    " * depth
        base_name = os.path.basename(name) if name else "/"

        if isinstance(obj, h5py.Dataset):
            print(f"{indent}├── {base_name} (Dataset: shape={obj.shape}, dtype={obj.dtype})")
        else:  # It's a group
            print(f"{indent}└── {base_name} (Group)")

        # Print attributes for this object
        if obj.attrs:
            attr_indent = indent + "    " + "|"
            for key, val in obj.attrs.items():
                print(f"{attr_indent}  @{key}: {val}")
