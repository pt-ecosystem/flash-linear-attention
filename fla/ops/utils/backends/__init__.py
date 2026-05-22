# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Utils backends (triton-ascend for cumsum and related helpers)."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.utils.backends.triton_ascend import UtilsTritonAscendBackend

utils_registry = BackendRegistry("utils")
utils_registry.register(UtilsTritonAscendBackend())

__all__ = ["utils_registry", "dispatch"]
