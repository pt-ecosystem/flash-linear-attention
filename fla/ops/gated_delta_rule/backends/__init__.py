# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Gated delta rule backends (registry; triton-ascend / tilelang live in subpackages)."""

from fla.ops.backends import BackendRegistry
from fla.ops.gated_delta_rule.backends.triton_ascend import GatedDeltaRuleTritonAscendBackend

gated_delta_rule_registry = BackendRegistry("gated_delta_rule")
gated_delta_rule_registry.register(GatedDeltaRuleTritonAscendBackend())

__all__ = ["gated_delta_rule_registry"]
