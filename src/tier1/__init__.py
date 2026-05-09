"""Tier 1: GNN-based fire risk prediction from binary detector signals.

See ``docs/tier1_gnn_design.md`` for the full design.

This package is intentionally separated from ``src/models`` (Tier 2: ConvLSTM,
PI-FNO) because Tier 1 operates on a building graph (16-20 nodes) rather than
the (60, 40, 6) grid that Tier 2 models use.
"""
