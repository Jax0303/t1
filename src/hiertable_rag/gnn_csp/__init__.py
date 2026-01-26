"""
GNN-CSP: Graph Neural Network + Constraint Solving for Table Structure Recognition

A neurosymbolic approach that combines:
- Graph Neural Networks for learning spatial and semantic patterns
- Constraint Satisfaction Problem solving for enforcing table structure rules

This module addresses Row/Col-Count-Mismatch errors (83% of failures) by
explicitly modeling physical table constraints.
"""

from .gnn_encoder import CellGraphBuilder, TableGNN
from .constraint_layer import ConstraintLayer, HardConstraints, SoftConstraints
from .csp_solver import TableCSPSolver
from .pipeline import GNNCSPPipeline

__all__ = [
    'CellGraphBuilder',
    'TableGNN',
    'ConstraintLayer',
    'HardConstraints',
    'SoftConstraints',
    'TableCSPSolver',
    'GNNCSPPipeline',
]
