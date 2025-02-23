"""
## Overview
This subpackage contains classes and functions for computing PAC-Bayes bounds.

## Contents
- **Concrete bounds** like KLBound, McAllesterBound
- **Interfaces** or base classes for creating custom bounds

Use these bounds to estimate or certify generalization risk after training a
probabilistic neural network.
"""

from core.bound.AbstractBound import AbstractBound
from core.bound.KLBound import KLBound
from core.bound.McAllesterBound import McAllesterBound
