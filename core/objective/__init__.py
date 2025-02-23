"""
## Overview
Hosts a variety of PAC-Bayes training objectives that combine empirical
loss and KL divergence in different ways.

## Contents
- **AbstractObjective**: A base interface
- **BBBObjective, FClassicObjective, FQuadObjective, McAllesterObjective, TolstikhinObjective**

By choosing an appropriate objective, you can guide training to minimize
a PAC-Bayes bound on the model’s risk.
"""

from core.objective.AbstractObjective import AbstractObjective
from core.objective.BBBObjective import BBBObjective
from core.objective.FClassicObjective import FClassicObjective
from core.objective.FQuadObjective import FQuadObjective
from core.objective.McAllesterObjective import McAllesterObjective
from core.objective.TolstikhinObjective import TolstikhinObjective
