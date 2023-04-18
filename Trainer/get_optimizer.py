'''
Author: Kai Li
Date: 2021-03-12 09:56:09
LastEditors: Kai Li
LastEditTime: 2021-03-12 09:56:10
'''
from torch.optim.optimizer import Optimizer
from torch.optim import (
    Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, AdamW, ASGD
)
from torch_optimizer import (
    AccSGD, AdaBound, AdaMod, DiffGrad, Lamb, NovoGrad, PID, QHAdam,
    QHM, RAdam, SGDW, Yogi, Ranger, RangerQH, RangerVA
)


def make_optimizer(params, optimizer='adam', **kwargs):
    """
    Examples:
        >>> from torch import nn
        >>> model = nn.Sequential(nn.Linear(10, 10))
        >>> optimizer = make_optimizer(model.parameters(), optimizer='sgd',
        >>>                            lr=1e-3)
    """
    return get(optimizer)(params, **kwargs)


def get(identifier):
    """ Returns an optimizer function from a string. Returns its input if it
    is callable (already a :class:`torch.optim.Optimizer` for example).
    Args:
        identifier (str or Callable): the optimizer identifier.
    Returns:
        :class:`torch.optim.Optimizer` or None
    """
    if isinstance(identifier, Optimizer):
        return identifier
    elif isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f'Could not interpret optimizer : {str(identifier)}')
        return cls
    raise ValueError(f'Could not interpret optimizer : {str(identifier)}')