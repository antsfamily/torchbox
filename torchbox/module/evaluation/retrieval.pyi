class Dice(nn.Module):
    ...

    def __init__(self, size_average=True, reduce=True):
        ...

    def soft_dice_coeff(self, P, G):
        ...

    def __call__(self, P, G):
        ...

class Jaccard(nn.Module):
    r"""Jaccard distance

    .. math::
       d_{J}({\mathbb A}, {\mathbb B})=1-J({\mathbb A}, {\mathbb B})=\frac{|{\mathbb A} \cup {\mathbb B}|-|{\mathbb A} \cap {\mathbb B}|}{|{\mathbb A} \cup {\mathbb B}|}

    """

    def __init__(self, size_average=True, reduce=True):
        ...

    def forward(self, P, G):
        ...

class Iridescent(nn.Module):
    r"""Iridescent Distance

    .. math::
       d_{J}({\mathbb A}, {\mathbb B})=1-J({\mathbb A}, {\mathbb B})=\frac{|{\mathbb A} \cup {\mathbb B}|-|{\mathbb A} \cap {\mathbb B}|}{|{\mathbb A} \cup {\mathbb B}|}

    """

    def __init__(self, size_average=True, reduce=True):
        ...

    def forward(self, P, G):
        ...

class F1(nn.Module):
    r"""F1 distance

    .. math::
       F_{\beta} = 1 -\frac{(1+\beta^2)  P  R}{\beta^2 P + R}

    where,

    .. math::
       {\rm PPV} = {P} = \frac{\rm TP}{{\rm TP} + {\rm FP}}

    .. math::
       {\rm TPR} = {R} = \frac{\rm TP}{{\rm TP} + {\rm FN}}

    """

    def __init__(self, size_average=True, reduce=True):
        ...

    def forward(self, P, G):
        ...


