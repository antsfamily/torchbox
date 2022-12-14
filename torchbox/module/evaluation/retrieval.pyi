class Dice(nn.Module):
    ...

class Jaccard(nn.Module):
    r"""Jaccard distance

    .. math::
       d_{J}({\mathbb A}, {\mathbb B})=1-J({\mathbb A}, {\mathbb B})=\frac{|{\mathbb A} \cup {\mathbb B}|-|{\mathbb A} \cap {\mathbb B}|}{|{\mathbb A} \cup {\mathbb B}|}

    """

class Iridescent(nn.Module):
    r"""Iridescent Distance

    .. math::
       d_{J}({\mathbb A}, {\mathbb B})=1-J({\mathbb A}, {\mathbb B})=\frac{|{\mathbb A} \cup {\mathbb B}|-|{\mathbb A} \cap {\mathbb B}|}{|{\mathbb A} \cup {\mathbb B}|}

    """

class F1(nn.Module):
    r"""F1 distance

    .. math::
       F_{\beta} = 1 -\frac{(1+\beta^2)  P  R}{\beta^2 P + R}
       :label: equ-F-measure

    where,

    .. math::
       {\rm PPV} = {P} = \frac{\rm TP}{{\rm TP} + {\rm FP}}
       :label: equ-Precision

    .. math::
       {\rm TPR} = {R} = \frac{\rm TP}{{\rm TP} + {\rm FN}}
       :label: equ-Recall

    """


