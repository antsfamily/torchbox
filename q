[1mdiff --git a/torchbox/evaluation/error.pyi b/torchbox/evaluation/error.pyi[m
[1mindex 18fc883..2f0b461 100644[m
[1m--- a/torchbox/evaluation/error.pyi[m
[1m+++ b/torchbox/evaluation/error.pyi[m
[36m@@ -481,7 +481,7 @@[m [mdef nsae(X, Y, cdim=None, dim=None, keepcdim=False, reduction='mean'):[m
     Both complex and real representation are supported.[m
 [m
     .. math::[m
[31m-       {\rm SAE}({\bf X, Y}) = \frac{\||{\bf X} - {\bf Y}|\|}{\||{\bf Y}|\|} = \frac{\sum_{i=1}^N |x_i - y_i|}{\sum_{i=1}^N |y_i|}[m
[32m+[m[32m       {\rm NSAE}({\bf X, Y}) = \frac{\||{\bf X} - {\bf Y}|\|}{\||{\bf Y}|\|} = \frac{\sum_{i=1}^N |x_i - y_i|}{\sum_{i=1}^N |y_i|}[m
 [m
     Parameters[m
     ----------[m
[1mdiff --git a/torchbox/module/layers/complex_layers.py b/torchbox/module/layers/complex_layers.py[m
[1mindex 08e09ff..24dce65 100644[m
[1m--- a/torchbox/module/layers/complex_layers.py[m
[1m+++ b/torchbox/module/layers/complex_layers.py[m
[36m@@ -14,6 +14,7 @@[m [mfrom torch.nn import Module, Parameter, init, Sequential[m
 from torch.nn import Conv2d, Conv1d, Linear, BatchNorm1d, BatchNorm2d[m
 from torch.nn import ConvTranspose2d, ConvTranspose1d[m
 from torch.nn import Upsample[m
[32m+[m[32mfrom torchbox.base.arrayops import sl[m
 from torchbox.layerfunction.complex_functions import complex_relu, complex_leaky_relu, complex_max_pool2d, complex_max_pool1d[m
 from torchbox.layerfunction.complex_functions import complex_dropout, complex_dropout2d[m
 from torchbox.layerfunction.cplxfunc import csoftshrink, softshrink[m
[36m@@ -208,14 +209,18 @@[m [mclass ComplexUpsample(Module):[m
 [m
 class ComplexLinear(Module):[m
 [m
[31m-    def __init__(self, in_features, out_features):[m
[32m+[m[32m    def __init__(self, in_features, out_features, bias=True, cdim=-1):[m
         super(ComplexLinear, self).__init__()[m
[31m-        self.fcr = Linear(in_features, out_features)[m
[31m-        self.fci = Linear(in_features, out_features)[m
[32m+[m[32m        self.fcr = Linear(in_features, out_features, bias=bias)[m
[32m+[m[32m        self.fci = Linear(in_features, out_features, bias=bias)[m
[32m+[m[32m        self.cdim = cdim[m
 [m
     def forward(self, input):[m
[31m-        return th.stack((self.fcr(input[..., 0]) - self.fci(input[..., 1]),[m
[31m-                         self.fcr(input[..., 1]) + self.fci(input[..., 0])), dim=-1)[m
[32m+[m[32m        D = input.dim()[m
[32m+[m[32m        idxr = sl(D, axis=self.cdim, idx=[0])[m
[32m+[m[32m        idxi = sl(D, axis=self.cdim, idx=[1])[m
[32m+[m[32m        return th.stack((self.fcr(input[idxr]) - self.fci(input[idxi]),[m
[32m+[m[32m                         self.fcr(input[idxi]) + self.fci(input[idxr])), dim=self.cdim)[m
 [m
 [m
 class NaiveComplexBatchNorm1d(Module):[m
[36m@@ -223,13 +228,17 @@[m [mclass NaiveComplexBatchNorm1d(Module):[m
     Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.[m
     '''[m
 [m
[31m-    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):[m
[32m+[m[32m    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, cdim=-1):[m
         super(NaiveComplexBatchNorm1d, self).__init__()[m
         self.bnr = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)[m
         self.bni = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)[m
[32m+[m[32m        self.cdim = cdim[m
 [m
     def forward(self, input):[m
[31m-        return th.stack((self.bnr(input[..., 0]), self.bni(input[..., 1])), dim=-1)[m
[32m+[m[32m        D = input.dim()[m
[32m+[m[32m        idxr = sl(D, axis=self.cdim, idx=[0])[m
[32m+[m[32m        idxi = sl(D, axis=self.cdim, idx=[1])[m
[32m+[m[32m        return th.stack((self.bnr(input[..., idxr]), self.bni(input[..., idxi])), dim=self.cdim)[m
 [m
 [m
 class NaiveComplexBatchNorm2d(Module):[m
[1mdiff --git a/torchbox/module/layers/complex_layers.pyi b/torchbox/module/layers/complex_layers.pyi[m
[1mindex 2d9de49..b66a741 100644[m
[1m--- a/torchbox/module/layers/complex_layers.pyi[m
[1m+++ b/torchbox/module/layers/complex_layers.pyi[m
[36m@@ -5,6 +5,7 @@[m [mfrom torch.nn import Module, Parameter, init, Sequential[m
 from torch.nn import Conv2d, Conv1d, Linear, BatchNorm1d, BatchNorm2d[m
 from torch.nn import ConvTranspose2d, ConvTranspose1d[m
 from torch.nn import Upsample[m
[32m+[m[32mfrom torchbox.base.arrayops import sl[m
 from torchbox.layerfunction.complex_functions import complex_relu, complex_leaky_relu, complex_max_pool2d, complex_max_pool1d[m
 from torchbox.layerfunction.complex_functions import complex_dropout, complex_dropout2d[m
 from torchbox.layerfunction.cplxfunc import csoftshrink, softshrink[m
[36m@@ -136,7 +137,7 @@[m [mclass ComplexUpsample(Module):[m
 class ComplexLinear(Module):[m
     ...[m
 [m
[31m-    def __init__(self, in_features, out_features):[m
[32m+[m[32m    def __init__(self, in_features, out_features, bias=True, cdim=-1):[m
         ...[m
 [m
     def forward(self, input):[m
[36m@@ -145,7 +146,7 @@[m [mclass ComplexLinear(Module):[m
 class NaiveComplexBatchNorm1d(Module):[m
     ...[m
 [m
[31m-    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):[m
[32m+[m[32m    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, cdim=-1):[m
         ...[m
 [m
     def forward(self, input):[m
