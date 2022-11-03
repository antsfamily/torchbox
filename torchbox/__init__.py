from __future__ import absolute_import
from .version import __version__

from .utils.const import *
from .utils.ios import loadyaml, saveyaml, loadjson, loadmat, savemat, loadh5, saveh5, mvkeyh5
from .utils.image import imread, imsave, imadjust, imadjustlog, histeq, imresize
from .utils.file import data_path, pkg_path, copyfile, copyfiles, listxfile, pathjoin, fileparts, readtxt, readnum, readsec
from .utils.convert import dict2str, str2list, str2num, str2sec, int2bstr, bstr2int
from .utils.colormaps import cmaps, viridis, parula
from .utils.colors import rgb2gray, gray2rgb, DISTINCT_COLORS_HEX, DISTINCT_COLORS_RGB, DISTINCT_COLORS_CMYK, DISTINCT_COLORS_RGB_NORM, BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS
from .utils.plot_show import cplot, plots, Plots, plot, imshow, mesh, mshow

from .base.baseops import upkeys, dreplace, dmka, cat
from .base.arrayops import sl, cut, arraycomb
from .base.mathops import db2mag, mag2db, fnab, ebeo, sinc, nextpow2, prevpow2, ematmul, matmul, c2r, r2c, conj, real, imag, abs, pow
from .base.randomfunc import setseed, randgrid, randperm, randperm2d
from .base.typevalue import peakvalue, dtypes

from .dsp.ffts import padfft, freq, fftfreq, fftshift, ifftshift, fft, ifft
from .dsp.convolution import cutfftconv1, fftconv1
from .dsp.correlation import cutfftcorr1, fftcorr1, xcorr, acorr, accc
from .dsp.normalsignals import rect, chirp
from .dsp.interpolation import interpolate, interpolatec
from .dsp.polynomialfit import polyfit, polyval, rmlinear
from .dsp.function_base import unwrap, unwrap2
from .dsp.window_function import window, windowing

from .evaluation.contrast import contrast
from .evaluation.entropy import entropy
from .evaluation.classification import accuracy
from .evaluation.norm import fnorm, pnorm
from .evaluation.error import mse, sse, mae, sae, nmse, nsse, nmae, nsae
from .evaluation.snrs import snr
from .evaluation.retrieval import true_positive, true_negative, \
    false_positive, false_negative, \
    precision, recall, sensitivity, selectivity, fmeasure
from .evaluation.similarity import jaccard_index, dice_coeff
from .evaluation.ssims import gaussian_filter, ssim, msssim

from .misc.transform import standardization, scale, quantization, db20, ct2rt, rt2ct
from .misc.mapping_operation import mapping
from .misc.sampling import slidegrid, dnsampling, sample_tensor, shuffle_tensor, split_tensor, tensor2patch, patch2tensor, read_samples
from .misc.draw_shapes import draw_rectangle, draw_eclipse
from .misc.noising import awgns, awgns2, imnoise, awgn, wgn

from .datasets.mnist import read_mnist
from .datasets.mstar import mstar_header, mstar_raw
from .nn.activations import linear, sigmoid, tanh, softplus, softsign, elu, relu, relu6, selu, crelu, leaky_relu

from .linalg.orthogonalization import orth

from .layerfunction.cplxfunc import csign, csoftshrink, softshrink
from .layerfunction.complex_functions import complex_relu, complex_leaky_relu, complex_max_pool2d, complex_dropout, complex_dropout2d, complex_upsample

from .module.dsp.convolution import FFTConv1, Conv1, MaxPool1, Conv2, MaxPool2
from .module.dsp.interpolation import Interp1
from .module.dsp.polynomialfit import PolyFit

from .module.misc.transform import Standardization


from .module.evaluation.contrast import Contrast
from .module.evaluation.entropy import Entropy
from .module.evaluation.norm import Fnorm, Pnorm
from .module.evaluation.error import MSE, SSE, MAE, SAE, NMSE, NSSE, NMAE, NSAE
from .module.evaluation.ssims import SSIM, MSSSIM
from .module.evaluation.variation import TotalVariation
from .module.evaluation.retrieval import Dice, Jaccard, F1

from .module.loss.contrast import ContrastLoss, NegativeContrastLoss, ReciprocalContrastLoss
from .module.loss.entropy import EntropyLoss
from .module.loss.norm import FnormLoss, PnormLoss
from .module.loss.perceptual import RandomProjectionLoss
from .module.loss.retrieval import DiceLoss, JaccardLoss, F1Loss
from .module.loss.variation import TotalVariation
from .module.loss.fourier import FourierLoss, FourierAmplitudeLoss, FourierPhaseLoss, FourierNormLoss
from .module.loss.error import MSELoss, SSELoss, MAELoss, SAELoss, NMSELoss, NSSELoss, NMAELoss, NSAELoss
from .module.loss.sparse_metric import LogSparseLoss, FourierLogSparseLoss


from .module.layers.balanceconv2d import BalaConv2d
from .module.layers.cnnsize import ConvSize1d, ConvTransposeSize1d, PoolSize1d, UnPoolSize1d, ConvSize2d, ConvTransposeSize2d, PoolSize2d, UnPoolSize2d
from .module.layers.edge import EdgeDetector, EdgeFeatureExtractor
from .module.layers.pool import MeanSquarePool2d, PnormPool2d
from .module.layers.complex_layers import ComplexSequential, ComplexMaxPool2d, ComplexMaxPool1d, ComplexDropout,  ComplexDropout2d, ComplexReLU, ComplexLeakyReLU, ComplexConvTranspose2d, ComplexConv2d, ComplexConvTranspose1d, ComplexConv1d, ComplexLinear, ComplexUpsample, NaiveComplexBatchNorm1d, NaiveComplexBatchNorm2d, NaiveComplexBatchNorm1d, ComplexBatchNorm2d, ComplexBatchNorm1d, ComplexConv1, ComplexMaxPool1, ComplexConv2, ComplexMaxPool2
from .module.layers.phase_convolution import PhaseConv1d, PhaseConv2d, ComplexPhaseConv1d, ComplexPhaseConv2d, PhaseConvTranspose1d, PhaseConvTranspose2d, ComplexPhaseConvTranspose1d, ComplexPhaseConvTranspose2d
from .module.layers.fft_layers import FFTLayer1d
from .module.layers.convolution import FFTConv1, Conv1, Conv2, MaxPool1, MaxPool2
from .module.layers.consistency_layers import DataConsistency2d
from .module.layers.flow_layers import ActNorm, InvConv2d, InvConv2dLU, ZeroConv2d, AffineCoupling, Flow, FlowBlock, Glow

from .spl import voptimizer
from .spl import spfunction

from .optim.learning_rate import gammalr, LrFinder
from .optim.lr_scheduler import GaussianLR, MountainLR
from .optim.save_load import device_transfer

from .summary.loss_log import LossLog

from .diagnose.plotgradflow import plot_gradflow_v1, plot_gradflow_v2



