
import torchbox as tb
import torch as th
import matplotlib.pyplot as plt


datafile = '/mnt/d/DataSets/sar/ERS/mat/E2_81988_STD_F327/E2_81988_STD_F327/ERS2_SAR_SLC=E2_81988_STD_F327(sl=1el=8192).mat'

sardata = tb.loadmat(datafile)['sardata']

print(sardata[0][0][1].shape)
SI = sardata[0][0][1]

print(SI.shape, SI.dtype)

SI = th.from_numpy(SI)

SI1 = tb.mapping(SI, method='1Sigma')
SI2 = tb.mapping(SI, method='2Sigma')
SI3 = tb.mapping(SI, method='3Sigma')

plt.figure()
plt.imshow(SI, cmap='gray')
plt.figure()
plt.imshow(SI1, cmap='gray')
plt.figure()
plt.imshow(SI2, cmap='gray')
plt.figure()
plt.imshow(SI3, cmap='gray')

plt.show()

