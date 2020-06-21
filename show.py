import joblib
o=joblib.load("pca.jbl")
print(o["z_q"].shape)
#print(o.keys())

import numpy as np
x=np.load("dataset/train_data.spec.npy")
s=np.load("dataset/train_step.spec.npy")
print(x.shape)
print(s.shape)
"""
step=s[0]
print(step)
v=x[0,step-1,:]
print(v)
"""
