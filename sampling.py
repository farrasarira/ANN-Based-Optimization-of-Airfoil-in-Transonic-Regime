
import numpy as np

# Latin Hypercube Sampling
def sampling_rlh(
    Nsamp, dimen, edges = 0
):

  X = np.zeros(shape=[Nsamp, dimen])

  for i in range(dimen):
     X[:, i] = np.random.permutation(Nsamp) + 1

  if edges == 1:
      X = (X - 1)/(Nsamp -1)
  else:
      X = (X - 0.5)/Nsamp

  return X

