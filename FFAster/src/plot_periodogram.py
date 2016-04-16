import pylab as plt
import numpy as np
import sys

z = np.fromfile(sys.argv[1],dtype=[("sn","float32"),("w","int32"),("p","float32")])
plt.plot(z["p"],z["sn"])
plt.show()
