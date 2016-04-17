import os
import sys
import matplotlib.pyplot as plt
import numpy as np

FILES = [
    "generic_pattern.bin",
    "chequer_pattern.bin",
    "pulse_pattern.bin"
    ]

def run(xdim,ydim):
    os.system("./test_test %d %d"%(xdim,ydim))

def main(xdim,ydim):
    run(xdim,ydim)
    
    fig,axes = plt.subplots(nrows=1,ncols=3)
    
    for ax,fname in zip(axes,FILES):
        ar = np.fromfile(fname,dtype="float32").reshape(ydim,xdim)
        ax.imshow(ar,aspect="auto",interpolation="none")
        ax.set_title(fname)
        
    plt.show()
        

if __name__ == "__main__":
    xdim = 1234
    ydim = 2345
    try:
        xdim = int(sys.argv[1])
        ydim = int(sys.argv[2])
    except:
        pass
    
    main(xdim,ydim)
