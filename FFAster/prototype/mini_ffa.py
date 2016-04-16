import numpy as np

def _calc_twiddle_facts(nturns):
    layers = int(np.log2(nturns))
    mods = 2**np.arange(1,layers+1).reshape(layers,1)
    return (-1)*((np.mod(np.arange(nturns),mods)+1)/2)


def execute(data):
    ydim,xdim = data.shape
    nlayers = int(np.log2(ydim))
    twiddles = _calc_twiddle_facts(ydim)
    out = np.zeros_like(data)
    for layer in range(nlayers):
        k = int(2**(layer+1))
        for node in range(0,ydim,k):
            for jj in range(k/2):
                for ii in range(2):
                    print "out node: ",node+2*jj+ii
                    print "in node0: ",node+jj
                    print "in node1: ",node+k/2+jj
                    print "shift: ",twiddles[layer][node+2*jj+ii]
                    print
                    out[node+2*jj+ii] = data[node+jj] + np.roll(data[node+k/2+jj], twiddles[layer][node+2*jj+ii])
        data = np.copy(out)
    return out

def indexer2(layer,size,nturns):
    d = {}

    k = int(2**(layer+1))
    twiddles = -1*_calc_twiddle_facts(nturns)
    for node in range(0,nturns,k):
        for jj in range(k/2):

            out_turn0 = node+2*jj
            out_turn1 = node+2*jj+1
            
            in_turn0 = node+jj
            in_turn1 = node+k/2+jj
            
            shift0 = twiddles[layer][out_turn0]
            shift1 = twiddles[layer][out_turn1]
            
            out_idx0 = out_turn0*size;
            out_idx1 = out_turn1*size;
            
            in_idx0 = in_turn0*size;
            in_idx1 = in_turn1*size;
            
            idx = np.arange(size)
           
            out = out_idx0+idx,in_idx0+idx,in_idx1 + (idx - shift0)%size,out_idx1+idx,in_idx0+idx,in_idx1 + (idx - shift1)%size
            d[(out_turn0,out_turn1,in_turn0,in_turn1,shift0,shift1)] = (out)

    return d

def indexer(layer,size,nturns):
    d = {}
    for block in np.arange(nturns/2):
        k = 1<<layer
        in_turn0 = block/k * k + block
        in_turn1 = in_turn0+k
        
        shift0 = block%k
        shift1 = shift0+1
        
        out_turn0 = block*2
        out_turn1 = out_turn0+1

        out_idx0 = (block*2)*size;
        out_idx1 = (block*2 + 1)*size;
        
        in_idx0 = in_turn0*size;
        in_idx1 = in_turn1*size;
        
        idx = np.arange(size)
        out = out_idx0+idx,in_idx0+idx,in_idx1 + (idx - shift0)%size,out_idx1+idx,in_idx0+idx,in_idx1 + (idx - shift1)%size
        d[(out_turn0,out_turn1,in_turn0,in_turn1,shift0,shift1)] = (out)

    return d

def test(layer,size,nturns):
    a = indexer(layer,size,nturns)
    b = indexer2(layer,size,nturns)
    
    for key,value in a.items():
        
        testval = b[key]
        for x,y in zip(value,testval):
            if not np.all(x==y):
                print "Fail:",key
            else:
                print "Pass:",key
        
        
        
