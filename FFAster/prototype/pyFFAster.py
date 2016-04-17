import numpy as np

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def first_prime_factor(n):
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            return d
        d += 1
    return n

def downsample(data,factor):
    print data,data.size,factor
    data = data[:data.size-(data.size%factor)]
    return data.reshape(data.size/factor,factor).sum(axis=1)/np.sqrt(factor)

class PrimeDownsampler(object):
    def __init__(self,data=None,parent=None,downsampled_by=1):
        self.parent = parent
        self.data = data
        self.children = {}
        self.used = False
        self.downsampled_by = downsampled_by
        self.downsampling_factor = 1
        self.factors = {}

    def _get_factor(self,dsamp_factor):
        try:
            return self.factors[dsamp_factor]
        except KeyError:
            factor = first_prime_factor(dsamp_factor)
            self.factors[dsamp_factor] = factor
            return factor

    def create_plan(self,dsamp_factor):
        factor = self._get_factor(dsamp_factor)
        if factor not in self.children:
            self.children[factor] = PrimeDownsampler(parent=self,downsampled_by=factor)
            self.children[factor].downsampling_factor = self.downsampling_factor*factor
        if factor != 1:
            self.children[factor].create_plan(dsamp_factor/factor)

    def get_data(self):
        if self.parent is None and self.data is None:
            raise Exception("No data in top level node")
        if self.data is None and self.parent is not None:
            self.data = downsample(self.parent.get_data(),self.downsampled_by)
        return self.data

    def get(self,dsamp_factor):
        factor = self._get_factor(dsamp_factor)
        if dsamp_factor == 1:
            return self.get_data()
        else:
            try:
                child = self.children[factor]
            except KeyError:
                self.create_plan(dsamp_factor)
            return self.children[factor].get(dsamp_factor/factor)

    def set_data(self,data):
        self.data = data

    def display(self,prefix=""):
        for factor,child in sorted(self.children.items()):
            if factor == 1: continue
            print "%sFactor: %d, Data: %s"%(prefix,factor,child.data)
            child.display(prefix=prefix+"  ")

class _FFAPlan(object):
    def __init__(self,N,tsamp,p_sec):
        self.N = N                                 # number of samples in times series
        self.tsamp = tsamp                         # sampling time of time series
        self.p_sec = p_sec                         # period in seconds to search
        self.P = np.floor(self.p_sec/self.tsamp)   # period in samples
        self.T = np.floor(self.N/self.P)           # number of full periods
        self.T2 = int(2**np.ceil(np.log2(self.T))) # next power of 2 periods
        self.padding = (self.T2-self.T)*self.P     # required padding to give power of 2 periods
        self.layers = int(np.log2(self.T2))        # number of "butterfly" layers in the FFA
        
        # calculate the periods searched in the FFA (both in samples and in seconds)
        self.pstep = ((self.P+1)/(self.T2*self.P))
        self.pstep_sec = self.pstep * self.tsamp
        self.periods = self.P + self.pstep*np.arange(1,self.T2+1)
        self.periods_sec = self.periods * self.tsamp
        self.next_period = self.periods_sec[-1]+self.pstep_sec
            
        # calculate rough operation count for FFA
        self.op_count = self.N * np.log2(self.N/self.P)

        self.input = None
        self.output = None

    def display(self):
        for key,val in self.__dict__.items():
            if "periods" in key:
                print key," =  %d periods between %.4f and %.4f"%(val.size,val[0],val[-1])
            else:
                print key," = ",val

class FFAPlan(object):
    """
    Container for execution plan of FFA
    """
    def __init__(self,nsamp, tsamp, p_min, p_max, min_dc):
        self.nsamp = nsamp
        self.tsamp = tsamp
        self.p_min = p_min
        self.p_max = p_max
        self.min_dc = min_dc
        self.downsamplings = PrimeDownsampler()
        self.by_profilelen = {}
        self.twiddle_factors = {}
        self.by_downsampling = {}
        self.by_period = {}

    def _calc_twiddle_facts(self,nturns):
        if nturns not in self.twiddle_factors:
            layers = int(np.log2(nturns))
            mods = 2**np.arange(1,layers+1).reshape(layers,1)
            self.twiddle_factors[nturns] = (-1)*((np.mod(np.arange(nturns),mods)+1)/2)

    def _dsamp_factor(self,period):
        return int(np.floor(period/self.tsamp * self.min_dc))

    def add(self,period):
        ds = self._dsamp_factor(period)
        plan = _FFAPlan(self.nsamp//ds,self.tsamp*ds,period)
        self._calc_twiddle_facts(plan.T2)
        self.downsamplings.create_plan(ds)

        try:
            self.by_downsampling[ds].append(plan)
        except KeyError:
            self.by_downsampling[ds] = []
            self.by_downsampling[ds].append(plan)
            
        try:
            self.by_profilelen[plan.P].append(plan)
        except KeyError:
            self.by_profilelen[plan.P] = []
            self.by_profilelen[plan.P].append(plan)

        try:
            self.by_period[plan.p_sec].append(plan)
        except KeyError:
            self.by_period[plan.p_sec] = []
            self.by_period[plan.p_sec].append(plan)

        return plan    

def deredden(data):
    from scipy.signal import medfilt
    data = data - medfilt(data,511)
    return data

def pyFFA_create_plan(nsamp, tsamp, p_min, p_max, min_dc):
    plan = FFAPlan(nsamp, tsamp, p_min, p_max, min_dc)
    while p_min < p_max:
        plan_ = plan.add(p_min)
        p_min = plan_.next_period
    return plan
        
def repeat(a,n):
    return np.repeat(a,n).reshape(a.size,n).transpose().ravel()

def multi_radon(data,steps=8):
    ndata = np.copy(data)
    out = []
    for _ in range(steps):
        out.append(radon(ndata)[0])
        for ii,row in enumerate(ndata):
            ndata[ii] = np.roll(ndata[ii],-ii)
    return np.vstack(out)

def radon(data,twiddles=None,nrows=None):
    rows,cols = data.shape
    if nrows:
        ndata = np.zeros([nrows,cols])
        if nrows >= rows:
            ndata[:rows,:] = data
        else:
            ndata[:nrows,:] = data[:nrows,:]
    else:
        nrows = int(2**np.ceil(np.log2(rows)))
        ndata = np.zeros([nrows,cols])
        ndata[:rows,:] = data
    data = ndata
    rows = nrows
    
    layers = int(np.log2(rows))
    if twiddles is None:
        twiddles = []
        for i in range(1,int(np.log2(rows)+1)):
            twiddles.append(repeat(-1*(np.arange(2**(i)))/2,2**layers/(2**i)))
    twiddles = np.array(twiddles)
    input_ = np.copy(data)
    out = np.zeros_like(data)
    p = [data]


    for layer in range(layers):
        k = int(2**(layer+1))
        for node in range(0,rows,k):
            for jj in range(k/2):
                for ii in range(2):
                    out[node+2*jj+ii] = data[node+jj] + np.roll(data[node+k/2+jj],
                                                               twiddles[layer][node+2*jj+ii])        
        data = np.copy(out)
    final_out = out/np.sqrt(rows)
    return final_out,twiddles

def psearch(tp_ar,widths):
    widths = np.asarray(widths)
    nrows,nphase = tp_ar.shape
    orig_nrows = nrows
    nwidths = widths.size
    profiles = np.zeros([nwidths,nphase])
    for ii,width in enumerate(widths):
        profiles[ii][0:width] = 1
    a,t = radon(tp_ar)
    b,t = radon(tp_ar,twiddles=-1*t)
    ar = np.vstack((np.flipud(b),a))
    ar-=np.median(ar)
    ar/= 1.4826 * np.median(abs(ar))
    nrows,nphase = ar.shape
    ar_f = np.fft.fft(ar).repeat(nwidths,axis=0).reshape(nrows,nwidths,nphase)
    pr_f = np.fft.fft(profiles)
    convd = abs(np.fft.ifft(ar_f * pr_f))
    convd/=np.sqrt(widths.reshape(1,nwidths,1))
    peaks = convd.max(axis=2)
    phase = convd.argmax()%convd.shape[-1]
    shift = float(peaks.argmax())/nwidths - (nrows-1)
    shift = shift/orig_nrows
    width = widths[peaks.argmax()%nwidths]
    tp = np.copy(tp_ar)
    for ii,row in enumerate(tp):
        tp[ii] = np.roll(tp[ii],int(np.round(shift*ii)))
    return peaks.max(),phase/nphase,shift,width,convd,tp
    

def _pyFFA_execute_plan(plan,data,master_plan):
    data = np.pad(data[:plan.T*plan.P],(0,plan.padding),mode="constant")
    data = data.reshape(plan.T2,plan.P)
    input_ = np.copy(data)
    out = np.zeros_like(data)
    for layer in range(plan.layers):
        k = int(2**(layer+1))
        for node in range(0,plan.T2,k):
            for jj in range(k/2):
                for ii in range(2):
                    out[node+2*jj+ii] = data[node+jj] + np.roll(data[node+k/2+jj],
                                                                master_plan.twiddle_factors[2**plan.layers][layer][node+2*jj+ii])
        data = np.copy(out)
    final_out = out/np.sqrt(plan.T)
    plan.output = final_out
    plan.input = input_
    return final_out.max(axis=1)

def pyFFA_execute_plan(input_plan,input_data):
    input_plan.downsamplings.set_data(input_data)
    outputs = []
    for downsamp,plans in input_plan.by_downsampling.items():
        data = input_plan.downsamplings.get(downsamp) #downsample(input_data,downsamp)
        
        data = deredden(data)
        data-=np.median(data)
        mads = np.median(abs(data))*1.4826
        data/=mads
        for plan in plans:
            print "Period:",plan.periods_sec[0]
            out = _pyFFA_execute_plan(plan,data,input_plan)
            outputs.append((plan.periods_sec,out))
    return np.hstack(outputs)
                                   


