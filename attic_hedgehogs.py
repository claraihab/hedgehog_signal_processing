'''
    attic - Library of unused functions 
'''


# unused !!!
# ------------
def findBaseFreq(FFT_data:np.ndarray, timebase:float, weighted:bool=True) ->float:
    '''
    find the 'base' frequency from FFT-data,
    
    :param FFT_data:
    :param timebase:   length of the data (in s) used to calculate FFT
    :param weighted:        if True -> FFT-values will be weighted by 1/n to favor low frequencies
    :return:
    '''
    data = np.absolute(FFT_data)    # use absolute values,
    data[0] = 0                     # set DC-component to zero
    
    if weighted:
        for i in range(1, len(data)):
            data[i] = data[i] /i
    peak_idx = np.argmax(data)           # find highest peak
    peak_hz  = peak_idx / timebase  # convert to hz
    return peak_hz

def maskFFTdata(FFT_data:np.ndarray,  timebase:float, min_Hz:float, max_Hz:float) -> float:
    '''
    return a numpy array where the values outside the range min-max are set to zero
    :param FFT_data:
    :param timebase:
    :param min_Hz:
    :param max_Hz:
    :return:
    '''
    min_idx = int(min_Hz * timebase)
    max_idx = int(max_Hz * timebase)

    data = FFT_data.copy()  # reserve original data
    if min_idx >0:
        data[0:min_idx] = 0
    if max_idx >0 and max_idx <=len(data):
        data[max_idx:]  = 0
    return data

