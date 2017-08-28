
import numpy
import sigproc
from scipy.fftpack import dct
from scipy.ndimage import convolve1d

def get_wav_feature(wav_file):
	'''
	获取WAV文件最终输入的特征向量的函数
	参数：
		wav_file: WAV文件对象
	返回值：
		神经网络可直接用的输入层向量
	'''
	vector = numpy.matrix([])
	#在这里添加代码
	
	return vector
	
def mfcc(signal, samplerate, conf):
    '''
    Compute MFCC features from an audio signal.
	从一个声音信号中计算MFCC特征向量
    Args:
	参数：
        signal: the audio signal from which to compute features. Should be an
            N*1 array
			通过这个声音信号计算特征向量，它应当是一个N*1的数组
        samplerate: the samplerate of the signal we are working with.
			要处理的信号的采样率
        conf: feature configuration
			特征的配置

    Returns:
	返回值：
        A numpy array of size (NUMFRAMES by numcep) containing features. Each
        row holds 1 feature vector, a numpy vector containing the signal
        log-energy
		返回一个包含特征向量的numpy数组，每一列是一个特征向量，每一个numpy向量包含信号的对数能量
    '''

    feat, energy = fbank(signal, samplerate, conf)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :int(conf['numcep'])]
    feat = lifter(feat, float(conf['ceplifter']))
    return feat, numpy.log(energy)

def fbank(signal, samplerate, conf):
    '''
    Compute fbank features from an audio signal.
	从一个声音信号中计算fbank特征向量
    Args:
	参数：
        signal: the audio signal from which to compute features. Should be an
            N*1 array
			要计算特征的声音信号，一个N*1维的数组
        samplerate: the samplerate of the signal we are working with.
			要处理信号的采样率
        conf: feature configuration
			特征的配置

    Returns:
	返回值：
        A numpy array of size (NUMFRAMES by nfilt) containing features, a numpy
        vector containing the signal energy
		返回一个包含特征向量的numpy数组，一个包含信号能量的numpy向量
    '''

    highfreq = int(conf['highfreq'])
    if highfreq < 0:
        highfreq = samplerate/2

    signal = sigproc.preemphasis(signal, float(conf['preemph']))
    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate,
                              float(conf['winstep'])*samplerate)
    pspec = sigproc.powspec(frames, int(conf['nfft']))

    # this stores the total energy in each frame
    energy = numpy.sum(pspec, 1)

    # if energy is zero, we get problems with log
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)

    filterbank = get_filterbanks(int(conf['nfilt']), int(conf['nfft']),
                                 samplerate, int(conf['lowfreq']), highfreq)

    # compute the filterbank energies
    feat = numpy.dot(pspec, filterbank.T)

    # if feat is zero, we get problems with log
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)

    return feat, energy

def logfbank(signal, samplerate, conf):
    '''
    Compute log-fbank features from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by nfilt) containing features, a numpy
        vector containing the signal log-energy
    '''
    feat, energy = fbank(signal, samplerate, conf)
    return numpy.log(feat), numpy.log(energy)

def ssc(signal, samplerate, conf):
    '''
    Compute ssc features from an audio signal.

    Args:
        signal: the audio signal from which to compute features. Should be an
            N*1 array
        samplerate: the samplerate of the signal we are working with.
        conf: feature configuration

    Returns:
        A numpy array of size (NUMFRAMES by nfilt) containing features, a numpy
        vector containing the signal log-energy
    '''

    highfreq = int(conf['highfreq'])
    if highfreq < 0:
        highfreq = samplerate/2
    signal = sigproc.preemphasis(signal, float(conf['preemph']))
    frames = sigproc.framesig(signal, float(conf['winlen'])*samplerate,
                              float(conf['winstep'])*samplerate)
    pspec = sigproc.powspec(frames, int(conf['nfft']))

    # this stores the total energy in each frame
    energy = numpy.sum(pspec, 1)

    # if energy is zero, we get problems with log
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)

    filterbank = get_filterbanks(int(conf['nfilt']), int(conf['nfft']),
                                 samplerate, int(conf['lowfreq']), highfreq)

    # compute the filterbank energies
    feat = numpy.dot(pspec, filterbank.T)
    tiles = numpy.tile(numpy.linspace(1, samplerate/2, numpy.size(pspec, 1)),
                       (numpy.size(pspec, 0), 1))

    return numpy.dot(pspec*tiles, filterbank.T) / feat, numpy.log(energy)

def hz2mel(rate):
    '''
    Convert a value in Hertz to Mels

    Args:
        rate: a value in Hz. This can also be a numpy array, conversion proceeds
            element-wise.

    Returns:
        a value in Mels. If an array was passed in, an identical sized array is
        returned.
    '''
    return 2595 * numpy.log10(1+rate/700.0)

def mel2hz(mel):
    '''
    Convert a value in Mels to Hertz

    Args:
        mel: a value in Mels. This can also be a numpy array, conversion
            proceeds element-wise.

    Returns:
        a value in Hertz. If an array was passed in, an identical sized array is
        returned.
    '''
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0,
                    highfreq=None):
    '''
    Compute a Mel-filterbank.

    The filters are stored in the rows, the columns correspond to fft bins.
    The filters are returned as an array of size nfilt * (nfft/2 + 1)

    Args:
        nfilt: the number of filters in the filterbank, default 20.
        nfft: the FFT size. Default is 512.
        samplerate: the samplerate of the signal we are working with. Affects
            mel spacing.
        lowfreq: lowest band edge of mel filters, default 0 Hz
        highfreq: highest band edge of mel filters, default samplerate/2

    Returns:
        A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each
        row holds 1 filter.
    '''

    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, nfilt+2)

    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bins = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbanks = numpy.zeros([nfilt, nfft/2+1])
    for j in xrange(0, nfilt):
        for i in xrange(int(bins[j]), int(bins[j+1])):
            fbanks[j, i] = (i - bins[j])/(bins[j+1]-bins[j])
        for i in xrange(int(bins[j+1]), int(bins[j+2])):
            fbanks[j, i] = (bins[j+2]-i)/(bins[j+2]-bins[j+1])
    return fbanks

def lifter(cepstra, liftering=22):
    '''
    Apply a cepstral lifter the the matrix of cepstra.

    This has the effect of increasing the magnitude of the high frequency DCT
    coeffs.

    Args:
        cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
        liftering: the liftering coefficient to use. Default is 22. L <= 0
            disables lifter.

    Returns:
        the lifted cepstra
    '''
    if liftering > 0:
        _, ncoeff = numpy.shape(cepstra)
        lift = 1+(liftering/2)*numpy.sin(numpy.pi
                                         *numpy.arange(ncoeff)/liftering)
        return lift*cepstra
    else:
        # values of liftering <= 0, do nothing
        return cepstra

def deriv(features):
    '''
    Compute the first order derivative of the features

    Args:
        features: the input features

    Returns:
        the firs order derivative
    '''
    return convolve1d(features, [2, 1, 0, -1, -2], 0)

def delta(features):
    '''
    concatenate the first order derivative to the features

    Args:
        features: the input features

    Returns:
        the features concatenated with the first order derivative
    '''
    return numpy.concatenate((features, deriv(features)), 1)

def ddelta(features):
    '''
    concatenate the first and second order derivative to the features

    Args:
        features: the input features

    Returns:
        the features concatenated with the first and second order derivative
    '''
    deltafeat = deriv(features)
    return numpy.concatenate((features, deltafeat, deriv(deltafeat)), 1)

