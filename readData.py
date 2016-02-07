import numpy as np
import wave
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from binascii import hexlify


def decodeManchester(data):
    periods = []
    pos = 0
    startVal = pos
    while pos < len(data):
        if data[pos] != data[startVal]:
            if data[pos] > data[startVal]:
                val = 0
            else:
                val = 1
            periods.append(((pos - startVal), val))
            startVal = pos
        pos += 1
    encoding = []

    frames = []
    frame = None
    infer_period = {}
    error = False

    for p, s in periods:
        if p > 1000:
            if frame is not None:
                if error:
                    print periods
                frames.append(frame)
                infer_period = {}
                error = False
            frame = []
            mid = False
            #encoding.append(s)
        elif s in infer_period and p > 1.5 * infer_period[s]:
            if not mid:
                error = True
            # assert mid
            mid = True
            frame.append(s)
        else:
            if s not in infer_period:
                infer_period[s] = p
            mid = not mid
            if mid:
                frame.append(s)
    if error:
        print periods
    frames.append(frame)
    return periods, frames

def decode(bits, lsb=True):
    code = []
    nibbles = []

    c = 0
    c2 = 0
    for i, b in enumerate(bits):
        j = i % 4
        if j == 0 and i > 0:
            code += [hexlify(chr(c))[1]]
            nibbles += [c]
            
            c = 0
        if lsb:
            c += b * (2**j)
        else:
            c += b * (2**(3-j))
    
    return "".join(code)


def readManyFiles(fList):
    decodedData = {}
    for i, f in enumerate(fList):
        print f
        try:
            decs = readFileAndDecode(f)
            decodedData[(i,f)] = decs
        except:
            print "Fuckup occurred"
            continue
        #for d in decs:
        #    decodedData.append(d)
    return decodedData


def readFileAndDecode(f):
    decodedData = []
    data = wavfile.read(f)

    nPoints = data[0]

    
    #data = data[1]

    # keep one channel only
    #data = data[:,0]
    data = data[1]
    data = (data[:, 0]**2 + data[:, 1]**2)

    # rectification
    data = np.abs(data)

    #plt.plot(data[:1000])
    #plt.show()


    # convolve to make simple low pass filter
    #kernel = np.hstack([-1 * np.ones(30) / 60.0, np.ones(30) / 60.0])
    kernel = np.ones(20) / 20.0
    y = np.convolve(data, kernel, mode='valid')

    signal = (y > 6000)

    #plt.plot(data, c="r")
    #plt.plot(y, c="b")
    #plt.show()

    # plt.plot(signal[:10000])
    # plt.ylim(-3, 3)
    # plt.show()

    periods, encoding = decodeManchester(signal)
    #print periods
    #print encoding, len(encoding), map(len, encoding), map(decode, encoding)
    for enc in encoding:
        dec = decode(enc)
        #print dec
        decodedData.append(dec)
    return decodedData



def main():
    f = "data/data_readings.wav"
    fList = [f]
    #readFileAndDecode(f)

    fListProcessed = []
    for i in range(18):
        fListProcessed.append("processed/test1 (%s).wav" % (i+1))

    data = readManyFiles(fListProcessed)
    #for k, ds in data.iteritems():
    for k in sorted(data.keys()):
        ds = data[k]
        for d in ds:
            print k, d    

if __name__ == "__main__":
    main()