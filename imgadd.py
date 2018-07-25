import os
import time
import random

import numpy as np
import cv2
from optparse import OptionParser

import itertools
import pprint
from PIL import Image

millis = int(round(time.time() * 1000))
def parse_options():
    parser = OptionParser()

    parser.add_option("-i",
                      dest="mp4file",
                      help="input mp4 file",
                      type="string",
                      action="store"
                      )


    parser.add_option("--ti",
                      dest="initTime",
                      help="init frame time in sec",
                      type="float",
                      default=0.0,
                      action="store"
                      )

    parser.add_option("-l",
                      dest="lstfile",
                      help="input list file",
                      type="string",
                      action="store"
                      )

    parser.add_option("-o",
                      dest="objtemplate",
                      help="template of object files",
                      type="string",
                      action="store"
                      )
    parser.add_option("-n",
                      dest="N",
                      help="number of fg.objects",
                      type="int",
                      action="store",
                      default=10
                      )
    (options, args) = parser.parse_args()

    if options.mp4file:
        return (options.N, options.mp4file, options.initTime, options.lstfile, options.objtemplate)

    parser.print_help()
    quit(-1)

import os

if __name__=='__main__':

    _N, mp4file, initTime, lstfile, objtemplate =  parse_options()
    _N = int(_N)
    _dir = objtemplate

    os.makedirs(_dir)

    if not os.path.exists(mp4file):
        print 'Input file not exist  %s' % mp4file
        quit(1)
    cv2.setUseOptimized(True)
    # capture from file
    cap = cv2.VideoCapture(mp4file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    framePeriod = int(1000.0 / fps)
    framePeriodInSec = 1.0 / fps
    print 'fps     %.4f' % fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print 'width   %d' % width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print 'height  %d' % height

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * initTime))
    ret, initFrame= cap.read()
    if not ret:
        print "Can't seek to time %s" % initTime

    b_channel, g_channel, r_channel = cv2.split(initFrame)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    # alpha = alpha_channel.astype(float) / 255
    
    initFrame = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    cv2.imshow('initFrame', initFrame)
    _fn = "{bg}".format(bg=millis)
    _bg_file = "{dir}\{bg}_background.png".format(dir=_dir,bg=_fn)
    cv2.imwrite(_bg_file, initFrame)

    _initF = Image.open(_bg_file)

    beens = {}
    for _l in [_ln.strip('\n') for _ln in open(lstfile).readlines() if _ln]:
        _, _f, _x, _y = _l.split('.')[0].split('_')
        _x = int(_x)
        _y = int(_y)
        if not os.path.exists(_l):
            continue
        beens.setdefault((_x, _y), []).append(_l)

    pprint.pprint(beens)
    _keys = beens.keys()

    _nn = 0
    for _V in itertools.combinations_with_replacement(_keys, _N):
        # cv2.waitKey(framePeriod) & 0xff
        # if k == 27:  # ESC
        #     break

        _V = list(_V)
        _V.sort(lambda _a, _b: _a[1] - _b[1], reverse=True)
        pprint.pprint(_V)
        frame = initFrame.copy()

        _bg = _initF.copy()

        final1 = Image.new("RGBA", _bg.size)
        final1.paste(_bg, (0,0), _bg)

        for _v in _V:
            # cv2.waitKey(framePeriod) & 0xff
            # if k == 27:  # ESC
            #     break

            _fs = beens.get(_v)
            if not _fs:
                continue
            _f = _fs[random.randint(0, len(_fs)-1)]

            # _img = cv2.imread(_f, cv2.IMREAD_UNCHANGED)


            # _dst = np.zeros_like(frame)
            # frame = cv2.im(frame, _img)
            #frame = cv2.addWeighted(frame[:,:,:3],1, _img[:,:,:3], _img[:,:,3], 0)

            _fg = Image.open(_f)
            # _bg.paste(_fg, (0, 0), _fg)
            _bg = Image.alpha_composite(_bg, _fg)

        # cv2.imshow('f', frame)
        _filename= '{dir}\{fn}_foreground_{n}.png'.format(dir=_dir, fn=_fn, n=_nn)
        _bg.save(_filename)
        # _bg.show()

        # cv2.imwrite("{bg}_foreground_{n}.png".format(n=_nn), frame)
        _nn += 1

        # cv2.waitKey(60)


        print "-"*10