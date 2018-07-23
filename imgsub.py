import os
import numpy as np
import cv2
from optparse import OptionParser


def parse_options():
    parser = OptionParser()

    parser.add_option("-i",
                      dest="mp4file",
                      help="input mp4 file",
                      type="string",
                      action="store"
                      )

    parser.add_option("-o",
                      dest="objtemplate",
                      help="template of object files",
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

    parser.add_option("--t0",
                      dest="startTime",
                      help="start time in sec",
                      type="float",
                      default=0.0,
                      action="store"
                      )

    parser.add_option("--t1",
                      dest="endTime",
                      help="end time in sec (if -1 then till the end of the stream)",
                      type="float",
                      default=-1.0,
                      action="store"
                      )

    (options, args) = parser.parse_args()

    if options.mp4file:
        return (options.mp4file, options.objtemplate, options.initTime, options.startTime, options.endTime)

    parser.print_help()
    quit(-1)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def getTracker(tracker_type = ''):
    # tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE']
    # tracker_type = tracker_types[2]
    import pprint
    pprint.pprint(dir(cv2))
    if int(minor_ver) < 3 or not tracker_type:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()

    return tracker

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h)

if __name__ == "__main__":

    mp4file, objtemplate, initTime, startTime, endTime = parse_options()
    if os.path.exists(mp4file) == False:
        print 'Input file not exist  %s' % mp4file
        quit(1)

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

    initGray = cv2.cvtColor(initFrame.copy(), cv2.COLOR_BGR2GRAY)
    # initGray = cv2.GaussianBlur(initGray , (7, 7), 0)  # remove noise to avoid false motion detection
    # initGray = cv2.fastNlMeansDenoising(initGray, None, 10, 10)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prevPts = cv2.goodFeaturesToTrack(initGray, mask=None, **feature_params)


    # initGray = cv2.GaussianBlur(initGray, (3, 3), 0)  # remove noise to avoid false motion detection
    # # initGray = cv2.medianBlur(initGray, 3)
    # initGray = cv2.blur(initGray, (7, 3))

    cv2.imshow('initFrame', initFrame)
    k = cv2.waitKey(framePeriod) & 0xff

    startF = int(fps * startTime)
    endF = int(fps * endTime)

    cap.set(cv2.CAP_PROP_POS_FRAMES, startF)

    # tracker = getTracker()
    # bbox = cv2.selectROI(initGray, False)
    #
    # # Initialize tracker with first frame and bounding box
    # ok = tracker.init(initGray, bbox)

    prevFrame = initFrame.copy()
    prevGray = initGray.copy()
    prevDelta = np.zeros_like(initGray)


    hsv = np.zeros_like(initFrame)
    # hsv = np.zeros_like(initFrame)

    # tr1 = cv2.createBackgroundSubtractorMOG2(30, 2.0, True)

    prevFgmask = np.zeros_like(initGray)
    # hsv[..., 1] = 255
    flow = None
    mask = np.zeros_like(initGray)
    color = np.random.randint(0, 255, (100, 3))
    for _n in xrange(endF - startF +1):
        k = cv2.waitKey(framePeriod) & 0xff
        if k == 27:  # ESC
            break

        for _ in xrange(2):
            ret, frame = cap.read()
        if not ret:
            break

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frameGray = cv2.GaussianBlur(frameGray, (3, 3), 0)  # remove noise to avoid false motion detection
        # frameGray = cv2.medianBlur(frameGray, 3)
        # frameGray = cv2.blur(frameGray, (7,3))
        # frameGray = cv2.fastNlMeansDenoising(frameGray, None, 10, 10)

        frameDelta = cv2.absdiff(initGray, frameGray)
        cv2.imshow('frameDelta', frameDelta)

        # fgmask = tr1.apply(frameGray)
        # cv2.imshow('fgmask', fgmask)

        #prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flag
        # flow = cv2.calcOpticalFlowFarneback(prevDelta, frameDelta, None, 0.5, 2, 15, 3, 5, 1.2, flags=0)
        # flow = cv2.calcOpticalFlowFarneback(prevDelta, frameDelta, flow, 0.5, 2, 15, 3, 5, 1.2, flags=0)
        flow = cv2.calcOpticalFlowFarneback(prevGray, frameGray, None, 0.5, levels=1, winsize=15,iterations= 1,poly_n= 5,poly_sigma= 1.1, flags=0)

        # print len(flow)

        # flow = cv2.calcOpticalFlowFarneback(prevDelta, frameDelta, None, 0.5, 2, 15, 3, 5, 1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # rgb = cv2.fastNlMeansDenoisingColored(rgb, None,10,10,37,21)
        cv2.imshow('flow', rgb)



        # cv2.filterSpeckles()
        # h, status = cv2.findHomography(initFrame, frame)


        # ok, bbox = tracker.update(frameDelta)
        #
        # if ok:
        #     # Tracking success
        #     p1 = (int(bbox[0]), int(bbox[1]))
        #     p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        #     cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        # else:
        #     # Tracking failure
        #     cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)






        rgbGray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        rgbGray = cv2.fastNlMeansDenoising(rgbGray, None,7,21)

        thresh = cv2.threshold(rgbGray, 25, 255, cv2.THRESH_BINARY)[1]
        # thresh = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY)[1]


        # kernel = np.ones((50, 50), np.uint8)
        # erosion = cv2.dilate(thresh, kernel, iterations=1)
        # closing = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)

        # img_, cnts, hie_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_, contours, hie_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt, 0, True) for cnt in contours]
        contours = [c for c in contours if cv2.contourArea(c) > 200]


        # contours2 = []
        # for _i in xrange(len(contours)):
        #     for _ii in xrange(_i +1, len(contours)):
        #         c = contours[_i]
        #         cc = contours[_ii]
        #         if not intersection(cv2.boundingRect(c), cv2.boundingRect(cc)):
        #             continue
        #         c = cv2.convexHull(c + cc)
        #
        #     contours2.append(c)
        # contours = contours2




        # thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # # kernel = np.ones((5, 5), np.uint8)
        # # # erosion = cv2.dilate(thresh, kernel, iterations=1)
        # # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        #
        # # img_, cnts, hie_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # img_, cnts, hie_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = cnts#[cv2.approxPolyDP(cnt, 0, True) for cnt in cnts]




        # def min_box(c):
        #
        #     if cv2.contourArea(c) < 500:
        #          return False
        #     r= cv2.boundingRect(c)
        #
        #     w= 0.0 + abs(r[0] - r[2])
        #     h= 0.0 + abs(r[1] - r[3])
        #     print r, w, h,
        #
        #     if h <2 or w <2 :#or h > height/2 or w > width/2:
        #         print
        #         return False
        #     dd = w / h
        #     print dd
        #     return dd > 0.9 and dd < 4
        # loop over the contours
        # contours = [c for c in contours if min_box(c)]

        # vis = np.zeros((height, width, 3), np.uint8)
        #                image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]



        mask = np.zeros_like(frame)  # Create mask where white is what we want, black otherwise


        cv2.drawContours(mask, contours, -1, 255, -1)  # Draw filled contour in mask
        out = np.zeros_like(frame, subok=True)  # Extract out the object and place into output image
        out[mask == 255] = frame[mask == 255]

        contours2 = []
        centers = []

        nextPts = cv2.goodFeaturesToTrack(frameGray, mask=None, **feature_params)

        for c in contours:
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            if center[1] < height/3:
                continue
            radius = int(radius)
            radius2 = radius * radius
            # a1= cv2.contourArea(c)
            # a2= np.pi * radius *radius
            # if a1 < a2/3:
            #     continue

            import math
            def _dist(_p):
                _h = center[0] - _p[0][0]
                _w = center[1] - _p[0][1]
                return _w *_w + _h * _h <= radius2
            _pass = False
            for _p in nextPts:
                if _dist(_p):
                    _pass = True
                    break
            if not _pass:
                continue

            cv2.circle(out, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            contours2.append(c)
            centers.append(center)

        good_new = nextPts

        _i = 0
        for p in good_new:
            a, b = p.ravel()
            frame = cv2.circle(frame, (a, b), 5, color[_i].tolist(), -1)
            _i += 1

        centers = list(set(centers))
        contours = contours2


        # flow = cv2.calcOpticalFlowFarneback(frameGray, initGray, None, 0.5, 2, 15, 3, 5, 1.2,
        #                                     flags=cv2.OPTFLOW_USE_INITIAL_FLOW)


        # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
        #
        # rgbGray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # disparity = stereo.compute(rgbGray, initGray)
        #
        # cv2.imshow('disparity', disparity)


        # ret = np.where(mask == 255)
        # x, y = ret[:2]
        # topx, topy = (np.min(x), np.min(y))
        # bottomx, bottomy = (np.max(x), np.max(y))
        # out = out[topx:bottomx + 1, topy:bottomy + 1]

        # _cc = 10
        # print "#", len(contours)
        # for c in contours:
        #     cv2.drawContours(vis, [c], -1, (_cc, 255 - _cc, 255-_cc), 3, 4)
        #     _cc += 10
        cv2.imshow('contours', out)

        cv2.imshow('frame', frame)



        # prevFrame = frame.copy()
        prevGray = frameGray.copy()
        prevDelta = frameDelta.copy()
        prevPts = nextPts
        # prevFgmask = fgmask


        # waitKeys(framePeriod)

    cv2.destroyAllWindows()
    cap.release()