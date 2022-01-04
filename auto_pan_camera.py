#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import math
import pyvirtualcam
from enum import Enum

class MoveState(Enum):
    WAIT = 0
    MOVE = 1

class frameMover:
    def __init__(self,maxSize,gain = 0.1):
        self.gain = gain
        self.targetPos = [0,0]
        self.currentPos = [0,0]
        self.maxSize = maxSize
        self.medianFilter = [0] * 30

    def move(self,bbox,frameSize):
        centroid = self.getCentroid(bbox)

        if self.checkPosUpdate(centroid) == True:
            self.targetPos = centroid

        moveSpeed = self.getMoveSpeed(self.targetPos,self.currentPos)
        
        self.updatePos(moveSpeed,frameSize)
        
        return self.currentPos

    def getCentroid(self,rect):
        return ((rect[0]+rect[2]) / 2.0,(rect[1]+rect[3]) / 2.0)

    def getMoveSpeed(self,targetPos,currentPos):
        xSpeed = (targetPos[0] - currentPos[0])*self.gain        
        ySpeed = (targetPos[1] - currentPos[1])*self.gain
        return(xSpeed,ySpeed)

    def checkPosUpdate(self,pos):
        dist = math.dist(pos,self.targetPos)

        self.medianFilter.append(dist)
        del self.medianFilter[0]

        median = np.median(self.medianFilter)
        
        print("Pos:",median)
        if median > 100:
            return True
        else:
            return False

    def getFrameCenterPos(self,rectSize,pos):
        retXPos = pos[0]
        retYPos = pos[1]
        if (retXPos - ((rectSize[0]+1) // 2)) < 0:
            retXPos = (rectSize[0] + 1) // 2
        if (retYPos - ((rectSize[1]+1) // 2)) < 0:
            retYPos = (rectSize[1] + 1) // 2

        if (retXPos + ((rectSize[0]+1) // 2)) > self.maxSize[0]:
            retXPos = self.maxSize[0] - (rectSize[0] + 1) // 2
        if (retYPos + ((rectSize[1]+1) // 2)) > self.maxSize[1]:
            retYPos = self.maxSize[1] - (rectSize[1] + 1) // 2  

        return [retXPos,retYPos]

    def updatePos(self,moveSpeed,frameSize):
        self.currentPos[0] = self.currentPos[0] + moveSpeed[0]
        self.currentPos[1] = self.currentPos[1] + moveSpeed[1]
        
        self.currentPos = self.getFrameCenterPos(frameSize, self.currentPos)


class frameZoomer:
    def __init__(self,maxSize,minSize,outXYRatio,gain,marginRatio):
        self.gain = gain
        self.marginRatio = marginRatio    
        self.outXYRatio = outXYRatio
        self.minSize = np.copy(minSize)
        self.maxSize = maxSize
        self.targetSize = np.copy(minSize)
        self.currentSize = np.copy(minSize)
        self.currentZoomRatio = 1.0
        self.baseSize = self.minClip([(minSize[0] - 1),(minSize[1] - 1)], minSize, outXYRatio)
        self.baseSize[0] = int(self.baseSize[0])
        self.baseSize[1] = int(self.baseSize[1])
        self.medianFilter = [1.0] * 30
    def zoom(self,bbox):
        cropSize = self.getCropFrameSize(bbox)

        if self.checkSizeUpdate(cropSize) == True:
            self.targetSize = cropSize
        
        zoomSpeed = self.getZoomSpeed(self.targetSize,self.currentSize)
        self.updateSize(zoomSpeed)

        return (self.currentSize[0],self.currentSize[1])

    def checkSizeUpdate(self,size):
        ratio = max(self.targetSize[0]/size[0],size[0]/self.targetSize[0])

        self.medianFilter.append(ratio)
        del self.medianFilter[0]

        median = np.median(self.medianFilter)

        print("Size:",median)
        if median > 1.02:
            return True
        else:
            return False      

    def getCropFrameSize(self,bbox):
        boxSize = [bbox[2] - bbox[0],bbox[3] - bbox[1]]
        (targetXSize,targetYSize) = self.getIncludingRect(boxSize)
        return (int(targetXSize + 0.5),int(targetYSize + 0.5))
    
    def getIncludingRect(self,size):
        #Add a margin
        xSize = size[0] * (1.0 + self.marginRatio[0])
        ySize = size[1] * (1.0 + self.marginRatio[1])

        boxXYRatio =  xSize / ySize 
        targetSize = [0,0]
        #if bbox has a wider aspect ratio than the target frame's one
        if boxXYRatio > self.outXYRatio:
            targetSize[0] = xSize
            targetSize[1] = xSize / self.outXYRatio
        else:
            targetSize[0] = ySize * self.outXYRatio
            targetSize[1] = ySize

        #Clip with max/min size
        (targetSize[0], targetSize[1]) = self.maxClip(targetSize,self.maxSize,self.outXYRatio)
        (targetSize[0], targetSize[1]) = self.minClip(targetSize,self.minSize,self.outXYRatio)              
        return targetSize

    def maxClip(self,targetSize,maxSize,outXYRatio):
        maxXYRatio = maxSize[0] / maxSize[1]
        if targetSize[0] > maxSize[0] or targetSize[1] > maxSize[1]:
            if outXYRatio > maxXYRatio:
                targetSize[0] = maxSize[0]
                targetSize[1] = maxSize[0] / outXYRatio
            else:
                targetSize[0] = maxSize[1] * outXYRatio 
                targetSize[1] = maxSize[1]

        return targetSize

    def minClip(self,targetSize,minSize,outXYRatio):
        minXYRatio = minSize[0] / minSize[1]
        if targetSize[0] < minSize[0] or targetSize[1] < minSize[1]:
            if outXYRatio < minXYRatio:
                targetSize[0] = minSize[0]
                targetSize[1] = minSize[0] / outXYRatio
            else:
                targetSize[0] = minSize[1] * outXYRatio 
                targetSize[1] = minSize[1]
                       
        return targetSize   

    def getZoomSpeed(self,targetSize,currentSize):
        speed = 1.0 + ((targetSize[0] / currentSize[0]) - 1.0) * self.gain
        return speed

    def updateSize(self,zoomSpeed):
        self.currentZoomRatio = self.currentZoomRatio * zoomSpeed
        self.currentSize[0] = int(self.baseSize[0] * self.currentZoomRatio + 0.5)
        self.currentSize[1] = int(self.baseSize[1] * self.currentZoomRatio + 0.5)

class imageCropper:
    def __init__(self,maxSize,minSize=[640,360],outSize=[1280,720],marginRatio=[0.05,0.05]):
        self.marginRatio = marginRatio
        self.mover = frameMover(maxSize)
        self.outSize=outSize
        self.zoomer = frameZoomer(maxSize,minSize,outSize[0]/outSize[1],gain=0.05,marginRatio=marginRatio)

    def cropFrame(self,inFrame,bbox):
        currentSize = self.zoomer.zoom(bbox)
        currentPos = self.mover.move(bbox,currentSize)

        targetRect = self.convCenterSizeToRect(currentSize,currentPos)
        resized = np.zeros(self.outSize)
        resized = cv2.resize(inFrame[targetRect[1]:targetRect[3],targetRect[0]:targetRect[2]:],dsize=self.outSize)
        return resized

    def convCenterSizeToRect(self,size,pos):
        targetXmin = int(pos[0] - (size[0]) // 2)
        targetYmin = int(pos[1] - (size[1]) // 2)
        targetXmax = int(targetXmin + size[0])
        targetYmax = int(targetYmin + size[1])
        return(targetXmin,targetYmin,targetXmax,targetYmax)

# Get argument first
nnPath = str((Path(__file__).parent / Path('./models/mobilenet-ssd_openvino_2021.4_5shave.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnPath = sys.argv[1]

if not Path(nnPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutPreview = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)
controlIn = pipeline.create(dai.node.XLinkIn)

xoutVideo.setStreamName("video")
xoutPreview.setStreamName("preview")
nnOut.setStreamName("nn")
controlIn.setStreamName('control')

# Properties
camRgb.setPreviewSize(300, 300)    # NN input
camRgb.setVideoSize(4192,3120)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_13_MP)
camRgb.setInterleaved(False)
camRgb.setPreviewKeepAspectRatio(False)
# Define a neural network that will make predictions based on the source frames
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Linking
camRgb.video.link(xoutVideo.input)
camRgb.preview.link(xoutPreview.input)
camRgb.preview.link(nn.input)
nn.out.link(nnOut.input)
controlIn.out.link(camRgb.inputControl)

# Connect to device and start pipeline
with dai.Device(pipeline) as device, pyvirtualcam.Camera(width=1280, height=720, fps=15,fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    print(f'Using virtual camera: {cam.device}')

    # Output queues will be used to get the frames and nn data from the outputs defined above
    qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    qPreview = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    controlQueue = device.getInputQueue('control')

    previewFrame = None
    videoFrame = None
    detections = []
    cropROIs = [[0,0,1000,1000],]
    frameX = 1000
    frameY = 1000
    marginX = 50
    marginY = 50
    outXSize = 1280
    outYSize = 720

    ctrl = dai.CameraControl()
    ctrl.setAutoExposureEnable()
    ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
    ctrl.setChromaDenoise(4)
    ctrl.setLumaDenoise(4)
    ctrl.setAutoExposureCompensation(5)
    controlQueue.send(ctrl)

    cropper = imageCropper(camRgb.getVideoSize())

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frameX,frameY, bbox):
        normVals = np.full(len(bbox), frameY)
        normVals[::2] = frameX
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def filterBbox(det,classIds):
        ret = []
        for detection in detections:
            if detection.label in classIds:
                bbox = frameNorm(frameX,frameY, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                ret.append((detection.confidence,list(bbox)))
        ret.sort(key=lambda x: x[0],reverse=True)
        return ret   

    def cropFrame(name, frame):
        resized = cropper.cropFrame(frame,cropROIs[0])
        # Show the frame
        cv2.imshow(name, resized)
        cam.send(resized)
        cam.sleep_until_next_frame()

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("video", outXSize//4, outYSize//4)
    print("Resize video window with mouse drag!")

    while True:
        # Instead of get (blocking), we use tryGet (nonblocking) which will return the available data or None otherwise
        inVideo = qVideo.tryGet()
        inDet = qDet.tryGet()

        if inVideo is not None:
            videoFrame = inVideo.getCvFrame()
            frameX = videoFrame.shape[1]
            frameY = videoFrame.shape[0]

        if inDet is not None:
            detections = inDet.detections
            bbox = filterBbox(detections, [15])
            if len(bbox) > 0:
                cropROIs = [r[1] for r in bbox]

        if videoFrame is not None:
            cropFrame("video", videoFrame)

        if cv2.waitKey(1) == ord('q'):
            break
