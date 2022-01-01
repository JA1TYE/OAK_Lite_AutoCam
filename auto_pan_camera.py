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
    def __init__(self,speed = 50,moveThresh = 100):
        self.state = MoveState.WAIT
        self.currentPos = [0,0]
        self.targetPos = [0,0]
        self.speed = speed
        self.ySpeed = 0.0
        self.xSpeed = 0.0
        self.moveThresh = moveThresh

    def move(self,pos):
        #print("Move:",self.state,pos,self.targetPos,self.currentPos,self.xSpeed,self.ySpeed)
        if self.state == MoveState.WAIT:
            dist = math.dist(pos, self.currentPos)
            if dist > self.moveThresh:
                self.targetPos = pos
                self.xSpeed = (self.targetPos[0] - self.currentPos[0])/self.speed
                self.ySpeed = (self.targetPos[1] - self.currentPos[1])/self.speed
                self.state = MoveState.MOVE
        elif self.state == MoveState.MOVE:
            diffTarget = math.dist(pos, self.targetPos)
            if diffTarget > self.moveThresh * 5:
                self.targetPos = pos
                self.xSpeed = (self.targetPos[0] - self.currentPos[0])/self.speed
                self.ySpeed = (self.targetPos[1] - self.currentPos[1])/self.speed

            xPos = self.currentPos[0] + self.xSpeed + 0.5
            yPos = self.currentPos[1] + self.ySpeed + 0.5

            if (xPos > self.targetPos[0] and self.currentPos[0] <= self.targetPos[0]) or\
                (xPos < self.targetPos[0] and self.currentPos[0] >= self.targetPos[0]):
                xPos = self.targetPos[0]             
                self.xSpeed = 0.0            
            if (yPos > self.targetPos[1] and self.currentPos[1] <= self.targetPos[1]) or\
                (yPos < self.targetPos[1] and self.currentPos[1] >= self.targetPos[1]):
                yPos = self.targetPos[1]            
                self.ySpeed = 0.0 
            if self.xSpeed == 0.0 and self.ySpeed == 0.0:
                self.state = MoveState.WAIT 
            self.currentPos[0] = xPos
            self.currentPos[1] = yPos
        
        return (int(self.currentPos[0]),int(self.currentPos[1]))
    

class frameZoomer:
    def __init__(self,speed = 0.005,zoomThresh = 1.05):
        self.state = MoveState.WAIT
        self.currentSize = [1280,720]
        self.currentZoomRatio = 1.0
        self.targetSize = [0,0]
        self.zoomSign = -1
        self.speed = speed
        self.zoomThresh = zoomThresh

    def zoom(self,size):
        #print("Zoom:",self.state,size,self.targetSize,self.currentSize)
        if self.state == MoveState.WAIT:
            if size[0]/self.currentSize[0] > self.zoomThresh:
                self.targetSize = size
                self.zoomSign = +1
                self.state = MoveState.MOVE
            elif self.currentSize[0]/size[0] > self.zoomThresh:
                self.targetSize = size
                self.zoomSign = -1
                self.state = MoveState.MOVE
        elif self.state == MoveState.MOVE:
            xSize = self.currentSize[0] * (1 + self.zoomSign * self.speed)
            ySize = self.currentSize[1] * (1 + self.zoomSign * self.speed)
            if (self.currentSize[0] > self.targetSize[0] and xSize <= self.targetSize[0]) or\
                 (self.currentSize[0] < self.targetSize[0] and xSize >= self.targetSize[0]):
                self.currentSize[0] = self.targetSize[0]
                self.currentSize[1] = self.targetSize[1]
                self.state = MoveState.WAIT
            else:
                self.currentSize[0] = xSize
                self.currentSize[1] = ySize
        return (int(self.currentSize[0]),int(self.currentSize[1]))        

class imageCropper:
    def __init__(self,inXSize,inYSize,outXSize,outYSize,minXSize = 640,minYSize = 360,xMarginRatio=0.05,yMarginRatio=0.05):
        self.inXSize = inXSize
        self.inYSize = inYSize
        self.outXSize = outXSize
        self.outYSize = outYSize
        self.minXSize = minXSize
        self.minYSize = minYSize
        self.xMarginRatio = xMarginRatio
        self.yMarginRatio = yMarginRatio
        self.outXYRatio = self.outXSize / self.outYSize
        self.mover = frameMover()
        self.zoomer = frameZoomer()

    def cropFrame(self,inFrame,bbox):
        centroid = self.getBboxCentroid(bbox)
        centroid = self.mover.move(centroid)

        cropSize = self.getCropFrameSize(bbox)
        cropSize = self.zoomer.zoom(cropSize)

        cropPos = self.getFrameCenterPos(cropSize, centroid)
        targetRect = self.convCenterSizeToRect(cropPos,cropSize)

        #print(targetRect)
        resized = np.zeros((outXSize,outYSize))
        resized = cv2.resize(inFrame[targetRect[1]:targetRect[3],targetRect[0]:targetRect[2]:],dsize=(self.outXSize,self.outYSize))
        return resized

    def convCenterSizeToRect(self,pos,size):
        targetXmin = int(pos[0] - (size[0]) // 2)
        targetYmin = int(pos[1] - (size[1]) // 2)
        targetXmax = int(targetXmin + size[0])
        targetYmax = int(targetYmin + size[1])
        return(targetXmin,targetYmin,targetXmax,targetYmax)

    def getBboxCentroid(self,bbox):
        return ((bbox[0]+bbox[2]) // 2,(bbox[1]+bbox[3]) // 2)
    
    def getCropFrameSize(self,bbox):
        boxXSize = bbox[2] - bbox[0]
        boxYSize = bbox[3] - bbox[1]
        (targetXSize,targetYSize) = self.getIncludingRect(boxXSize, boxYSize)
        return (targetXSize,targetYSize)
    
    def getIncludingRect(self,xSize,ySize):
        #Add a margin
        xSize = xSize * (1.0 + self.xMarginRatio)
        ySize = ySize * (1.0 + self.yMarginRatio)

        boxXYRatio =  xSize/ ySize 
        #if bbox has a wider aspect ratio than the target frame's one
        if boxXYRatio > self.outXYRatio:
            targetXSize = xSize
            targetYSize = xSize / self.outXYRatio
        else:
            targetXSize = ySize * self.outXYRatio
            targetYSize = ySize
        
        #Clip with max/min size
        (targetXSize, targetYSize) = self.maxClip(targetXSize, targetYSize,self.inXSize,self.inYSize,self.outXYRatio)
        (targetXSize, targetYSize) = self.minClip(targetXSize, targetYSize,self.minXSize,self.minYSize,self.outXYRatio)              
        
        return (int(targetXSize + 0.5),int(targetYSize + 0.5))

    def maxClip(self,targetXSize,targetYSize,maxXSize,maxYSize,outXYRatio):
        maxXYRatio = maxXSize / maxYSize
        if targetXSize > maxXSize or targetYSize > maxYSize:
            if outXYRatio > maxXYRatio:
                targetXSize = maxXSize
                targetYSize = maxXSize / outXYRatio
            else:
                targetXSize = maxYSize * outXYRatio 
                targetYSize = maxYSize

        return (targetXSize,targetYSize)

    def minClip(self,targetXSize,targetYSize,minXSize,minYSize,outXYRatio):
        minXYRatio = minXSize / minYSize
        if targetXSize < minXSize or targetYSize < minYSize:
            if outXYRatio < minXYRatio:
                targetXSize = minXSize
                targetYSize = maxXSize / outXYRatio
            else:
                targetXSize = minYSize * outXYRatio 
                targetYSize = minYSize
                       
        return (targetXSize,targetYSize)   

    def getFrameCenterPos(self,rectSize,centroidPos):
        retXPos = centroidPos[0]
        retYPos = centroidPos[1]
        if (retXPos - ((rectSize[0]+1) // 2)) < 0:
            retXPos = (rectSize[0] + 1) // 2
        if (retYPos - ((rectSize[1]+1) // 2)) < 0:
            retYPos = (rectSize[1] + 1) // 2

        if (retXPos + ((rectSize[0]+1) // 2)) > self.inXSize:
            retXPos = self.inXSize - (rectSize[0] + 1) // 2
        if (retYPos + ((rectSize[1]+1) // 2)) > self.inYSize:
            retYPos = self.inYSize - (rectSize[1] + 1) // 2        

        return(retXPos,retYPos)

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

    cropper = imageCropper(camRgb.getVideoWidth(),camRgb.getVideoHeight(),1280,720)

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
