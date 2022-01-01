# OAK_Lite_AutoCam
Auto PTZ camera app using Open CV AI Kit Lite

Tested on Windows 11 + Open CV AI Kit Lite Fixed focus model.

## Dependencies
### Libraries
You should install libraries described in the requirements.txt 
### Virtual Camera App
You need to install a virtual camera app (OBS Studio or Unity Capture).
Please see [letmaik/pyvirtualcam](https://github.com/letmaik/pyvirtualcam) for details.
### mobilenet-ssd model blob
You need to modify the path to the mobilenet-ssd blob written in the auto_pan_camera.py.

If you installed [luxonis/depthai-python](https://github.com/luxonis/depthai-python/tree/main/examples) and run the install_requirements.py in its /example/ directory, you can find the model blob under the /examples/models/mobilenet-ssd/.