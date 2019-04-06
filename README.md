# Globus Object Detection Web App

![Logo LRI](img/Logo-LRI.jpg)


## Executive summary
A brief documentation describing a grocery store object detection web app built for Globus supermarket.

## Environment
We use Docker to build ready-to-deploy image. Which is a virtual operating system with all the requirements that contains a running web server hosting a REST web service. In addition to the web server, we include the trained (Deep Neural Network) DNN model in the image files.

## Web server
We use FLASK as a web server to host the REST web service which will operate as an interface on top of the trained model. Flask is implemented in Python, therefore integrating  the REST web service on top of the trained model should be smooth since both are written in python.

## Model Wrapper
We are using a python wrapper to load the trained model into GPU’s memory. The wrapper uses libraries built from the source code written in C. Integrating the wrapper and the web service is smooth since both are written in python and require few packages to be pre-installed. The wrapper is used to load two separate models, the first is used for grocery product detection and the second is for grocery tags detection. Due to GPU memory problems we can load a single model at a time. We’ll elaborate more on this subject in the deployment section.

## Web service structure
The web service which can be found in (file app.py) include the following:

### Detector Class
The class contains an initialization method to load the trained DNN model and meta information. The init method loads the trained model and meta information into the class context and it is called when a new object instance of the class is created  when we first start the web server. In addition to the init method the class contains three main methods:

#### get_state()
This method returns the state of the trained model (loaded, not loaded) in the class context.

#### detect_image()
This method uses the loaded model to predict objects in a still image. It takes two inputs: the filename which was uploaded using the web service and a detection choice forwarded by the calling webmethod. The detection choice is used to decide what to detect (grocery product, grocery tag). This parameter will be removed in the future where we’ll merge both models into a single model. The method returns a list structure that contains a boolean indicator (results, no results) and a dictionary list of objects (if found), in addition to the elapsed time to perform the detection.

#### detect_video()
This method loops over a video file to extract frames and uses the loaded model to predict objects in each extracted frame. The method uses a single temporary variable to load each extracted frame and perform the detection. It takes two inputs as the previous method: the filename which was uploaded using the web service and a detection choice forwarded by the calling webmethod. The method returns  a list structure that contains a boolean indicator (results, no results) and a list of dictionary lists (one per frame - if any objects are found), in addition to the elapsed time to perform the detection.

### Web Methods
The web methods are the functionality of the REST web service, each method calls a different method from Detector class. The web methods are registered at the route of the Flask app. 

#### /state
This method calls get_state() method from Detector class and forwards the return as a JSON string.

#### /detect_img
This method calls detect_image() method from Detector class and forwards the return as a json array of objects.

#### /detect_vid
This method calls detect_video() method from Detector class and forwards the return as a json array of arrays.

In the following table we list the web methods along with their expected request inputs and response outputs. The request inputs for all methods are structured as form data and the outputs are in JSON formatting. For testing we use a REST API client e.g. Postman or Curl.

| __Method__ | __Type__ | __Input__ | __Output__ (expected_responses folder) |
| ---------- |----------|---------- |---------- |
|   /state   | GET      | (empty)   | response_state.json |
| /detect_img| POST     | Imagefile (PNG,JPG,JPEG) ; to_detect (product,tag) | response_tag_img.json response_product_img.json |
| /detect_vid| POST     | vidfile (AVI,MP4) ; to_detect (product,tag) | response_tag_vid.json response_product_vid.json |


By looking at the JSON files of detections we can notice three parameters in the response, in the following table we show possible values for each parameter

| __Parameter__ | __Description__ | __Possible values__ |
|---------------|-----------------|---------------------|
| objs, objects | detected objects | [], dictionary, list|
| cls | object's class | class name, unique ID |
| h | height of detected object in pixels |  integer |
| w | width of detected object in pixels | integer |
| x | object's upper left corner X coord. | integer |
| y | object's upper left corner Y coord. | integer |
| t | elapsed time for detection | seconds |
| ret | return state | objects found, empty, detectiontype error ,filetype error, error |


The 'ret' parameter explains the state of the REST web service response which can be:
* _objects_ _found_: it means that the detection model was able to detect objects in an image or any frame in the video
* _empty_: it means no objects were detected in an image or a any frame in the video
* _detectiontype_ _error_: indicates that the value of 'to_detect' parameter in the web request is not linked with the current detection model. Since the current implementation includes two models, the user should specify which object to detect (tag or product). This value has to be compatible with the depolyed model e.g. if we deploy the 'grocery tags' model the value of 'to_detect' parameter should be 'tag'.
* _filetype_ _error_: indicates that the uploaded file type is not in the list of allowed types.
* _error_: indicates a general error in the web application that has to be reviewed by the developer.

### Static Methods
There are some static methods included in the application for debugging or file handling purposes. The methods are:
* allow_file() checks if the uploaded file extension in the list of allowed extensions.
* cvDrawBoxes() draws bounding boxes extracted from the detection results of the trained model.
* convertBack() called for converting detected object information to bounding box coordinates.

## Deployment
To deploy the web application on a local server follow these steps:
1. Install Docker or Nvidi-Docker version 18.09.1 and above
2. Clone the repository to your local machine
3. Init darknet submodule by executing the following command:
```
git submodule init
git submodule update
```
4. Copy the config files (cfg) and the data files (.data .names) from "globus_web_app_config_files" folder to "darknet/data" and "darknet/cfg" folders where the folder and file structure should be as follows:
```
+globus_web_app
|	+--darknet
|	|     +--cfg
|	|     |	   +--globus13-yolov3.cfg
|	|     |    +--globustags-yolov3.cfg
|	|     +--data
|	|     |    +--globus_13.data
|	|     |    +--globus_13.names
|	|     |    +--globus_tags.data
|	|     |    +--globus_tags.names
```
5. Download models weights files from [Here](https://www.lri.fr/owncloud/index.php/s/tl6DSkTrqNMY346) and [Here](https://www.lri.fr/owncloud/index.php/s/CoSIqoJmg4dyJ87) Then, move them into “darknet/backup” folder
6. Edit Dockerfile contents to enable/disable GPU support
If you want to disable GPU support or do not have Nvidia GPU set GPU, CUDNN, and CUDNN_HALF to 0 in Dockerfile and comment the following line
“CMD nvidia-smi -q”. If you want to keep GPU support keep the Dockerfile as is.
7. Edit Dockerfile contents to choose web app entry point
   * If you want to use DNN model for detecting products, under [“ENTRYPOINT”], set the name of the file “app.py”
   * If you want to use DNN model for detecting tags, under [“ENTRYPOINT”], set the name of the file “app2.py”
8. Run Docker build inside the repository folder by executing the following command:
```
docker (or nvidia-docker) build -t [image_name_of_choice]:latest .
```
9. After building the image successfully run a docker container by executing the following command:
```
docker (or nvidia-docker) run -d -p 5000:5000 [image_name_set_in_6]:latest
```

The last command creates a new docker container and forwards the port 5000 to the host machine. Currently, the web app operates on port 5000 which can be changed later.