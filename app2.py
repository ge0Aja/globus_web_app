import flask
import os
import logging
import time
import datetime
import werkzeug
import sys
import urllib
import traceback
from json import dumps
import cv2
import imutils


base_path = os.getcwd()

py_path = os.path.join(base_path,'darknet/python')
darknet_path = os.path.join(base_path,'darknet')
upload_path = 'uploads'

sys.path.append(py_path)
sys.path.append(darknet_path)

import darknet as dn

UPLOAD_FOLDER = os.path.join(base_path,'uploads')

ALLOWED_IMAGE_EXTENSIONS = set(['png','jpg','jpeg'])
ALLOWED_VIDEO_EXTENSIONS = set(['avi','mp4'])
ALLOWED_DETECTIONS = set(['tag','product'])

app = flask.Flask(__name__)


# STATICMETHOD: used for debugging and drawing on image or video for results comparison
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

# STATICMETHOD: used for debugging and drawing on image or video for results comparison
def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 3)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

# STATICMETHOD: check uploaded files name formatting and extension
def allow_file(filename,t):
    if t == 'IMG':
        return ('.' in filename and filename.split('.')[-1] in ALLOWED_IMAGE_EXTENSIONS)
    else:
        return ('.' in filename and filename.split('.')[-1] in ALLOWED_VIDEO_EXTENSIONS)

# WEBMETHOD: get the state of the loaded model
@app.route('/state',methods=['GET'])
def state():
    r = app.det.get_state()
    try:
        return flask.jsonify(state=r)
    except:
        logging.info('Error in state method\n')
        traceback.print_exc()
        return flask.jsonify(state='error')
    
# WEBMETHOD: detect objects in an uploaded image
@app.route('/detect_img',methods=['POST'])
def detect_img():
    #get the image load from the request
    try:
        imagefile = flask.request.files['imagefile'] # name is a variable
        to_detect = flask.request.form['to_detect']
        # check if in allowed detection types 
        if not to_detect in ALLOWED_DETECTIONS:
            return flask.jsonify(ret='detectiontype_error')
        # wrap the file name and check its extension
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER,filename_)
        logging.info('New File name %s' % (str(filename)))

        if allow_file(filename,'IMG'): 
            imagefile.save(filename)

            logging.info('Save Uploaded file to %s' % (str(filename)))
            #call detect_image
            rtn = app.det.detect_image(filename,to_detect)
            #process the return if True and JSONIFY
            if rtn[0]:
                return flask.jsonify(ret='objects_found',objs=rtn[1],t=rtn[2])
            else:
                return flask.jsonify(ret='empty')
        else:
            logging.info('File is not allowed %s' % (str(filename)))
            return flask.jsonify(ret='filetype_error')
            
    except:
        #log error and print, we can check the error from docker logs outside the image
        logging.info('Error in image detect method\n')
        traceback.print_exc()
        return flask.jsonify(ret='error')

# WEBMETHOD: detect objects in an uploaded video 
@app.route('/detect_vid',methods=['POST'])
def detect_vid():
    try:
        #get video load from the request
        vidfile = flask.request.files['vidfile'] # name is a variable
        to_detect = flask.request.form['to_detect']
        # check if in allowed detection types 
        if not to_detect in ALLOWED_DETECTIONS:
            return flask.jsonify(ret='detectiontype_error')
        # wrap the file name and check its extension
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(vidfile.filename)
        filename = os.path.join(UPLOAD_FOLDER,filename_)
        logging.info('New File name %s' % (str(filename)))

        if allow_file(filename,'VID'): 
            vidfile.save(filename)
            logging.info('Save Uploaded video file to %s' % (str(filename)))
            # call the video detection method
            rtn = app.det.detect_video(filename,to_detect)

            if rtn[0]:
                # we have results
                return flask.jsonify(ret='objects_found',objs=rtn[1],t=rtn[2])
            else:
                return flask.jsonify(ret='empty')
        else:
            logging.info('File is not allowed %s' % (str(filename)))
            return flask.jsonify(ret='filetype_error')
    except:
        #log error and print, we can check the error from docker logs outside the image
        logging.info('Error in vid detect method\n')
        traceback.print_exc()
        return flask.jsonify(ret='error')

# Detector class contains
# init()
# get_state()
# detect_image()
# detect_video
class Detector():
    def __init__(self):
        #define weights, data, cfg path and load net, paths should be in utf-8
        base_path = os.getcwd()
        #data_path = 'darknet/data/globus_13.data'
        data_path2 = 'darknet/data/globus_tags.data'
        #cfg_path = 'darknet/cfg/globus13-yolov3.cfg'
        cfg_path2 = 'darknet/cfg/globustags-yolov3.cfg'
        #weights_path = 'darknet/backup/globus13-yolov3_2000.weights'
        weights_path2 = 'darknet/backup/globustags-yolov3_1000.weights'

        #read gtins and save them to a data structure
        reader =csv.DictReader(open('globus_13_gtins.csv','r'))
        dict_l = []
        dict_g = {}
        for line in reader:
            dict_l.append(line) 

        for coi in dict_l:
            dict_g[coi['class']] = coi['gtin']


        
        #load network and meta using darkflow library
        #self.net = dn.load_net(os.path.join(base_path,cfg_path).encode('utf-8'),os.path.join(base_path,weights_path).encode('utf-8'),0)
        #self.meta = dn.load_meta(os.path.join(base_path,data_path).encode('utf-8'))

        self.net2 = dn.load_net(os.path.join(base_path,cfg_path2).encode('utf-8'),os.path.join(base_path,weights_path2).encode('utf-8'),0)
        self.meta2 = dn.load_meta(os.path.join(base_path,data_path2).encode('utf-8'))

        
    def get_state(self):
        #check if the model is loaded then the state is ready
        net1 = False
        net2 = False
        
        #if self.net:
        #    net1= True

        if self.net2:
            net2 = True

        return dict(net1_state=net1,net2_state=net2)

    # detect image for groceries or tags
    def detect_image(self, filename,to_detect):
        try:
            s= time.time()

            #if to_detect == 'product':
            #    res = dn.detect(self.net,self.meta,filename.encode('utf-8'))
            #else:
            #    return(False,[],0)
            
            if to_detect == 'tag':
                res = dn.detect(self.net2,self.meta2,filename.encode('utf-8'))
            else:
                return(False,[],0)

            e = time.time() - s
            if len(res) > 0:
                # process the output into a dictionary
                i=0
                lis = []
                while(i < len(res)):
                    class_type = res[i][0].decode('utf-8')
            
                    center_x = int(res[i][2][0])
                    center_y = int(res[i][2][1])
                    width = int(res[i][2][2])
                    height = int(res[i][2][3])

                    UL_x = int(center_x - width/2) #Upper Left corner X coord
                    #sometimes we get negative coords when the box is outside the image or frame
                    if UL_x < 0:
                        UL_x = 0
                    UL_y = int(center_y + height/2) #Upper left Y
                    LR_x = int(center_x + width/2)
                    LR_y = int(center_y - height/2)
                    if LR_y < 0:
                        LR_y = 0
                    #old return
                    #lis.append(dict(cls=class_type,x=UL_x,y=UL_y,w=width,h=height))
                    #new return (dummy data gtin and price)
                    #the new return contains dummy data for the sake of project completion, this section will be updated when the new model is uploaded
                    lis.append(dict(typ='Tag',gtin='7613033635205',txt='4.79',x=UL_x,y=UL_y,w=width,h=height))
                    i+=1
                # draw results for debugging
                im  = cvDrawBoxes(res,cv2.imread(filename))
                im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
                cv2.imwrite('result.png',im)
                
                rtn = (True,lis,e)
            else:
                rtn = (False,[],0)
            return rtn
        except:
            logging.info('Error in detect_image method\n')
            traceback.print_exc()

    
    def detect_video(self,filename,to_detect):
        try:
            # get network size
            # change line below to self.net2 if tags to be detected
            
            #net_w = dn.network_width(self.net)
            #net_h = dn.network_height(self.net)

            net_w = dn.network_width(self.net2)
            net_h = dn.network_height(self.net2)

            # read video and set size 
            cap = cv2.VideoCapture(filename)
            cap.set(3,1280)
            cap.set(4,720)

            # get frames count from file
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
            total_frames = int(cap.get(prop))
            # log info
            logging.info('File uploaded : {} , total number of frames in video {}'.format(filename,total_frames))

            #create a darknet image to be used for each extracted frame
            darknet_image = dn.make_image(net_w,net_h,3)

            # create a new video to write images and save them
            out = cv2.VideoWriter(os.path.join(os.getcwd(),'output.avi'),cv2.VideoWriter_fourcc(*"MJPG"),30.0,(net_w,net_h))
            
            # list of results for frames
            lis_outer = []
            #call forward when start grabing the frames
            grab, frame = cap.read()
            s = time.time()
            counter_out = 1

            while grab:
                frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb,(net_w,net_h),interpolation=cv2.INTER_LINEAR)
                dn.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

                # decide which model to use for detection based on input

                # time per frame uncomment below
                # inn_t = time.time()

                #if to_detect == 'product':
                #    res = dn.detect_image(self.net,self.meta,darknet_image) # assign a threshold
                #else:
                #    return (False,[],0)

                if to_detect == 'tag':
                    res = dn.detect_image(self.net2,self.meta2,darknet_image) # assign a threshold
                else:
                    return (False,[],0)

                # time per frame uncomment below
                #inn_t = time.time() - inn_t

                # save results to a dict and add it to a list
                lis_inner = []
                inner = 0

                while(inner < len(res)):
                    class_type = res[inner][0].decode('utf-8')
            
                    center_x = int(res[inner][2][0])
                    center_y = int(res[inner][2][1])
                    width = int(res[inner][2][2])
                    height = int(res[inner][2][3])

                    UL_x = int(center_x - width/2) #Upper Left corner X coord
                    if UL_x < 0:
                        UL_x = 0
                    UL_y = int(center_y + height/2) #Upper left Y
                    LR_x = int(center_x + width/2)
                    LR_y = int(center_y - height/2)
                    if LR_y < 0:
                        LR_y = 0
                    #old return
                    #lis_inner.append(dict(cls=class_type,x=UL_x,y=UL_y,w=width,h=height))
                    #new return (dummy data gtin and price)
                    #the new return contains dummy data for the sake of project completion, this section will be updated when the new model is uploaded
                    lis.append(dict(typ='Tag',gtin='7613033635205',txt='4.79',x=UL_x,y=UL_y,w=width,h=height))
                    inner+=1 

                lis_outer.append(dict(frame=counter_out,objects=lis_inner))
                if inner > 0:
                    counter_out+=1
                
                # for debuging draw the detections around the frame and save the file to check it later
                im  = cvDrawBoxes(res,frame_resized)
                im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)

                out.write(im)
                
                grab, frame = cap.read()
    
            t = time.time() - s
            cap.release()
            out.release()

            if counter_out > 0:
                rtn = (True,lis_outer,t)
            else:
                rtn = (False,[],0)
            return rtn
        
        except:
            logging.info('Error in detect video method')
            traceback.print_exc()

    
if __name__ == '__main__':
    try:
        # init logger
        logging.getLogger().setLevel(logging.INFO)
        # create uploads directory 
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        # create a new instance of the Detector class
        app.det = Detector()
        #start the application port 5000
        app.run(debug=True,host='0.0.0.0',port=5000)
    except:
        logging.info('Error in main')
        traceback.print_exc()
