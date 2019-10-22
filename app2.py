import flask
import os
import logging
import time
import datetime
import werkzeug
import sys
import traceback
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
ALLOWED_VIDEO_EXTENSIONS = set(['avi','mp4','mov'])
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
            
            # default error response
            ret = flask.jsonify(ret='empty')

            #process the return if True
            if rtn[0]:
                ret = flask.jsonify(ret='objects_found',objs=rtn[1],t=rtn[2])
            
            # add CORS header
            ret.headers['Access-Control-Allow-Origin'] = '*'

            return ret
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
        data_path2 = 'darknet/data/globus_tags.data'
        cfg_path2 = 'darknet/cfg/globustags-yolov3.cfg'
        weights_path2 = 'darknet/backup/globustags-yolov3.weights'

       #TODO: add barcode and digit extraction

        self.net2 = dn.load_net(os.path.join(base_path,cfg_path2).encode('utf-8'),os.path.join(base_path,weights_path2).encode('utf-8'),0)
        self.meta2 = dn.load_meta(os.path.join(base_path,data_path2).encode('utf-8'))

        
    def get_state(self):
        #check if the model is loaded then the state is ready
        net1 = False
        net2 = False
        
        if self.net2:
            net2 = True

        return dict(net1_state=net1,net2_state=net2)

#detect image
    def detect_image(self, filename,to_detect):
        try:
            s= time.time()
            img = cv2.imread(filename)
            print(img.shape)
            if to_detect == 'tag':
                res = dn.detect(self.net,self.meta,filename.encode('utf-8'))
                #res = dn.detect_image(self.net,self.meta,img) # set threshold
            else:
                return(False,[],0)
        
            e = time.time() - s
            if len(res) > 0:
                # process the output into a dictionary
                i=0
                lis = []
                while(i < len(res)):
                    class_type = res[i][0].decode('utf-8')
                    pt1,pt2  = cvDrawBoxes(res[i])
                    y_mn = max(pt1[1],0)
                    x_mn = max(pt1[0],0)

                    w = pt2[0] - x_mn
                    h = pt2[1] - y_mn
        
                    # debug
                    #cv2.rectangle(img, pt1, pt2, (0, 255, 0), 3)
                    #cv2.putText(img,class_type +" [" + str(round(res[i][1] * 100, 2)) + "]",(pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 255, 0], 2)
                    try:
                        lis.append(dict(typ='Tag',gtin="dummy data",txt='',x=x_mn,y=y_mn,w=w,h=h))
                    except Exception as e2:
                        logging.error(e2)
                    i+=1

                rtn = (True,lis,e)
            else:
                rtn = (False,[],0)
            return rtn
        except Exception as e:
            logging.info('Error in detect_image method\n')
            logging.error(str(e))


#detect_video
    def detect_video(self,filename,to_detect):
        try:        
            if not to_detect == 'tag':
                return (False,[],0)
           
            cap = cv2.VideoCapture(filename)
           
            # if the video is rotated, we need to read that flag
            #cap.set(3,1280) with opposite value
            #cap.set(4,720)
           
            # get frames count from file
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
            
            total_frames = int(cap.get(prop))
            # log info
            logging.info('File uploaded : {} , total number of frames in video {}'.format(filename,total_frames))
            
            lis_outer = []
            grab, frame = cap.read()
            
            f_h,f_w,f_c = frame.shape

            darknet_image = dn.make_image(f_w,f_h,f_c)
               
            #out = cv2.VideoWriter(os.path.join(os.getcwd(),'output-{}.avi'.format(filename)),cv2.VideoWriter_fourcc(*"MJPG"),30.0,(f_w,f_h))
            
            s = time.time()
            counter_out = 1
            
            while grab:
                dn.copy_image_from_bytes(darknet_image,frame.tobytes())
                res = dn.detect_image(self.net,self.meta,darknet_image,0.65)
                lis_inner = []
                inner = 0
                
                while(inner < len(res)):
                    class_type = res[inner][0].decode('utf-8')
                    pt1,pt2  = cvDrawBoxes(res[inner])
                    y_mn = max(pt1[1],0)
                    x_mn = max(pt1[0],0)
                    
                    w = pt2[0] - x_mn
                    h = pt2[1] - y_mn
                    
                    #debug
                    #cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 3)
                    #cv2.putText(frame,class_type +" [" + str(round(res[inner][1] * 100, 2)) + "]",(pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 255, 0], 2)
                   
                    lis_inner.append(dict(typ='Tag',gtin='dummy number',txt='',x=x_mn,y=y_mn,w=w,h=h))
                    inner+=1
                lis_outer.append(dict(frame=counter_out,objects=lis_inner))
            
                if inner > 0:
                    counter_out+=1
            
                #out.write(frame)
                grab, frame = cap.read()
    
            t = time.time() - s
            cap.release()
            #out.release()

            if counter_out > 0:
                rtn = (True,lis_outer,t)
            else:
                rtn = (False,[],0)
            return rtn
        
        except Exception as e:
            logging.info('Error in detect video method')
            logging.error(str(e))
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
