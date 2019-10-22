import flask
import os
import logging
import time
import datetime
import werkzeug
import sys
import cv2
import imutils
import csv
import pickle
import statistics
import traceback

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
def cvDrawBoxes(detection):
    x, y, w, h = detection[2][0],\
                 detection[2][1],\
                 detection[2][2],\
                 detection[2][3]
    xmin, ymin, xmax, ymax = convertBack(
        float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)

    return (pt1,pt2)

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
    except Exception as e:
        logging.error(str(e))
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
            
    except Exception as e:
        #log error and print, we can check the error from docker logs outside the image
        logging.error(str(e))
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
    except Exception as e:
        #log error and print, we can check the error from docker logs outside the image
        logging.error(str(e))
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
        ## old model path
        #data_path = 'darknet/data/globus_13.data'
        #cfg_path = 'darknet/cfg/globus13-yolov3.cfg'
        #weights_path = 'darknet/backup/globus13-yolov3_2000.weights'

        data_path = 'darknet/data/globus_seed_wendel.data'
        cfg_path = 'darknet/cfg/globus_seed_wendel.cfg'
        weights_path = 'darknet/backup/globus_seed_wendel.weights'

        #read reference images for orb comparison
        with open('orb_reference_calculated3.pickle','rb') as handle:
            self.class_orb_ref = pickle.load(handle)

        #set orb variables
        self.orb = cv2.ORB_create(nfeatures=500)
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,table_number = 6, key_size = 12,multi_probe_level = 1)
        search_params = dict() #checks=50  # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

        #read gtins and save them to a data structure
        reader =csv.DictReader(open('test_shelf_list.csv','r'))
        dict_l = []
        self.dict_g = {}
        self.dict_count = {}
        
        for line in reader:
            dict_l.append(line) 

        for coi in dict_l:
            if coi['Product'] not in self.dict_g.keys():
                self.dict_g[coi['Product']] = {coi['ORB']:coi['Barcode']}
            else:
                self.dict_g[coi['Product']].update({coi['ORB']:coi['Barcode']})

            try:
                self.dict_count[coi['Product']]+=1
            except:
                self.dict_count[coi['Product']]=1
                
        #load network and meta using darkflow library
        self.net = dn.load_net(os.path.join(base_path,cfg_path).encode('utf-8'),os.path.join(base_path,weights_path).encode('utf-8'),0)
        self.meta = dn.load_meta(os.path.join(base_path,data_path).encode('utf-8'))
        
    def get_state(self):
        #check if the model is loaded then the state is ready
        net1 = False
        net2 = False
        
        if self.net:
            net1= True

        return dict(net1_state=net1,net2_state=net2)

    def calc_orb(self,img,ccls):
        if self.dict_count[ccls] == 1:
            return 0
        try:
            #cv2.imwrite('{}.jpg'.format(randint(0,10000)),img)
            _,desorb = self.orb.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None)
            m_counts = [[] for x in range(self.dict_count[ccls])]
            for k in self.class_orb_ref[ccls].keys():
                h_ref,w_ref = self.class_orb_ref[ccls][k].shape
                h,w,_ = img.shape
                img_ref = cv2.resize(self.class_orb_ref[ccls][k],None,fx=h/h_ref,fy=w/w_ref,interpolation=cv2.INTER_AREA)
                _,desorb2 = self.orb.detectAndCompute(img_ref,None)
                    
                matches = self.flann.knnMatch(desorb,desorb2,k=2)
                    
                #count the matches that fulfill a certain threshold
                if len(matches) < 1:
                    continue
                
                m_counter = 0
                total_counter = 0
                #can be done in parallel
                for match in matches:
                    if len(match) == 2:
                        # 0.65 of distance is best so far
                        if match[0].distance < 0.65 * match[1].distance:
                            m_counter+=1
                        total_counter+=1
                #print("counters values ...................",m_counter,total_counter)            
                m_counts[max(int(k.split('-')[0][-1])-1,0)].append(m_counter / total_counter)
                    
                # get the mean value for every sub and assign the larget val
            mxx = 0
            orb_res = 0
            #print(m_counts)
            for i,m_count in enumerate(m_counts):
                mn = statistics.mean(m_count)
                if  mn > mxx:
                    mxx = mn
                    orb_res = i+1
            
            return orb_res
        except Exception  as e :
            logging.error("calculate orb for class {}".format(ccls))
            logging.error(str(e))

    # detect image for groceries or tags
    def detect_image(self, filename,to_detect):
        try:
            s= time.time()
            img = cv2.imread(filename)
            print(img.shape)
            if to_detect == 'product':
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
                    
                    orb_class = self.calc_orb(img[y_mn:y_mn+h,x_mn:x_mn+w],class_type)
        
                    # debug
                    #cv2.rectangle(img, pt1, pt2, (0, 255, 0), 3)
                    #cv2.putText(img,class_type +" [" + str(round(res[i][1] * 100, 2)) + "]",(pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 255, 0], 2)
                    try:
                        lis.append(dict(typ='Product',gtin=self.dict_g[class_type][str(orb_class)],txt='',x=x_mn,y=y_mn,w=w,h=h))
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

    def detect_video(self,filename,to_detect):
        try:        
            if not to_detect == 'product':
                return (False,[],0)
           
            cap = cv2.VideoCapture(filename)
           
            # if the video is rotated, we need to read that flag
            #cap.set(3,1280)
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
                       
                    orb_class = self.calc_orb(frame[y_mn:y_mn+h,x_mn:x_mn+w],class_type)
                    
                    #debug
                    #cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 3)
                    #cv2.putText(frame,class_type +" [" + str(round(res[inner][1] * 100, 2)) + "]",(pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 255, 0], 2)
                   
                    lis_inner.append(dict(typ='Product',gtin=self.dict_g[class_type][str(orb_class)],txt='',x=x_mn,y=y_mn,w=w,h=h))
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
    except Exception as e:
        logging.error(str(e))
