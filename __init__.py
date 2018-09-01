import json
import logging
import os
import sys
from pprint import pprint
import cv2
import numpy as np
from PIL import Image
from sanskrit_data.schema import ullekhanam
import preprocessing
from skimage import measure
import pickle as pkl

logging.basicConfig(
  level=logging.DEBUG,
  format="%(levelname)s: %(asctime)s {%(filename)s:%(lineno)d}: %(message)s "
)

class DisjointRectangles:
  def __init__(self, segments=None):
    if segments is None:
      segments = []
    self.img_targets = []
    self.merge(segments)

  def overlap(self, testseg):
    #testrect = self.to_rect(testseg)
    for i in range(len(self.img_targets)):
      if self.img_targets[i] == testseg:
        return i
    return -1

  def insert(self, newseg):
    #print "Inserting " + str(newseg)
    i = self.overlap(newseg)
    if i >= 0:
      if self.img_targets[i].score >= newseg.score:
        #print "Skipping " + str(newseg) + " < " + str(self.segments[i])
        return False
      else:
        self.remove(i)
        #for r in self.segments:
        #    print "->  " + str(r)
    #print "--> at " + str(len(self.segments))
    self.img_targets.append(newseg)
    return True

  def merge(self, segments):
    merged = [r for r in segments if self.insert(r)]
    return merged

  def get(self, i):
    return self.img_targets[i] if 0 <= i < len(self.img_targets) else None

  def remove(self, i):
    #print "deleting " + str(i) + "(" + str(len(self.segments)) + "): " + str(self.segments[i])
    if 0 <= i < len(self.img_targets):
      del self.img_targets[i]


class DocImage:
  def __init__(self, imgfile = None, workingImgFile = None):
    self.fname = ""
    self.working_img_rgb = None
    self.working_img_gray = None
    self.img_rgb = None
    self.img_gray = None
    self.OutputFile = None
    self.w = 0
    self.h = 0
    self.ww = 0
    self.wh = 0
    self.img_bin = None

    if imgfile:
      #print "DocImage: loading ", imgfile
      self.update_image_file(imgfile)
    if workingImgFile:
      #print "DocImage: loading ", origImgFile
      self.update_working_file(workingImgFile)

  def update_image_file(self, imgfile):
    self.fname = imgfile
    self.img_rgb = cv2.imread(self.fname)
    self.init()

  def update_working_file(self, workingImgFile):
    self.fname = workingImgFile
    self.working_img_rgb = cv2.imread(self.fname)
    if (self.working_img_rgb is None) :

      temp_img = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR)
      pil_im = Image.fromarray(temp_img)
      self.working_img_rgb = DocImage.resize(pil_im, (1920, 1080), False)
      self.working_img_rgb = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

      cv2.imwrite(workingImgFile, self.working_img_rgb)
    self.OutputFile = self.working_img_rgb
    self.working_img_gray = cv2.cvtColor(self.working_img_rgb, cv2.COLOR_BGR2GRAY)
    self.ww, self.wh = self.working_img_gray.shape[::-1]
    #logging.info("W width = " + str(self.ww) + ", W ht = " + str(self.wh))

  def init(self):
    self.img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
    self.img_bin = preprocessing.binary_img(self.img_gray)
    self.w, self.h = self.img_gray.shape[::-1]
    #logging.info("width = " + str(self.w) + ", ht = " + str(self.h))

  def from_image(self, img_cv):
    self.img_rgb = img_cv
    self.init()

  @classmethod
  def from_path(cls, path):
    from os.path import join
    [base_path, ext] = os.path.splitext(path)
    workingImgPath = join(base_path + "_working.jpg")
    #logging.info("Image path = " + path)
    #logging.info("Working Image path = " + workingImgPath)
    return DocImage(path, workingImgPath)

  @staticmethod
  def resize( img, box, fit):
    """Downsample the image.
    @param img: Image -  an Image-object
    @param box: tuple(x, y) - the bounding box of the result image
    @param fit: boolean - crop the image to fill the box

    @:returns: file-like-object - save the image into the output stream
    """
    #preresize image with factor 2, 4, 8 and fast algorithm
    factor = 1
    while img.size[0]/factor > 2*box[0] and img.size[1]*2/factor > 2*box[1]:
      factor *=2

    if factor > 1:
      img.thumbnail((img.size[0]/factor, img.size[1]/factor), Image.NEAREST)

    #calculate the cropping box and get the cropped part
    if fit:
      x1 = y1 = 0
      x2, y2 = img.size
      wRatio = 1.0 * x2/box[0]
      hRatio = 1.0 * y2/box[1]
      if hRatio > wRatio:
        y1 = int(y2/2-box[1]*wRatio/2)
        y2 = int(y2/2+box[1]*wRatio/2)
      else:
        x1 = int(x2/2-box[0]*hRatio/2)
        x2 = int(x2/2+box[0]*hRatio/2)
      img = img.crop((x1,y1,x2,y2))

    #Resize the image with best quality algorithm ANTI-ALIAS
    img.thumbnail(box, Image.ANTIALIAS)
    return img

    #save it into a file-like object
  #    img.save(out, "JPEG", quality=100)
  #resize

  def save(self, dstfile):
    cv2.imwrite(dstfile, self.img_rgb)

  def find_matches(self, template_img, thres = 0.7, known_segments = None):
    res = cv2.matchTemplate(self.img_bin,  template_img.img_bin, cv2.TM_CCOEFF_NORMED )
    loc = np.where(res >= float(thres))
    def ptToImgTarget(pt):
      return ullekhanam.Rectangle.from_details(x=pt[0], y= pt[1],
                                               w=template_img.w, h=template_img.h,
                                               score = float("{0:.2f}".format(res[pt[1], pt[0]]))
                                               )
    matches = map(ptToImgTarget, zip(*loc[::-1]))

    if known_segments is None:
      known_segments = DisjointRectangles()
    disjoint_matches = known_segments.merge(matches)
    known_segments.img_targets.sort()
    #for r in known_segments.segments:
    #   logging.info(str(r))
    return disjoint_matches

  def snippet(self, r):
    #template_img = self.img_rgb[r.y:(r.y+r.h), r.x:(r.x+r.w)]
    template_img = self.img_rgb[r["y"]:(r["y"]+r["h"]), r["x"]:(r["x"]+r["w"])]
    template = DocImage()
    template.from_image(template_img)
    return template

  def find_recurrence(self, r, thres = 0.7, known_segments = None):
    #logging.info("Searching for recurrence of " + json.dumps(r))

    template = self.snippet(r)

    if known_segments is None:
      known_segments = DisjointRectangles()
    known_segments.insert(ullekhanam.Rectangle.from_details(x=r["x"], y=r["y"], w=r["w"], h=r["h"]))
    return self.find_matches(template, thres, known_segments)

  def self_to_image(self):
    return self.img_rgb


  def find_text_regions(self, show_int=0, pause_int=0):

    if self.working_img_gray is None:
      img = self.img_gray
      totalArea = self.w * self.h
    else:
      img = self.working_img_gray
      totalArea = self.ww * self.wh

    kernel1 = np.ones((2,2),np.uint8)

    def show_img(name, fname):
      if int(show_int) != 0:
        screen_res = 1280.0, 720.0
        scale_width = screen_res[0] / fname.shape[1]
        scale_height = screen_res[1] / fname.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(fname.shape[1] * scale)
        window_height = int(fname.shape[0] * scale)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, window_width, window_height)
        cv2.imshow(name, fname)

      if int(pause_int) != 0:
        cv2.waitKey(0)

    def non_max_suppression_fast(boxes, overlapThresh):
      boxes = np.array(boxes)
      if len(boxes) == 0:
        return []
      # if the bounding boxes integers, convert them to floats --
      # this is important since we'll be doing a bunch of divisions
      if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
      pick = []
      print("Boxes",boxes)
      x1 = boxes[:,0]
      y1 = boxes[:,1]
      x2 = boxes[:,2]
      y2 = boxes[:,3]

      
      area = (x2 - x1 + 1) * (y2 - y1 + 1)
      idxs = np.argsort(y2)
      
      while len(idxs) > 0:
      
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
      
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
     
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
     
        overlap = (w * h) / area[idxs[:last]]
     
        idxs = np.delete(idxs, np.concatenate(([last],
          np.where(overlap > overlapThresh)[0])))
      
      return boxes[pick].astype("int")


    images1=[]
    show_img('Output0',img)
    images1.append(img)
    

    #ret3,th1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #show_img('otsuThreshold after adaptive', th1)

    #ret,th2 = cv2.threshold(img,ret3-10,255,cv2.THRESH_BINARY)
    #show_img('binary', th2)


    th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
    show_img('adaptiveThreshold', th1)
    images1.append(th1)

    #th1 = cv2.dilate(th1, kernel1, iterations=1)
    #show_img('dilation', th1)
    

    blur = cv2.medianBlur(th1,3)
    show_img('median filtering',blur)
    images1.append(blur)

    #blur = cv2.GaussianBlur(blur,(5,5),0)
    #show_img('blur', blur)

    #opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel1)
    #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel1)

    #ret3,th1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #show_img('otsuThreshold after adaptive', th1)

    img = th1

    boxes_temp = np.zeros(img.shape[:2],np.uint8)
    #logging.info("boxes generated")

    binary = 255-img
    dilation=binary

    #erosion = cv2.erode(binary, kernel1, iterations=1)
    #show_img('Erosion', erosion)

    # dilation = cv2.dilate(dilation,kernel1,iterations = 1)
    # show_img('dilationation', dilation)

    #dilation = cv2.GaussianBlur(dilation,(1,3),0)
    #show_img('GaussianBlur',dilation)

    labels = measure.label(dilation, neighbors=8, background=0)
    mask = np.zeros(dilation.shape, dtype="uint8")

    for label in np.unique(labels):
      
      if label == 0:
        continue

      labelMask = np.zeros(dilation.shape, dtype="uint8")
      labelMask[labels == label] = 255
      numPixels = cv2.countNonZero(labelMask)
     
      if numPixels > 5:
        mask = cv2.add(mask, labelMask)

    show_img('mask', mask)
    images1.append(mask)
    cv2.imwrite('ttttttt.jpg',mask)

    if self.working_img_gray is None:
      factorX = float(1.0)
      factorY = float(1.0)
    else:
      factorX = float(self.w) / float(self.ww)
      factorY = float(self.h) / float(self.wh)
    #        logging.info("factorx:"+str(factorX)+"factory:"+str(factorY))

    # Bounds are a guess work, more can be on it.
    lower_bound = totalArea / 8000
    upper_bound = totalArea / 10
    
    ret,thresh = cv2.threshold(mask.copy(),127,255,0)
    immm,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    allsegments = []

    ret,thresh = cv2.threshold(mask.copy(),127,255,0)
    immm,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    print("Lower="+str(lower_bound)+" Upper="+str(upper_bound))

    print("Contours 3 Len = "+str(len(contours)))
    print("contours",contours)
    listx=[]
    listy=[]
    for c in contours:
      coordinates = {'x': 0, 'y':0, 'h':0, 'w':0, 'score':float(0.0)}
      x,y,w,h = cv2.boundingRect(c)
      #            logging.info("x:"+str(x)+"y:"+str(y)+"w:"+str(w)+"h"+str(h))
      if (((w*h) <= lower_bound or (w*h) >= upper_bound)) :
        continue
      # cv2.rectangle(boxes_temp,(x,y),(x+w,y+h),(255,0,0),1)
      listy=[x,y,x+w,y+h]
      listx.append(listy)
      listy=[]


    
    boxes_temp1= non_max_suppression_fast(listx,0.3)
    for c in boxes_temp1:
      cv2.rectangle(boxes_temp,(c[0],c[1]),(c[2],c[3]),(255,0,0),2)
    show_img('Boxes_temp 3',boxes_temp)
    images1.append(boxes_temp)    

    return boxes_temp1,images1



  def add_rectangles(self, sel_areas, color = (0, 0, 255), thickness = 2):
    for rect in sel_areas:
      cv2.rectangle(self.img_rgb, (rect[0], rect[1]),
                    (rect[2],rect[3]), color, thickness)


def main(args):
   img = DocImage(args[0])
   rect = { 'x' : int(args[1]),
            'y' : int(args[2]),
            'w' : int(args[3]), 'h' : int(args[4]), 'score' : float(1.0) }
   logging.info("Template rect = " + json.dumps(rect))
   matches = img.find_recurrence(rect, 0.7)
   pprint(matches)
   logging.info("Total", len(matches), "matches found.")

   #logging.info(json.dumps(matches))
   img.add_rectangles(matches)
   img.add_rectangles([rect], (0, 255, 0))

   cv2.namedWindow('Annotated image', cv2.WINDOW_NORMAL)
   cv2.imshow('Annotated image', img.img_rgb)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   sys.exit(0)

def mainTEST(arg):
  [bpath, filename] = os.path.split(arg)
  [fname, ext] = os.path.splitext(filename)

  image = Image.open(arg).convert('RGB')
  workingFilename = fname+"_working.jpg"
  out = open(workingFilename, "w")
  img = DocImage.resize(image, (1920,1080), False)
  img.save(out, "JPEG", quality=100)
  out.close()

  img = DocImage(arg,fname+"_working.jpg")
  segments, images1 = img.find_text_regions(1, 1)
  print("special:  ",len(segments))

  with open('segments.pkl','wb') as f:
    pkl.dump(segments,f)

  #for seg in segments:
  
  #  first_snippet = img.snippet(seg)
  #  cv2.imshow('First snippet', first_snippet.img_rgb)
  #  cv2.waitKey(0)
    #first_snippet.save(fname + "_snippet1.jpg")
  #first_snippet = img.snippet(segments[5])
  #cv2.imshow('First snippet', first_snippet.img_rgb)
  #cv2.waitKey(0)
  #first_snippet.save(fname + "_snippet1.jpg")
  
  anno_img = DocImage()
  anno_img.from_image(img.OutputFile)
  anno_img.add_rectangles(segments)
  #    img.annotate(img.find_sections(1,1))
  #img.annotate(img.find_segments(1,1))

  screen_res = 1280.0, 720.0
  scale_width = screen_res[0] / anno_img.w
  scale_height = screen_res[1] / anno_img.h
  scale = min(scale_width, scale_height)
  window_width = int(anno_img.w * scale)
  window_height = int(anno_img.h * scale)

  cv2.namedWindow('Final image', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('Final image', window_width, window_height)

  cv2.imshow('Final image', anno_img.img_rgb)
  images1.append(anno_img.img_rgb)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


  height = sum(image.shape[0] for image in images1)
  width = max(image.shape[1] for image in images1)
  output = np.zeros((height,width,3))
  print(output.shape)

  y = 0
  for image in images1:
    if(len(image.shape)==2):
      image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
      h,w,c = image.shape
      output[y:y+h,0:w,0:c] = image
    else:
      h,w,c = image.shape
      output[y:y+h,0:w,0:c] = image

    y += h
  nmm="test_"+fname+".jpg"
  cv2.imwrite(nmm, output)


if __name__ == "__main__":
  #main(sys.argv[1:])
  mainTEST(sys.argv[1])
