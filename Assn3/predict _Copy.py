from scipy import ndimage
import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import skimage
from skimage.feature import hog
import pickle
# create an instance of each transformer


# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.

def decaptcha( filenames ):
	# The use of a model file is just for sake of illustration


	labels_list = []
	X = np.zeros(shape=(len(filenames)*3, 150, 150))
	k = 0
	for i in filenames:
		img = cv2.imread(i)
		ind = GetIndices(img)
		
		for j in range(len(ind)):
            
			if(ind[j][0] < 0):
				ind[j][0] = 0
			cropped = img[5:145, ind[j][0]:ind[j][1]]
			init_process = initial_preprocess(cropped)
			translation_process = translation(init_process)
			rotational_process, _ = rotation(translation_process)
			X[k] = 255-rotational_process
			k += 1
	hogify = HogTransformer(pixels_per_cell=(27, 27), cells_per_block=(2,2), orientations=9, block_norm='L2-Hys')
	scalify = StandardScaler()#  (27, 27)
	X_train_hog = hogify.fit_transform(X)
	X_train_prepared = scalify.fit_transform(X_train_hog)
	
	with open(f'svm_init.pkl', 'rb') as f:
		classifier = pickle.load(f)

	preds = classifier.predict(X_train_prepared)
	preds = np.array_split(preds, len(preds)/3)
	
	return [",".join(i) for i in preds]
    



def GetIndices(img):
    
    bg=img[0,0]
    
    set1=np.array([],int)
    for i in range(200):
        if np.sum(np.array(img[:,i]!=bg))>50:
            set1=np.append(set1,i)       
    pos1=int(np.mean(set1))


    set2=np.array([],int)
    for i in range(pos1+50,350):
        if np.sum(np.array(img[:,i]!=bg))>50:
            set2=np.append(set2,i)       
    pos2=int(np.mean(set2))



    set3=np.array([],int)
    for i in range(pos2+50,500):
        if np.sum(np.array(img[:,i]!=bg))>50:
            set3=np.append(set3,i)
    pos3=int(np.mean(set3))
    

    ## Below are the row and colum slicing indices for the three segments img[row:row, column:column]
#     segment1=[5:175,pos1-70:pos1+70]
#     segment2=[5:175,pos2-70:pos2+70]
#     segment3=[5:175,pos3-70:pos3+70]
    
    segment1=[pos1-70,pos1+70]
    segment2=[pos2-70,pos2+70]
    segment3=[pos3-70,pos3+70]
    
    return [segment1,segment2,segment3]

def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    cntr = (int(mean[0,0]), int(mean[0,1]))
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0])
    return angle



def rotation(ximg):
    #img = ximg.astype(np.uint8)
    #_, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = ximg.astype(np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    
    max_angle = 0
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 40 or 1e4 < area:
            continue

        angle = getOrientation(c, img)
        angle = math.ceil(angle*180/math.pi)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)


        center = (int(rect[0][0]),int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        #angle = int(rect[2])
        
        if(angle>45):
            angle = -90+angle
        if(angle<-45):
            angle = -angle
        if(abs(angle) > abs(max_angle)):
            if(abs(angle)<= 32):
                max_angle = angle
            
    
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), max_angle, 1)
    img = cv2.warpAffine(ximg, M, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
    return img, max_angle

def get_center_of_mass(img):
        Y,X = ndimage.center_of_mass(img)
        x,y = img.shape
        delta_x = np.round(y/2.0-X).astype(int)
        delta_y = np.round(x/2.0-Y).astype(int)
        return delta_x, delta_y
    
def get_to_center(image ,x, y):

        (rows , cols) = image.shape
        M = np.float32([[1,0,x],[0,1,y]])
        centered = cv2.warpAffine(image,M,(cols,rows))
        return centered 

def initial_preprocess(img):   # give raw images
    image = cv2.resize(img, (150,150))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (T, threshInv) = cv2.threshold(gray.copy(), 127, 255,
        cv2.THRESH_BINARY_INV )

    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    cimg = cv2.erode(threshInv, kernel_2, iterations= 1)
    return cimg

def translation(cimg):  # give thresholded images from 

        while np.sum(cimg[0]) == 0:  #making squared image with respective pixels
            cimg = cimg[1:]

        while np.sum(cimg[0,:]) == 0:
            cimg = cimg[:,1:]

        while np.sum(cimg[-1]) == 0:
            cimg = cimg[:-1]

        while np.sum(cimg[:, -1])==0:
            cimg = cimg[:,:-1]
            
        rows,cols = cimg.shape
      
        if rows == cols:
            nrows = 130
            ncols = 130
            cimg = cv2.resize(cimg, (ncols,nrows))
           

        if rows > cols:
            nrows = 130
            ncols = int(round((cols*130.0/rows), 0))
            cimg = cv2.resize(cimg, (ncols,nrows))
            
        else:
            ncols = 130
            nrows = int(round((rows*130.0/cols), 0))
            
            cimg = cv2.resize(cimg, (ncols,nrows))
            
                             
        
        col_pad = (int(math.ceil((150-ncols)/2.0)), int(math.floor((150-ncols)/2.0)))

        row_pad = (int(math.ceil((150-nrows)/2.0)), int(math.floor((150-nrows)/2.0)))
        cimg = np.lib.pad(cimg,(row_pad,col_pad),'constant')


        del_x, del_y = get_center_of_mass(cimg) 
        centered = get_to_center(cimg ,del_x, del_y)
         
        ximg = centered.reshape(1,150,150).astype(np.float32)
        ximg-= int(33.3952)
        ximg/= int(78.6662)
        return ximg[0]

class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])


filena= ['0.png', '1.png']
a=decaptcha(filena)
print(a)