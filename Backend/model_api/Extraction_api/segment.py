import cv2
import numpy as np
from PIL import Image, ImageOps

def get_exact_bbox(img, bbox_list):
    PIL_img = Image.fromarray(img)
    new_bbox_list=[]
    segmented_list = []
    j=0
    
    for i in bbox_list:
        segmented_sample = {
            "img_name": None,
            "pil": None
        }
        
        xmin, ymin, xmax, ymax = i[0], i[1], i[2], i[3] # coordinate of yolo bbox is [xmin, ymin, xmax, ymax]
        #crop box format: xmin, ymin, xmax, ymax
        crop_box = (xmin, ymin, xmax, ymax)
        im = PIL_img.crop(crop_box)
    
        # remove alpha channel
        invert_im = im.convert("RGB")

        # invert image (so that white is 0)
        invert_im = ImageOps.invert(invert_im)
        imageBox = invert_im.getbbox()
        imageBox = tuple(np.asarray(imageBox))
        
        cropped=im.crop(imageBox)
        
        # remove alpha channel
        cropped = cropped.convert("L")
        
        img_with_border = ImageOps.expand(cropped, border = 4, fill = 255) #add white border for crop image
        
        new_bbox_list.append((imageBox[0], i[1] + imageBox[1], imageBox[2],  i[1] + imageBox[3]))
        
        segmented_sample['img_name'] = str(j)
        segmented_sample['pil'] = np.array(img_with_border).tolist()
        segmented_list.append(segmented_sample)
        
        j = j + 1
    return new_bbox_list, segmented_list

def segment_line(pil_img):
    ## (1) read
    # img = cv2.imread(img_path + file_name)

    # img = np.array(pil_img) # just added
    img = np.array(pil_img, dtype="uint8")
    
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = np.array(pil_img) # just added
    gray = np.array(pil_img, dtype="uint8")

    ## (2) threshold
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    
    dilate_kernel = np.ones((10, 10), np.uint8)
    dilate_img = cv2.dilate(threshed, dilate_kernel, iterations=1)
    erode_kernel = np.ones((10, 10), np.uint8)
    erode_img = cv2.erode(dilate_img, erode_kernel, iterations=1)
    
    # erode_img = threshed

    ## (3) minAreaRect on the nozeros
    pts = cv2.findNonZero(erode_img)
    ret = cv2.minAreaRect(pts)

    (cx,cy), (w,h), ang = ret
    if w>h:
        w,h = h,w
        
    ## (4) Find rotated matrix, do rotation
    M = cv2.getRotationMatrix2D((cx,cy), 0, 1.0)
    rotated = cv2.warpAffine(erode_img, M, (img.shape[1], img.shape[0]))
    
    ## (5) find and draw the upper and lower boundary of each lines
    hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32S).reshape(-1)
    hist_sum = cv2.reduce(rotated, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).reshape(-1)
    H,W = img.shape[:2]

    ycoords = []
    y = 0
    count = 0
    count_sum = 0
    isSpace = False

    for i in range(len(rotated)):
        if isSpace == False:
            if hist[i] == 0:
                isSpace = True
                if hist_sum[i] == 0:
                    count_sum = 1
                    count = 0
                else:
                    count_sum = 0
                    count = 1
                y = i
        else:
            # if (hist[i] > 0) or (i == len(rotated) - 1):
            if (hist[i] > 0):
                isSpace = False
                if count > 10: # has just add
                    ycoords.append(int(y / count))
                    count = 0
                if count_sum > 4:
                    ycoords.append(int(y / count_sum))
                    count_sum = 0
                    
                    
            else:
                if hist_sum[i] == 0:
                    if count > 0:
                        count = 0
                        y = i
                    else:
                        y = y + i
                    count_sum = count_sum + 1
                    
                else:
                    if count_sum > 4:
                        ycoords.append(int(y / count_sum))
                        count_sum = 0
                        isSpace = False
                    else:
                        if count_sum > 0:
                            count_sum = 0
                            y = i
                        else:
                            y = y + i
                        count = count + 1
    
    # remove 1st line coordinate from ycoords
    if len(ycoords) > 0:
        ycoords.pop(0)

    rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)


    # list store bbox coordinates list:
    bbox_list = []
    
    if len(ycoords) > 0: # "multi-line" case
        for i in range(len(ycoords)):
            cv2.line(rotated, (0, ycoords[i]), (W, ycoords[i]), (0,255,0), 1)
            if i == 0:
                bbox_list.append((0, 0, W, ycoords[i]))
            else:
                bbox_list.append((0, ycoords[i - 1], W, ycoords[i]))
        # add final bbox to list
        if len(ycoords) > 0:
            bbox_list.append((0, ycoords[-1], W, H))
    else: # "single-line" or "multi-line but can not be segmented" case
        bbox_list.append((0, 0, W, H))
            
    new_bbox_list, segmented_list = get_exact_bbox(gray, bbox_list)
    
    print(new_bbox_list)
    return segmented_list

# segment_line(processed_img_dir, cropped_dir, '0301236-page14.jpg_14838.jpg')