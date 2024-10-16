import os
import cv2
import glob
import numpy as np
import tifffile
import numpy as np
from PIL import Image

from dFluor import plotSummary


def mp4Summary(imgPath, speed):
    imgName = os.path.basename(imgPath)
    os.chdir(imgPath)

    ### getfps from Log.txt
    log = open('./Log.txt', 'r')
    logRead =log.read()
    loglines = logRead.split('\n')
    fpsTxt =  [ s for s in loglines if "fps:" in s ]
    fpsTitle, fpsValue= fpsTxt[0].split(": ")
    fps = float(fpsValue)

    ### get npF1 for standardization
    #inputNp = [npF1RAW, npF1, npMip, npThr, npRAWmean]
    input = glob.glob("./resultArray/*_inputNp.npy")
    inputNp = np.load(input[0])
    npF1 = inputNp[1]

    ### read tiffStack, standardize by npF1
    moviePath = glob.glob("./resultMovie/*differential_RAW.tiff")

    ref, img=cv2.imreadmulti(moviePath[0],flags=-1)
    arrImg = np.asarray(img)
    arrImgSt = np.nan_to_num(arrImg/npF1) # standardize by F1，remove 0 by nan conversion 

    ### remove IQR*1.5 as outlier. minus values to 0
    arrImgSt = plotSummary.outlier(arrImgSt)

    ### arr to uint8
    arrImgInt = plotSummary.array_to_img(arrImgSt)

    # export peseudo colored RGB stack 
    color_stack = np.zeros((arrImgInt.shape[0], arrImgInt.shape[1], arrImgInt.shape[2], 3), dtype=np.uint8)
    for i in range(arrImgInt.shape[0]):
        color_stack[i] = cv2.applyColorMap(arrImgInt[i], cv2.COLORMAP_TURBO)

    # BGR order[openCV] to RGB order
    # colourstack shape = TYXC
    # ImageJ hyperstack axes must be in TZCYXS order (hyperstack must be in TCYX)
    colorStack_TYXC_RGB = np.empty_like(color_stack)
    colorStack_TYXC_RGB[:,:,:,0] = color_stack[:,:,:,2]
    colorStack_TYXC_RGB[:,:,:,1] = color_stack[:,:,:,1]
    colorStack_TYXC_RGB[:,:,:,2] = color_stack[:,:,:,0]
    tiffName = f"{imgName}_diff_pseudo.tif"
    tifffile.imsave(f"./{tiffName}", colorStack_TYXC_RGB, photometric="rgb") #, imagej=True, metadata={'axes': 'XYCT', 'mode': 'composite', 'fps': fps})

    #　export mp4
    mp4Name = f"{imgName}_{speed}xSpeed_diff.mp4"
    playbackFps = int(fps * speed)
    height = np.shape(arrImgInt)[1]
    width = np.shape(arrImgInt)[2]
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'./{mp4Name}', codec, playbackFps, (width, height), True)
        
    for i in range(len(arrImgInt)):
        img = cv2.applyColorMap(arrImgInt[i], cv2.COLORMAP_TURBO)
        video.write(img)     
    video.release()
    

def tif_to_mp4(moviePath, speed):
    imgName = os.path.basename(moviePath)
    ref, img=cv2.imreadmulti(moviePath,flags=-1)
    arrImg = np.asarray(img, dtype="float") # if do not specify float, max will be 1


    ### get fps from metadata of images
    with Image.open(moviePath) as im:
        title = im.tag.get(270)
    fpsList = title[0].split("\n")

    for item in fpsList:
        if item.startswith("fps="):
            fps = float(item.split("=")[1])
            break
    fps = float(round(fps, 2))

    # remove outlier
    arrImgOut = plotSummary.outlier(arrImg)

    # standardize to uint8
    arrImgInt = plotSummary.array_to_img(arrImgOut)

    playbackFPS = float(fps * speed)

    #　export mp4
    savename = imgName.replace("_prePro.tif", "")
    mp4Savepath = f"./{savename}/{savename}_{speed}xSpeed_raw.mp4"

    height = np.shape(arrImgInt)[1]
    width = np.shape(arrImgInt)[2]
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(mp4Savepath, codec, playbackFPS, (width, height), True) 
        
    for i in range(len(arrImgInt)):
        img = cv2.applyColorMap(arrImgInt[i], cv2.COLORMAP_VIRIDIS)
        video.write(img)     
    video.release()
    
    
