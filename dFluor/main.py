#!/usr/bin/env python
# coding: utf-8


### Version info
# v2.7 in be-ta is released as release1.0


### import
import sys
import os 
import cv2
import nd2
import glob
import datetime
import tifffile
import numpy as np
from PIL import Image
import napari
from napari.layers import Shapes
from magicgui import magicgui
import shutil
import tifffile
from pystackreg import StackReg
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
import xarray
import pandas as pd
from tqdm import tqdm
import warnings
import itertools
from concurrent.futures import ProcessPoolExecutor as ppe

warnings.filterwarnings("ignore", category=DeprecationWarning) 
import scipy.signal

from dFluor.manualThreshold import thresholding



def dif(pathDir, taskName = "task", fitFunction = "exGauss", nd2ScaleFunction = True, nd2Scale = 1.0, ROIsetting = False,
        stackReg = False, generateMovie = True, gaussFilter = True, smoothing = True, smoothingWindow = 1, BGframes = 30,
        manualFps = None, usePreviousAnalysis = False, usePreviousAnalysisDir = ""):
    '''
    initial setting 
    pathDir = "/mnt/c/users/SDH/Desktop/" # path for directory should contain "IN" folder with input nd2/tif files

    taskName       = "task"               # Appended to the output folder name
    fitFunction    = "exGauss"        # Choose functions: (only "exGauss")
    nd2ScaleFunction = True           # Whether to read nd2 files and scale them
    nd2Scale         = 1.0            # Resize ratio when converting nd2 to tif, better to scale the resolution to be less than 1024x1024 to avoid crashing
    ROIsetting = False                # Whether to get ROI with napari and XYTcrop
    stackReg = False                  # Whether to perform stackReg (using 1st frame for reference) at the end of preprocess

    generateMovie = True              # Whether to output a differentiated movie, set to False if unnecessary due to high memory load
    gaussFilter = True                # Whether to apply a Gaussian filter
    smoothing = True                  # Whether to apply smoothing using savgol_filter before fitting. It slows down the process
    smoothingWindow = 1               # How many seconds to set for savgol filter smoothing. default: 1sec
    BGframes = 30                     # How many frames to use as background? default: 30

    manualFps = None                  # Referenced when the file has no fps metadata
    usePreviousAnalysis = False       # If using previous settings, skip the function and use existing ones
    usePreviousAnalysisDir = ""       # If the above setting is True, enter the directory of interest here like "out_230627_17.14.17___test"
    '''
    # ImageFit用にglobalを宣言
    global npT
    global npThr
    global bgGreaterMip
    global npSt
    global npMip
    global npF1
    global bgStd


    ipynbFolderPath = os.getcwd()
    ipynbFolderName = os.path.basename(os.getcwd())

    if usePreviousAnalysis:
        nd2ScaleFunction = False
        ROIsetting = False      
        stackReg = False
    ### cd setting
    os.chdir(pathDir)
    cwdPath = os.getcwd()
    starttime = datetime.datetime.now()
    outdir = './OUT/out_' + starttime.strftime('%y%m%d_%H.%M.%S') + '___' + taskName
    os.makedirs(outdir)
    os.chdir(outdir)
    os.makedirs(f"./IN/")




    ### Process nd2File
    # Create a tiff file scaled to the specified scale
    # Skipped when the number of nd2 files is 0
    # Skipped when there is a tiff file with the same name as nd2

    nd2List = glob.glob('../../IN/*.nd2')
    imgList = glob.glob('../../IN/*.tif')

    if nd2ScaleFunction:
        if len(nd2List) >= 1:
            print("converting nd2 files...")
            for count, nd2Path in enumerate(nd2List, 1):
                print(f"converting: {count}/{len(nd2List)}")

                nd2Name = os.path.splitext(os.path.basename(nd2Path))[0]
                nd2NameSearch = [i  for i in imgList if nd2Name in i]
                if len(nd2NameSearch)>0: # Function is skipped when there is a tiff file with the same name as nd2
                    continue
                else:
                    pass

                f = nd2.ND2File(nd2Path)
                frames = f.metadata.contents.frameCount
                params = f.experiment[0].parameters
                periodAvg =params.periodDiff.avg
                fps = frames/((frames*periodAvg)/1000)

                nd2Array = nd2.imread(nd2Path) #import nd2 as array
                if len(nd2Array.shape) == 4: # Exception handling for cases with many channels
                    nd2Array = nd2Array[:, 0, :, :] #greenChannel
                    # redChannel = nd2Array[:, 1, :, :]
                    # grayChannel = nd2Array[:, 2, :, :]
                elif len(nd2Array.shape) == 3:  
                    pass
                else:
                    print("nd2 shape is strange. Program stopped.")

                nd2Array = [nd2Array[i] for i in range(nd2Array.shape[0])]
                nd2DownSize = [cv2.resize(nd2Array[i], dsize=None, fx=nd2Scale, fy=nd2Scale) for i in range(len(nd2Array))]  # scaling
                nd2DownSizeArray = np.asarray(nd2DownSize)
                nd2Name, ext = os.path.splitext(os.path.basename(nd2Path))  
                tifSavePath = f"../../IN/{nd2Name}_scaled.tif"
                if os.path.exists(tifSavePath):
                    os.remove(tifSavePath)
                else:
                    pass 
                tifffile.imwrite(tifSavePath, nd2DownSizeArray, imagej=True ,photometric='minisblack', metadata={'axes': 'TYX', 'fps': fps}) # The fps here does not reference the manual threshold setting


                # memory release
                f.close()
                del nd2Array
                del nd2DownSize
                del nd2DownSizeArray

            print("converted")

        else:
            pass
        
        
        
        
        
    ### ROI setting and Threshold acquisition
    # The following warning appears:
    # DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.  elif LooseVersion(this_version) < '2.1':
    # When ROIsetting is True, ROI is acquired using napari
    # Time crop is not applied when the values in napari set to 0,0.
    # manualThreshold is always executed.

    if usePreviousAnalysis:
        fromPrepro = f"../{usePreviousAnalysisDir}/IN"
        toPrepro = './IN'
        shutil.rmtree(toPrepro)
        shutil.copytree(fromPrepro, toPrepro)

        fromROI = f"../{usePreviousAnalysisDir}/ROI.csv"
        toROI = "./ROI.csv"
        shutil.copy(fromROI, toROI)

        fromThr = f"../{usePreviousAnalysisDir}/threshold.npy"
        toThr = "./threshold.npy"
        shutil.copy(fromThr, toThr)

        manualThrArr = np.load("./threshold.npy")

    else:
        imgList = glob.glob('../../IN/*.tif')
        manualThrList = []
        ROIdf = pd.DataFrame(columns=['imgName', 'From', 'To', 'minXCoord', 'maxXCoord', 'minYCoord', 'maxYCoord'])

        for count, imgPath in enumerate(imgList, 1):
            imgStartTime = datetime.datetime.now()
            imgName, ext = os.path.splitext(os.path.basename(imgPath))     

            print(f"ROI and Thresholding: {count}/{len(imgList)}")

            ref, img=cv2.imreadmulti(imgPath,flags=-1)
            arrImg = np.asarray(img)
            image_shape = arrImg.shape # arrayshape: TYX

            # get fps
            with Image.open(imgPath) as im:
                title = im.tag.get(270)
            fpsList = title[0].split("\n")

            for item in fpsList:
                if item.startswith("fps="):
                    fps = float(item.split("=")[1])
                    break
            if "fps" in locals():
                pass
            else:
                fps = float(manualFps) # if there is no fps in metadata, use manual setting
            fps = float(round(fps, 2))

            ### ROI setting using napari
            # Open the video in napari
            # Draw one ROI you want to acquire
            # Set the time range to acquire
            # Press Run and close the window to obtain the ROI and time range.

            From = 0
            To = image_shape[0]-1
            layerLen = 0

            if ROIsetting:
                print("Sending to napari...")

                @magicgui(auto_call=True)
                def cropFrame(from_value=From, to_value=To):
                    global From, To
                    From = from_value
                    To = to_value
                    return From, To

                cropFrame.from_value.max = 99999   # release limit of the widget　999 -> 99999
                cropFrame.to_value.max   = 99999   # release limit of the widget　999 -> 99999

                viewer = napari.view_image(arrImg, ndisplay=2, colormap='turbo', title=imgName)
                viewer.theme = 'light'

                roi_layer = Shapes(
                    shape_type='rectangle',
                    edge_width=2,
                    face_color="red",
                    edge_color="red",
                    opacity   =0.3
                )
                
                viewer.window.add_dock_widget(cropFrame, name="Crop frame function", area="left", allowed_areas=["left", "bottom"])
                viewer.add_layer(roi_layer)

                napari.window.Window.set_geometry = (0, 0, 1920, 1080) # specify window location on display
                napari.run()      
                
                layer = viewer.layers["Shapes"]       
                layerLen = len(layer.data)
                
                if layerLen == 0:
                    minXCoord = 0
                    minYCoord = 0
                    maxXCoord = image_shape[2] # X
                    maxYCoord = image_shape[1] # Y
                    
                elif layerLen ==1:
                    ROIpoints = layer.data[0]
                
                    ## get coordinates from ROIpoints
                    xCoords = ROIpoints[:,1]
                    minXCoord = int(min(xCoords))
                    maxXCoord = int(max(xCoords))

                    yCoords = ROIpoints[:,0]
                    minYCoord = int(min(yCoords))
                    maxYCoord = int(max(yCoords))
                    
                    maxY = image_shape[1]
                    maxX = image_shape[2]
                    if maxYCoord > maxY:
                        maxYCoord = maxY
                    else:
                        pass
                    if maxXCoord > maxX:
                        maxXCoord = maxX
                    else:
                        pass
                    if minYCoord < 0:
                        minYCoord = 0
                    else:
                        pass
                    if minXCoord < 0:
                        minXCoord = 0
                    else:
                        pass
            else:
                minXCoord = 0
                minYCoord = 0
                maxXCoord = image_shape[2] #X
                maxYCoord = image_shape[1] #Y
                From = 0
                To = image_shape[0]

            columns=['imgName','From', 'To', 'minXCoord', 'maxXCoord', 'minYCoord', 'maxYCoord']
            ROIlist = [[imgName, From, To, minXCoord, maxXCoord, minYCoord, maxYCoord]]
            df_append = pd.DataFrame(data=ROIlist, columns=columns)
            ROIdf = pd.concat([ROIdf, df_append], ignore_index=True, axis=0)

            ### getThreshold
            matplotlib.use('Qt5Agg')
            manualThr = thresholding(imgPath,img)
            manualThr = list(manualThr)
            manualThrList.append(manualThr)

        ROIdf.to_csv("./ROI.csv", index=False)
        manualThrArr = np.array(manualThrList)
        np.save("./threshold.npy", manualThrArr)
        
        
        
        
        
    ### Preprocess(Crop,Stackreg)
    # In preprocessing, Crop and stackReg are executed in tyx stack. This is saved as _prepro.tif.

    imgList = glob.glob('../../IN/*.tif')

    if usePreviousAnalysis:
        pass

    else:
        for count, imgPath in enumerate(imgList, 1):
            imgName, ext = os.path.splitext(os.path.basename(imgPath))     

            ref, img=cv2.imreadmulti(imgPath,flags=-1)
            arrImg = np.asarray(img)

            print(f"Preprocessing: {count}/{len(imgList)}")

            ### get ROI information from ROIdf
            imgROI = ROIdf[ROIdf['imgName'] == imgName]
            From = int(imgROI.loc[:, "From"])
            To = int(imgROI.loc[:, "To"])
            minXCoord = int(imgROI.loc[:, "minXCoord"])
            maxXCoord = int(imgROI.loc[:, "maxXCoord"])
            minYCoord = int(imgROI.loc[:, "minYCoord"])
            maxYCoord = int(imgROI.loc[:, "maxYCoord"])

            # Time selection: If 3-8 is selected, frames will be 4-9.
            tCrop = arrImg[From:To+1] # Without +1, it would only capture up to the frame before the end.
            tyxCrop = tCrop[:, minYCoord:maxYCoord+1, minXCoord:maxXCoord+1]

            if stackReg:
                sr = StackReg(StackReg.RIGID_BODY)
                tyxCropReg = sr.register_transform_stack(tyxCrop, reference='first') # previous, first, mean are available for reference setting
                cropSave= np.array(tyxCropReg, dtype='uint16')      
                
            else:
                cropSave = tyxCrop
                pass

            ### save preprocessed file
            cropSavePath = f"./IN/{imgName}_prePro.tif"
            tifffile.imwrite(cropSavePath, cropSave, imagej=True ,photometric='minisblack', metadata={'axes': 'TYX', 'fps': fps})
            
            
            
            
            
    ### Fitting
    # Apply a Gaussian filter with cv2.GaussianBlur(imgSlice, (3, 3), sigmaX=1, sigmaY=1).
    # Standardize so that the max becomes 1 using xSt = (xStack-xF1)/(xMip-xF1).
    # Determine the threshold using manually obtained values with 2 dilations and 2 erosions.
    # Calculate the standard deviation for bg below the threshold. Calculate the 1st frame + Std*sdMagni (default is 2), and perform fitting for pixels above this value.

    imgList = glob.glob('./IN/*_prePro.tif')

    for count, imgPath in enumerate(imgList, 1):

        imgStartTime = datetime.datetime.now()
        imgName, ext = os.path.splitext(os.path.basename(imgPath))  
        imgName = imgName.replace('_prePro', '')

        thrArrWhere    = np.where(manualThrArr == imgName)
        thresholdValue = float(manualThrArr[thrArrWhere[0],1])  # Search the string array to extract the ratio
        sdMagni        = float(manualThrArr[thrArrWhere[0],2])  # Search the string array to extract the SD

        print(f"Fitting: {count}/{len(imgList)}")
        print(f"{imgName}")

        ref, img=cv2.imreadmulti(imgPath,flags=-1) # If -1 is not specified, the gradation will be lost (flags 0:Gray, 1:RGB, -1:RGBA)

        ### get fps from metadata
        with Image.open(imgPath) as im:
            title = im.tag.get(270)
        fpsList = title[0].split("\n")

        for item in fpsList:
            if item.startswith("fps="):
                fps = float(item.split("=")[1])
                break
        fps = float(round(fps, 2))

        print("processing...")

        if os.path.exists(f"./{imgName}") and os.path.isdir(f"./{imgName}"):
            shutil.rmtree(f"./{imgName}")
        else:
            pass

        os.makedirs(f"./{imgName}")

        npRAWmean = np.mean(img, axis=0)
        npF1RAW= img[0]

        blurStack = []
        for imgSlice in img:
            imgSlice = imgSlice.astype("float64")
            if gaussFilter == True:
                blur = cv2.GaussianBlur(imgSlice, (3, 3), sigmaX=1, sigmaY=1) # Gaussian filter
            else:
                blur = imgSlice
            blurStack.append(blur)

        npStack = np.asarray(blurStack) #axis: T, Y(ROW), X(COL)
        npF1 = np.mean(npStack[:BGframes], axis=0)
        npMip = np.amax(blurStack, axis=0)

        imgDim = npStack.shape
        frame  = imgDim[0]
        maxRow = imgDim[1]
        maxCol = imgDim[2]

        ### npSt
        xStack = xarray.DataArray(npStack, dims = ["t", "Row", "Col"])
        xF1 = xarray.DataArray(npF1, dims = ["Row", "Col"])
        xMip = xarray.DataArray(npMip, dims = ["Row", "Col"])

        xSt = (xStack-xF1)/(xMip-xF1) # standardize
        npSt = xSt.values

        npT = np.array(range(frame), dtype="float64")

        npF1RawMed = cv2.medianBlur(npF1RAW, 5)
        ret, npBin = cv2.threshold(npF1RawMed,thresholdValue ,255,cv2.THRESH_BINARY) # Cut using the obtained threshold value

        ### Erode/Dilate
        kernel = np.ones((3,3),np.uint8)
        npThr = np.copy(npBin)
        npThr = cv2.dilate(npThr, kernel, iterations = 2) #Dilate
        npThr =  cv2.erode(npThr, kernel, iterations = 2) #Erode
        ### Image processing up to this point
        
        npThrBool = np.array(npThr, dtype=bool) # True for values greater than or equal to BG
        npThrBg = np.logical_not(npThrBool) # Invert boolean values

        
        npBg = npStack[:,npThrBg]  # getarray only background time courses. [timeseries,pixels]
        bgSummary = pd.DataFrame(pd.Series(npBg.ravel()).describe()).transpose() #Summarize
        bgStd = bgSummary.loc[0,"std"] # SD of BG   
        
        BgPlusStd = npF1 + bgStd*sdMagni # BG + SD
        bgGreaterMip = np.greater_equal(BgPlusStd,npMip) # Returns True if BgPlusStd is greater than MIP

        iterList = list(itertools.product(range(maxRow), range(maxCol)))    # 

        ### multiprocessor
        cpuNum = int(os.cpu_count() //1.5) # use 2/3 cores for multiprocessing

        def processor():
            warnings.filterwarnings("ignore")
            

            
            with tqdm(total=len(iterList)) as progress: # For displaying elapsed time
                with ppe(max_workers = cpuNum) as executor: 
                    futures = []  # result list
                    for i in iterList:
                        future = executor.submit(imageFit, i, generateMovie, smoothing, smoothingWindow, fitFunction, fps)
                        future.add_done_callback(lambda p: progress.update()) # For displaying elapsed time
                        futures.append(future)
                    result = [f.result() for f in futures]
            warnings.resetwarnings()
            return result

        resultList = processor()

        print("saving...")
        print("")


        ### resultArray
        # e.g., resultArray[npMaxT][row,col] = 0
        resultArrayFromList = [resultList[i][0:9] for i in range(len(resultList))]  # List[0]~[8]
        resultArrayT = np.asarray(resultArrayFromList)
        resultArray = resultArrayT.transpose(1,0,2) # change shape to  [factors, pixels, values] where factors represent maxT or other params.

        ### transform shape to array[factors][2d img]
        def arrayShape(resultValues, npInit):
            emp = np.zeros_like(npInit, dtype="float64")
            for i in range(resultValues.shape[0]):
                iArray = resultValues[i]
                row = int(iArray[0])
                col = int(iArray[1])
                value = iArray[2]
                emp[row,col] = value
            return emp

        HMS1 = arrayShape(resultArray[0], npMip)
        HMS2 = arrayShape(resultArray[1], npMip)
        Smax = arrayShape(resultArray[2], npMip)
        Smin = arrayShape(resultArray[3], npMip)
        HMT1 = arrayShape(resultArray[4], npMip)
        HMT2 = arrayShape(resultArray[5], npMip)
        HWTi = arrayShape(resultArray[6], npMip)
        MaxT = arrayShape(resultArray[7], npMip)
        ThrT = arrayShape(resultArray[8], npMip)

        resultNp = np.asarray([HMS1, HMS2, Smax, Smin, HMT1, HMT2, HWTi, MaxT, ThrT])
        
        # Adjust by fps to convert them to per-second values
        HMS1AdjFPS = HMS1*fps
        HMS2AdjFPS = HMS2*fps
        SmaxAdjFPS = Smax*fps
        SminAdjFPS = Smin*fps
        HMT1AdjFPS = HMT1/fps
        HMT2AdjFPS = HMT2/fps
        HWTiAdjFPS = HWTi/fps
        MaxTAdjFPS = MaxT/fps
        ThrTAdjFPS = ThrT/fps
        
        resultNpFps = np.asarray([HMS1AdjFPS, HMS2AdjFPS, SmaxAdjFPS, SminAdjFPS, HMT1AdjFPS, HMT2AdjFPS, HWTiAdjFPS, MaxTAdjFPS, ThrTAdjFPS]) # adjusted by fps
        resultNpFpsSt = np.nan_to_num(np.asarray([HMS1AdjFPS/npF1, HMS2AdjFPS/npF1, SmaxAdjFPS/npF1, SminAdjFPS/npF1, HMT1AdjFPS, HMT2AdjFPS, HWTiAdjFPS, MaxTAdjFPS, ThrTAdjFPS])) # adjusted by fps and standardize by F1 and remove NaN

        ### save popt
        resultPoptFromList = [resultList[i][9:10] for i in range(len(resultList))] # List[9]
        resultPoptLen = max([resultPoptFromList[i][0][2] for i in range(len(resultList))])

        def poptShape(resultValues, resultPoptLen, npInitSlice):
            emp = np.zeros_like(npInitSlice, dtype="float64")
            empCp = np.repeat(emp[None, :], resultPoptLen, axis=0)
            
            for i in range(len(resultValues)): # repeat in pixels
                iArray = resultValues[i] # array in pixel[i]
                row = int(iArray[0][0])
                col = int(iArray[0][1])
                values = iArray[0][3]
                if len(values) == 0:
                    values = [0] * resultPoptLen
                else:
                    pass

                for f in range(len(values)): # Repeat along the time axis. Compensate for the 1-frame shift when reverting fData from histogram
                    value = values[f]
                    factor = int(f)
                    empCp[factor,row,col] = value
            return empCp

        poptNp = poptShape(resultPoptFromList, resultPoptLen, npF1)

        ### save movie from resultList
        # shape of resultSlope: [2900][0][2]. This list is structured as [pixel coordinates], [none], [0=row, 1=col, 2=list of values]

        def arrayShapeMovie(resultValues, npInit):
            emp = np.zeros_like(npInit, dtype="float64")
            
            for i in range(len(resultValues)): # repeat in pixels
                iArray = resultValues[i] # array in pixel[i]
                row = int(iArray[0][0])
                col = int(iArray[0][1])
                values = iArray[0][2]

                for t in range(len(values)-1): # Repeat along the time axis. Compensate for the 1-frame shift when reverting fData from histogram
                    value = values[t]
                    time = int(t)
                    emp[time,row,col] = value

            return emp

        if generateMovie:
            resultSlope = [resultList[i][10:11] for i in range(len(resultList))] # List[10] (slope)
            slopeStack = arrayShapeMovie(resultSlope, npStack)
            os.makedirs(f"./{imgName}/resultMovie/")
            tifffile.imsave(f'./{imgName}/resultMovie/{imgName}_differential_RAW.tiff', slopeStack) # no adjust by fps, no adjust by standardize
        else:
            pass

        ### Save array
        # npF1RAW: first frame in original stack
        # npF1: gaussian filtered original stack
        # npMip: MaxIntensityProjection
        # npThr: Threshold mask
        # npRAWmean: mean image of RAWstack
        inputNp = [npF1RAW, npF1, npMip, npThr, npRAWmean]
        
        os.makedirs(f"./{imgName}/resultArray/")
        outNp = f"./{imgName}/resultArray/{imgName}_resultRAWNp.npy"
        outNp2 = f"./{imgName}/resultArray/{imgName}_inputNp.npy"
        outNp3 = f"./{imgName}/resultArray/{imgName}_poptNp.npy"
        outNp4 = f"./{imgName}/resultArray/{imgName}_resultStNp.npy"
        np.save(outNp, resultNpFps)
        np.save(outNp2, inputNp)
        np.save(outNp3, poptNp)
        np.save(outNp4, resultNpFpsSt)

        ### save image
        os.makedirs(f"./{imgName}/resultImage/")
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_SPD_HalfMax1_RAW.tiff'     ,  resultNpFps[0]) # The "RAW" in the filename means that it has not been standardized with npF1
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_SPD_HalfMax2_RAW.tiff'     ,  resultNpFps[1]) # The "RAW" in the filename means that it has not been standardized with npF1
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_SPD_max_RAW.tiff'          ,  resultNpFps[2]) # The "RAW" in the filename means that it has not been standardized with npF1
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_SPD_min_RAW.tiff'          ,  resultNpFps[3]) # The "RAW" in the filename means that it has not been standardized with npF1
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_Time_HalfMax1_RAW.tiff'    ,  resultNpFps[4]) # The "RAW" in the filename means that it has not been standardized with npF1
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_Time_HalfMax2_RAW.tiff'    ,  resultNpFps[5]) # The "RAW" in the filename means that it has not been standardized with npF1
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_Duration_RAW.tiff'         ,  resultNpFps[6]) # The "RAW" in the filename means that it has not been standardized with npF1
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_Time_Max_RAW.tiff'         ,  resultNpFps[7]) # The "RAW" in the filename means that it has not been standardized with npF1
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_Time_Threshold_RAW.tiff'   ,  resultNpFps[8]) # The "RAW" in the filename means that it has not been standardized with npF1
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_blur_Frame1.tiff'      ,  npF1          )
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_blur_MaxIP.tiff'       ,  npMip         )
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_binary_Threshold.tiff' ,  npThr         )
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_RAW_MeanIP.tiff'       ,  npRAWmean     )
        tifffile.imsave(f'./{imgName}/resultImage/{imgName}_RAW_Frame1.tiff'       ,  npF1RAW       )
        
        imgEndTime = datetime.datetime.now()

        ### Save log
        with open(f"./{imgName}/Log.txt", mode ="w") as f:
            print(f"Start: {imgStartTime}" ,file=f)
            print(f"End:   {imgEndTime}" ,file=f)
            print(f"FileName: {imgName}{ext}" ,file=f)
            print("", file=f)
            print(f"Fitting function: {fitFunction}" ,file=f)
            print(f"Threshold Value: {thresholdValue}" ,file=f)
            print(f"Amplification of SD add to BG: {sdMagni}" ,file=f)
            print(f"fps: {fps}" ,file=f)
            print(f"ND2 scaleing: {nd2ScaleFunction}" ,file=f)
            print(f"ND2 scale: {nd2Scale}" ,file=f)
            print(f"manual ROI setting: {ROIsetting}" ,file=f)
            print(f"stackReg: {stackReg}" ,file=f)
            print(f"movieGeneration: {generateMovie}" ,file=f)
            print(f"GaussFilter: {gaussFilter}" ,file=f)
            print(f"Smoothing (Savitzky-Golay): {smoothing}" ,file=f)
            print(f"SmoothingWindow (Savitzky-Golay, seconds): {smoothingWindow}" ,file=f)
            print(f"usePreviousAnalysis: {usePreviousAnalysis}" ,file=f)
            print(f"usePreviousAnalysisDir: {usePreviousAnalysisDir}" ,file=f)
            print(f"use BG frames for: {BGframes} frames" ,file=f)
        

        ### memory release
        del img
        del npF1RAW
        del blurStack
        del imgSlice
        del npF1
        del npStack
        del npMip
        del xStack
        del xMip
        del xSt
        del npSt
        del npT
        del npF1RawMed
        del npBin
        del npThr
        del npThrBool
        del npThrBg
        del BgPlusStd
        del bgGreaterMip

    endtime = datetime.datetime.now()
    print(f"Start: {starttime}")
    print(f"End:   {endtime}")
    print("Results exported")



# divided from dif()
def fit(func, x, param_init, boundLimit, frame):
    """
    func:データxに近似したい任意の関数
    x:データ
    param_init:パラメータの初期値
    popt:最適化されたパラメータ
    pocv:パラメータの共分散
    """
    ### Fitting function
    sample_x = np.arange(0, frame-1, 0.01) # Image range for display

    X = x[0]
    Y = x[1]
    popt,pocv=sp.optimize.curve_fit(func, X, Y, p0=param_init, bounds=boundLimit)
    perr = np.sqrt(np.diag(pocv)) # The diagonal elements correspond to the SE of each parameter
    y=func(sample_x, *popt)
    return y, popt, perr


def imageFit(i, generateMovie, smoothing, smoothingWindow, fitFunction, fps):
    # send by np
    row = i[0]
    col = i[1]
    frame = npT.size

    # definition of resultArray
    HMS1 = [row, col, 0]
    HMS2 = [row, col, 0]
    Smax = [row, col, 0]
    Smin = [row, col, 0]
    HMT1 = [row, col, 0]
    HMT2 = [row, col, 0]
    HWTi = [row, col, 0]
    MaxT = [row, col, 0]
    ThrT = [row, col, 0]
    poptList = [row, col, 0, []] # Number of elements in popt and list of elements
    
    resultList = [HMS1, HMS2, Smax, Smin, HMT1, HMT2, HWTi, MaxT, ThrT, poptList]

    if generateMovie:
        slop = [row, col, [np.zeros(frame).tolist()]] # Insert the differentiated dataset
        resultList = [HMS1, HMS2, Smax, Smin, HMT1, HMT2, HWTi, MaxT, ThrT, poptList, slop]
    else:
        pass

    npThrValue = npThr[row,col] # shape: t, row, col
    if npThrValue == 0: # Skip evaluation for areas where the value is 0 after binarization.
        return resultList
    else:
        pass

    if bgGreaterMip[row,col]: # An array storing boolean values. True when bg is higher than Mip.

        return resultList
    else:
        pass

    npStRowCol = npSt[:,row,col] # shape: t, row, col
    npStRowColNA = np.isnan(npStRowCol).sum() != 0 # Returns True if there is even one NA
    if npStRowColNA == True:
        return resultList
    else:
        pass

    npIn = np.stack([npT, npStRowCol]) # 2D numpy array

    mip = npMip[row, col] - npF1[row, col] # valuable"mip" = MIP - F1
    maxTotal = np.amax(npIn[1]) # max value of npIn
    mipInd = np.where(npIn[1]==maxTotal) # coordinate in npIn
    mipT =  int(npIn[0][mipInd[0][0]]) # Obtain the matrix coordinates of mipInd and get time as mipT

    bgThrValue = npSt[0, row, col] + bgStd # [row, col] # Get the first time when the threshold is exceeded
    thrInd = np.where(npIn[1]>=bgThrValue)
    if len(thrInd[0])==0: # Error handling for when thrInd becomes NaN
        thrT =  int(mipT)
    else:
        thrT =  int(npIn[0][thrInd[0][0]])        


    if smoothing:
        #window_length
        roundFps = int(round(fps*smoothingWindow, 0)) # time window for specified time length
        if roundFps % 2 == 0:
            winLength = roundFps + 1
        else:
            winLength = roundFps
        
        if winLength < 5:
            winLength = 5
                        
        npIn = scipy.signal.savgol_filter(npIn, window_length = winLength, polyorder = 3) # smoothing with savgol filter
    
    ### Execute the function specified by fitFunction
    if fitFunction == "exGauss":
        def exGauss(x, K, xcoef, ycoef):
            x = x - xcoef
            co = (1/(2*K))*np.exp(1/(2*K**2)-x/K)
            x_erf = (-x+1/K)/np.sqrt(2)
            y = co * (1.0 - sp.special.erf(x_erf)) * ycoef # erfc(x) = 1 - erf(x)
            return y
        K = 10 # coefficiency
        xcoef = mipT  # Horizontal intersept
        ycoef = 100   # scale factor of y
        try:
            result = fit(exGauss, npIn, [K, xcoef, ycoef], (0,np.Inf), frame)
            popt = result[1] # returned coefficient
        except:
            return resultList
        def fitResult(x):
            K     = popt[0]
            xcoef = popt[1]
            ycoef = popt[2]
            x = x - xcoef
            co = (1/(2*K))*np.exp(1/(2*K**2)-x/K)
            x_erf = (-x+1/K)/np.sqrt(2)
            y = co * (1.0 - sp.special.erf(x_erf)) * ycoef
            ySt = y * mip # y = (F-F1)/mip, ySt = F-F1. F = raw value, mip = Fmax-F1
            return ySt
    else:
        pass


    ### Function for differentiation
    def f(f, x, h):
        return (f(x+h))
    def derivative(f, x, h):
        return  (f(x+h)-f(x-h))/(2*h) # Central difference method, Store the fit result "fData" and its derivative value "fDataDeriv" as a List
    h = 0.1 # Specification of small distance for differentiation. Rate-limiting step.
    repRange = np.arange(0, frame, h)

    try:
        fData = [f(fitResult, x, h) for x in repRange] # Repeatedly obtain the fitted values
        fDataDeriv = [derivative(fitResult, x, h) for x in repRange]   
    except:
        return resultList

    # fData and related data are derived from fitResult, so MIP (Maximum Intensity Projection) is already applied. The horizontal axis is not time but the value obtained by time/h.
    fDataMax = np.max(fData) # max intensity in fitted value
    fDataHalfMax = 1/2 * fDataMax # half-max in fitted value
    fDataDerivMax = np.max(fDataDeriv) # max slope
    fDataDerivMin = np.min(fDataDeriv) # min slope
    maxT = fData.index(fDataMax)*h # Time to reach fmax

    ### for plotting
    # plt.plot(fData)
    # plt.plot(fDataDeriv)
    # plt.show()

    ### Calculate numerical solution using fsolve
    def halfmaxRoot(x):
        return abs(fitResult(x) - fDataHalfMax)
    halfmaxT1 = sp.optimize.minimize_scalar(halfmaxRoot, method='bounded', bounds=(0, maxT))
    halfmaxT2 = sp.optimize.minimize_scalar(halfmaxRoot, method='bounded', bounds=(maxT, frame))

    halfmaxT1 = halfmaxT1.x # extract from list
    halfmaxT2 = halfmaxT2.x
    halfWidthT = halfmaxT2 - halfmaxT1 # duration
    halfMaxSlope1 = derivative(fitResult, halfmaxT1, h)
    halfMaxSlope2 = derivative(fitResult, halfmaxT2, h)
    
    ### Assign array values
    resultList[0][2] = halfMaxSlope1 # HMS1
    resultList[1][2] = halfMaxSlope2 # HMS2
    resultList[2][2] = fDataDerivMax # Smax
    resultList[3][2] = fDataDerivMin # Smin
    resultList[4][2] = halfmaxT1     # HMT1
    resultList[5][2] = halfmaxT2     # HMT2
    resultList[6][2] = halfWidthT    # HWTi
    resultList[7][2] = maxT          # MaxT
    resultList[8][2] = thrT          # thrT
    resultList[9][2] = len(popt)     # number of popt factors
    resultList[9][3] = popt          # popt
    
    ### Assign slope to List[10]
    if generateMovie:
        step = int(len(fData)/frame) # step number in 1 frame
        slop = fDataDeriv[0:len(fData):step] # Retrieve values repeatedly at each step number
        resultList[10][2] = slop
    else:
        pass

    return resultList

