# %%
import os 
import glob
import numpy as np
import pandas as pd
import threading
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RangeSlider, TextBox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.style as mplstyle
mplstyle.use('fast')
import tkinter as tk
import PyQt5.QtWidgets as QtWidgets
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

ipynbFolderName = os.path.basename(os.getcwd())

#%%
# for movie export, standardize numpyarray to unit8
def array_to_img(x):
    if x.min() == x.max():
        return np.zeros_like(x, dtype=np.uint8)
    img = 255 * (x - x.min()) / (x.max() - x.min())
    img = img.astype(np.uint8)
    return img


#%%

### remove outlier (>IQR*1.5)
def outlier(arrImg):
    arrImgOut = arrImg.copy()
    arrImgOut[arrImgOut <=0] = np.NaN
    per3rd = np.nanpercentile(arrImgOut, 75)
    per1st = np.nanpercentile(arrImgOut, 25)
    outlierVal = per3rd+(per3rd-per1st)*1.5 #Outlier value
    arrImgOut[arrImgOut >= outlierVal] = outlierVal  #Outlier value
    arrImgOut = np.nan_to_num(arrImgOut)
    # Find frames with high total brightness values
    sum1 = np.sum(arrImgOut, axis=1)
    sum2 = np.sum(sum1, axis=1)
    maxFrame = np.argmax(sum2)
    # Set outliers to a fixed value using IQR*1.5 in maxFrame. Adjust values less than or equal to 0 to 0.
    arrImgOut = arrImg.copy() # overwrite arrImgOut
    arrImgOut[arrImgOut <=0] = np.NaN
    per3rd = np.nanpercentile(arrImgOut[maxFrame], 75) #ignore NaN
    per1st = np.nanpercentile(arrImgOut[maxFrame], 25) #ignore NaN
    outlierVal = per3rd+(per3rd-per1st)*1.5
    arrImgOut[arrImgOut >= outlierVal] = outlierVal
    arrImgOut = np.nan_to_num(arrImgOut)
    return arrImgOut

#%%

# Enable retrieval of paths containing []
def globOutWoBlace(pathDir):
    exportDir = pathDir.replace("[","\\[").replace("]","\\]")
    exportDir = exportDir.replace("\\[","[[]").replace("\\]","[]]")
    exportDir += "/*"
    dirList = glob.glob(exportDir)
    for i in dirList[:]:
        imgName = os.path.basename(i)
        extList = os.path.splitext(i)    
        if imgName == "IN":
            dirList.remove(i)      
        elif imgName == "ROI.csv":
            dirList.remove(i)      
        elif imgName == "threshold.npy":
            dirList.remove(i)      
        elif extList[1] == ".ipynb":
            dirList.remove(i)      
        
    return dirList

def globWoBlace(pathDir):
    exportDir = pathDir.replace("[","\\[").replace("]","\\]")
    exportDir = exportDir.replace("\\[","[[]").replace("\\]","[]]")
    exportDir += "/*"
    dirList = glob.glob(exportDir)
    return dirList
    


# %%
### summarizeFitting
def autoSummarize(imgPath, rawOrSt, colormap, colormap2, a, b, c, d, e):
    '''
    colormap  : SlopeとDuration用
    colormap2 : Time用
    a         : Slope用vmaxのpercentile
    b         : Slope用vminのpercentile
    c         : Duration用vmaxのpercentile
    d         : Time用vmaxのpercentile
    e         : Time用vminのpercentile
    '''
    os.chdir(imgPath)
    imgName = os.path.basename(imgPath)

    ### import
    #                0     1     2     3     4     5     6     7     8
    # resultArray = [HMS1, HMS2, Smax, Smin, HMT1, HMT2, HWTi, MaxT, ThrT]
    if rawOrSt == "raw":
        outNp = f"./resultArray/{imgName}_resultRAWNp.npy"
        resultNpFps = np.load(outNp)
    elif rawOrSt == "st":
        outNp = f"./resultArray/{imgName}_resultStNp.npy"
        resultNpFps = np.load(outNp)        
    else:
        pass        

    ### calculte summary for vmax and vmin
    nanPad = np.copy(resultNpFps)
    
    
    # Mask unnecessary elements in the plot
    npMask0 = np.ma.masked_where(resultNpFps[0] <= 0, resultNpFps[0])
    npMask1 = np.ma.masked_where(resultNpFps[1] >= 0, resultNpFps[1])
    npMask2 = np.ma.masked_where(resultNpFps[2] <= 0, resultNpFps[2])
    npMask3 = np.ma.masked_where(resultNpFps[3] >= 0, resultNpFps[3])
    npMask4 = np.ma.masked_where(resultNpFps[4] <= 0, resultNpFps[4])
    npMask5 = np.ma.masked_where(resultNpFps[5] <= 0, resultNpFps[5])
    npMask6 = np.ma.masked_where(resultNpFps[6] <= 0, resultNpFps[6])
    npMask7 = np.ma.masked_where(resultNpFps[7] <= 0, resultNpFps[7])
    npMask8 = np.ma.masked_where(resultNpFps[8] <= 0, resultNpFps[8])


    # Remove negative and positive values
    nanPad[0][nanPad[0] <= 0] = np.NaN
    nanPad[1][nanPad[1] >= 0] = np.NaN
    nanPad[2][nanPad[2] <= 0] = np.NaN
    nanPad[3][nanPad[3] >= 0] = np.NaN
    nanPad[4][nanPad[4] <= 0] = np.NaN
    nanPad[5][nanPad[5] <= 0] = np.NaN
    nanPad[6][nanPad[6] <= 0] = np.NaN
    nanPad[7][nanPad[7] <= 0] = np.NaN
    nanPad[8][nanPad[8] <= 0] = np.NaN
        
    nanSummary0 = pd.DataFrame(pd.Series(nanPad[0].ravel()).describe()).transpose()
    nanSummary1 = pd.DataFrame(pd.Series(nanPad[1].ravel()).describe()).transpose()
    nanSummary2 = pd.DataFrame(pd.Series(nanPad[2].ravel()).describe()).transpose()
    nanSummary3 = pd.DataFrame(pd.Series(nanPad[3].ravel()).describe()).transpose()
    nanSummary4 = pd.DataFrame(pd.Series(nanPad[4].ravel()).describe()).transpose()
    nanSummary5 = pd.DataFrame(pd.Series(nanPad[5].ravel()).describe()).transpose()
    nanSummary6 = pd.DataFrame(pd.Series(nanPad[6].ravel()).describe()).transpose()
    nanSummary7 = pd.DataFrame(pd.Series(nanPad[7].ravel()).describe()).transpose()
    nanSummary8 = pd.DataFrame(pd.Series(nanPad[8].ravel()).describe()).transpose()

    summary = pd.concat([nanSummary0, nanSummary1,nanSummary2,nanSummary3,nanSummary4,nanSummary5,nanSummary6,nanSummary7, nanSummary8])
    summary["mean+SD"] = summary["mean"] +summary["std"]*2
    summary["mean-SD"] = summary["mean"] -summary["std"]*2
    summaryR = summary.reset_index()

    '''
    SD+-1~3.5
    1.0 SD: 68.3 %
    1.5 SD: 86.6 %
    2.0 SD: 95.4 %
    2.5 SD: 98.8 %
    3.0 SD: 99.7 %
    3.5 SD: 100.0 %
    '''

    ### add percentile result to summary
    percent0 = np.nanpercentile(nanPad[0],     a)
    percent1 = np.nanpercentile(nanPad[1], 100-a)
    percent2 = np.nanpercentile(nanPad[2],     a)
    percent3 = np.nanpercentile(nanPad[3], 100-a)
    percent4 = np.nanpercentile(nanPad[4],     d)
    percent5 = np.nanpercentile(nanPad[5],     d)
    percent6 = np.nanpercentile(nanPad[6],     c)
    percent7 = np.nanpercentile(nanPad[7],     d)
    percent8 = np.nanpercentile(nanPad[8],     d)

    lowPercent0 = np.nanpercentile(nanPad[0],     b)
    lowPercent1 = np.nanpercentile(nanPad[1],     100-b)
    lowPercent2 = np.nanpercentile(nanPad[2],     b)
    lowPercent3 = np.nanpercentile(nanPad[3],     100-b)
    lowPercent4 = np.nanpercentile(nanPad[4],     e)
    lowPercent5 = np.nanpercentile(nanPad[5],     e)
    lowPercent7 = np.nanpercentile(nanPad[7],     e)
    lowPercent8 = np.nanpercentile(nanPad[8],     e)


    percent = pd.DataFrame({'percent': [percent0, percent1, percent2, percent3, percent4, percent5, percent6, percent7, percent8]})
    summaryR = pd.concat([summaryR, percent], axis=1)

    ### plotSummary
    # the RAW intensity/second corrected by fps is shown as the slope.

    # inputNp
    #0        1     2      3     4
    #npF1RAW, npF1, npMip, npThr npRAWmean
    outNp2 = f"./resultArray/{imgName}_inputNp.npy"
    inputNp = np.load(outNp2)
    npThrmask = np.ma.masked_where(inputNp[3] == 255, inputNp[3])

    fig, ax=plt.subplots(3,4, figsize=(20,12),subplot_kw=({"xticks":(), "yticks":()}))
    colormap_r = f"{colormap}_r" # color

    ax[0,0].set_title("IncreaseSPD at HalfMax")
    ax[0,1].set_title("DecreaseSPD at HalfMax")
    ax[0,2].set_title("MaxSPD")
    ax[0,3].set_title("MinSPD")
    ax[1,0].set_title("Duration")
    ax[1,1].set_title("Time at threshold")
    ax[1,2].set_title("Time at HalfMax")
    ax[1,3].set_title("Time at Max")
    ax[2,0].set_title("Mean intensity projection")
    ax[2,1].set_title("IncreaseSPD at HalfMax (Overlay)")
    ax[2,2].set_title("Duration (Overlay)")
    ax[2,3].set_title("Time at HalfMax (Overlay)")

    ### data
    #                0     1     2     3     4     5     6     7     8
    # resultArray = [HMS1, HMS2, Smax, Smin, HMT1, HMT2, HWTi, MaxT, ThrT]
 
    ### main plot
    # The scale is aligned with the largest value in the same series.    

    # BG color in plot
    ax[0,0].set_facecolor('#C0C0C0')
    ax[0,1].set_facecolor('#C0C0C0')
    ax[0,2].set_facecolor('#C0C0C0')
    ax[0,3].set_facecolor('#C0C0C0')
    ax[1,0].set_facecolor('#C0C0C0')
    ax[1,1].set_facecolor('#C0C0C0')
    ax[1,2].set_facecolor('#C0C0C0')
    ax[1,3].set_facecolor('#C0C0C0')
    ax[2,0].set_facecolor('#C0C0C0')
    ax[2,1].set_facecolor('#C0C0C0')
    ax[2,2].set_facecolor('#C0C0C0')
    ax[2,3].set_facecolor('#C0C0C0')

    heatmap1  = ax[0,0].imshow(npMask0,cmap=colormap  , vmin = lowPercent0, vmax = summaryR.loc[0, "percent"]) 
    heatmap2  = ax[0,1].imshow(npMask1,cmap=colormap_r, vmax = lowPercent1, vmin = summaryR.loc[1, "percent"]) # positive-negative reverse
    heatmap3  = ax[0,2].imshow(npMask2,cmap=colormap  , vmin = lowPercent2, vmax = summaryR.loc[2, "percent"]) 
    heatmap4  = ax[0,3].imshow(npMask3,cmap=colormap_r, vmax = lowPercent3, vmin = summaryR.loc[3, "percent"]) # positive-negative reverse
    heatmap5  = ax[1,0].imshow(npMask6,cmap=colormap  , vmin = 0          , vmax = summaryR.loc[6, "percent"]) 
    heatmap6  = ax[1,1].imshow(npMask8,cmap=colormap2 , vmin = lowPercent8, vmax = summaryR.loc[8, "percent"]) 
    heatmap7  = ax[1,2].imshow(npMask4,cmap=colormap2 , vmin = lowPercent4, vmax = summaryR.loc[4, "percent"]) 
    heatmap8  = ax[1,3].imshow(npMask7,cmap=colormap2 , vmin = lowPercent7, vmax = summaryR.loc[7, "percent"]) 
    heatmap9  = ax[2,0].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap10 = ax[2,1].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap11 = ax[2,2].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap12 = ax[2,3].imshow(inputNp[4],cmap="binary") #RAW mean

    ### overlay mask of the threshold
    # binary pixel
    ax[0,0].imshow(npThrmask, cmap="gray", vmin = -10, vmax = 25) 
    ax[0,1].imshow(npThrmask, cmap="gray", vmin = -10, vmax = 25) 
    ax[0,2].imshow(npThrmask, cmap="gray", vmin = -10, vmax = 25) 
    ax[0,3].imshow(npThrmask, cmap="gray", vmin = -10, vmax = 25) 
    ax[1,0].imshow(npThrmask, cmap="gray", vmin = -10, vmax = 25) 
    ax[1,1].imshow(npThrmask, cmap="gray", vmin = -10, vmax = 25) 
    ax[1,2].imshow(npThrmask, cmap="gray", vmin = -10, vmax = 25) 
    ax[1,3].imshow(npThrmask, cmap="gray", vmin = -10, vmax = 25) 

    # image overlay
    ax[2,1].imshow(npMask0,cmap=colormap  , vmin = lowPercent0, vmax = summaryR.loc[0, "percent"]) 
    ax[2,2].imshow(npMask6,cmap=colormap  , vmin = 0,           vmax = summaryR.loc[6, "percent"]) 
    ax[2,3].imshow(npMask4,cmap=colormap2 , vmin = lowPercent4, vmax = summaryR.loc[4, "percent"]) 


    # align colorbar sizes
    divider1 = make_axes_locatable(ax[0,0])
    divider2 = make_axes_locatable(ax[0,1])
    divider3 = make_axes_locatable(ax[0,2])
    divider4 = make_axes_locatable(ax[0,3])
    divider5 = make_axes_locatable(ax[1,0])
    divider6 = make_axes_locatable(ax[1,1])
    divider7 = make_axes_locatable(ax[1,2])
    divider8 = make_axes_locatable(ax[1,3])

    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cax4 = divider4.append_axes("right", size="5%", pad=0.1)
    cax5 = divider5.append_axes("right", size="5%", pad=0.1)
    cax6 = divider6.append_axes("right", size="5%", pad=0.1)
    cax7 = divider7.append_axes("right", size="5%", pad=0.1)
    cax8 = divider8.append_axes("right", size="5%", pad=0.1)

    fig.colorbar(heatmap1, cax=cax1)
    fig.colorbar(heatmap2, cax=cax2)
    fig.colorbar(heatmap3, cax=cax3)
    fig.colorbar(heatmap4, cax=cax4)
    fig.colorbar(heatmap5, cax=cax5)
    fig.colorbar(heatmap6, cax=cax6)
    fig.colorbar(heatmap7, cax=cax7)
    fig.colorbar(heatmap8, cax=cax8)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    fig.suptitle(f"{imgName}")

    summaryOutDir = f"./{imgName}_Summary.png"
    fig.savefig(summaryOutDir, format="png", dpi=300)    



# %%
### summarizeFitting
def IAsummarize(imgPath, colormap, colormap2):
    
    os.chdir(imgPath)
    imgName = os.path.basename(imgPath)
    print("Summarizing...")

    ### import
    #                0     1     2     3     4     5     6     7     8
    # resultArray = [HMS1, HMS2, Smax, Smin, HMT1, HMT2, HWTi, MaxT, ThrT]
    outNp = f"./resultArray/{imgName}_resultStNp.npy"
    resultNpFps = np.load(outNp)
    
    ### calculte summary for vmax and vmin
    nanPad = np.copy(resultNpFps)
    
    # Mask unnecessary elements in the plot
    npMask0 = np.ma.masked_where(resultNpFps[0] <= 0, resultNpFps[0])
    npMask1 = np.ma.masked_where(resultNpFps[1] >= 0, resultNpFps[1])
    npMask2 = np.ma.masked_where(resultNpFps[2] <= 0, resultNpFps[2])
    npMask3 = np.ma.masked_where(resultNpFps[3] >= 0, resultNpFps[3])
    npMask4 = np.ma.masked_where(resultNpFps[4] <= 0, resultNpFps[4])
    npMask5 = np.ma.masked_where(resultNpFps[5] <= 0, resultNpFps[5])
    npMask6 = np.ma.masked_where(resultNpFps[6] <= 0, resultNpFps[6])
    npMask7 = np.ma.masked_where(resultNpFps[7] <= 0, resultNpFps[7])
    npMask8 = np.ma.masked_where(resultNpFps[8] <= 0, resultNpFps[8])

    # Mask unnecessary elements in the plot
    nanPad[0][nanPad[0] <= 0] = np.NaN
    nanPad[1][nanPad[1] >= 0] = np.NaN
    nanPad[2][nanPad[2] <= 0] = np.NaN
    nanPad[3][nanPad[3] >= 0] = np.NaN
    nanPad[4][nanPad[4] <= 0] = np.NaN
    nanPad[5][nanPad[5] <= 0] = np.NaN
    nanPad[6][nanPad[6] <= 0] = np.NaN
    nanPad[7][nanPad[7] <= 0] = np.NaN
    nanPad[8][nanPad[8] <= 0] = np.NaN


    nanSummary0 = pd.DataFrame(pd.Series(nanPad[0].ravel()).describe()).transpose()
    nanSummary1 = pd.DataFrame(pd.Series(nanPad[1].ravel()).describe()).transpose()
    nanSummary2 = pd.DataFrame(pd.Series(nanPad[2].ravel()).describe()).transpose()
    nanSummary3 = pd.DataFrame(pd.Series(nanPad[3].ravel()).describe()).transpose()
    nanSummary4 = pd.DataFrame(pd.Series(nanPad[4].ravel()).describe()).transpose()
    nanSummary5 = pd.DataFrame(pd.Series(nanPad[5].ravel()).describe()).transpose()
    nanSummary6 = pd.DataFrame(pd.Series(nanPad[6].ravel()).describe()).transpose()
    nanSummary7 = pd.DataFrame(pd.Series(nanPad[7].ravel()).describe()).transpose()
    nanSummary8 = pd.DataFrame(pd.Series(nanPad[8].ravel()).describe()).transpose()

    summary = pd.concat([nanSummary0, nanSummary1,nanSummary2,nanSummary3,nanSummary4,nanSummary5,nanSummary6,nanSummary7, nanSummary8])
    summary["mean+SD"] = summary["mean"] +summary["std"]*2
    summary["mean-SD"] = summary["mean"] -summary["std"]*2
    summaryR = summary.reset_index()

    '''
    SD+-1~3.5
    1.0 SD: 68.3 %
    1.5 SD: 86.6 %
    2.0 SD: 95.4 %
    2.5 SD: 98.8 %
    3.0 SD: 99.7 %
    3.5 SD: 100.0 %
    '''

    ### add percentile result to summary
    percent0 = np.nanpercentile(nanPad[0],     99)
    percent1 = np.nanpercentile(nanPad[1], 100-99)
    percent2 = np.nanpercentile(nanPad[2],     99)
    percent3 = np.nanpercentile(nanPad[3], 100-99)
    percent4 = np.nanpercentile(nanPad[4],     95)
    percent5 = np.nanpercentile(nanPad[5],     95)
    percent6 = np.nanpercentile(nanPad[6],     95)
    percent7 = np.nanpercentile(nanPad[7],     95)
    percent8 = np.nanpercentile(nanPad[8],     95)

    percent = pd.DataFrame({'percent': [percent0, percent1, percent2, percent3, percent4, percent5, percent6, percent7, percent8]})
    summaryR = pd.concat([summaryR, percent], axis=1)

    ### plotSummary
    # the RAW intensity/second corrected by fps is shown as the slope.

    # inputNp
    #0        1     2      3     4
    #npF1RAW, npF1, npMip, npThr npRAWmean
    outNp2 = f"./resultArray/{imgName}_inputNp.npy"
    inputNp = np.load(outNp2)
    npThrmask = np.ma.masked_where(inputNp[3] == 255, inputNp[3])

    # backend
    root = tk.Tk()
    plt.switch_backend('Qt5Agg')
    fig, ax=plt.subplots(3, 4, figsize=(25,15), subplot_kw=({"xticks":(), "yticks":()}))
    axRows, axCols = ax.shape
    
    colormap_r = f"{colormap}_r"

    ax[0,0].set_title("IncreaseSPD at HalfMax")
    ax[0,1].set_title("DecreaseSPD at HalfMax")
    ax[0,2].set_title("MaxSPD")
    ax[0,3].set_title("MinSPD")
    ax[1,0].set_title("Duration")
    ax[1,1].set_title("Time at threshold")
    ax[1,2].set_title("Time at HalfMax")
    ax[1,3].set_title("Time at Max")
    ax[2,0].set_title("Mean intensity projection")
    ax[2,1].set_title("IncreaseSPD at HalfMax (Overlay)")
    ax[2,2].set_title("Duration (Overlay)")
    ax[2,3].set_title("Time at HalfMax (Overlay)")

    ## data
    #                0     1     2     3     4     5     6     7     8
    # resultArray = [HMS1, HMS2, Smax, Smin, HMT1, HMT2, HWTi, MaxT, ThrT]
    
    ## main plot
    # The scale is aligned with the largest value in the same series.    

    # BG color in plot
    ax[0,0].set_facecolor('#C0C0C0')
    ax[0,1].set_facecolor('#C0C0C0')
    ax[0,2].set_facecolor('#C0C0C0')
    ax[0,3].set_facecolor('#C0C0C0')
    ax[1,0].set_facecolor('#C0C0C0')
    ax[1,1].set_facecolor('#C0C0C0')
    ax[1,2].set_facecolor('#C0C0C0')
    ax[1,3].set_facecolor('#C0C0C0')
    ax[2,0].set_facecolor('#C0C0C0')
    ax[2,1].set_facecolor('#C0C0C0')
    ax[2,2].set_facecolor('#C0C0C0')
    ax[2,3].set_facecolor('#C0C0C0')

    heatmap1  = ax[0,0].imshow(npMask0,cmap=colormap  , vmin = 0, vmax = summaryR.loc[0, "percent"]) 
    heatmap2  = ax[0,1].imshow(npMask1,cmap=colormap_r, vmax = 0, vmin = summaryR.loc[1, "percent"]) # positive-negative reverse
    heatmap3  = ax[0,2].imshow(npMask2,cmap=colormap  , vmin = 0, vmax = summaryR.loc[2, "percent"]) 
    heatmap4  = ax[0,3].imshow(npMask3,cmap=colormap_r, vmax = 0, vmin = summaryR.loc[3, "percent"]) # positive-negative reverse
    heatmap5  = ax[1,0].imshow(npMask6,cmap=colormap  , vmin = 0, vmax = summaryR.loc[6, "percent"]) 
    heatmap6  = ax[1,1].imshow(npMask8,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[8, "percent"]) 
    heatmap7  = ax[1,2].imshow(npMask4,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[4, "percent"]) 
    heatmap8  = ax[1,3].imshow(npMask7,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[7, "percent"]) 
    heatmap9  = ax[2,0].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap10 = ax[2,1].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap11 = ax[2,2].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap12 = ax[2,3].imshow(inputNp[4],cmap="binary") #RAW mean

    ### overlay mask of the threshold
    # binary pixel
    ax[0,0].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[0,1].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[0,2].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[0,3].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,0].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,1].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,2].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,3].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 

    # image overlay
    ax[2,1].imshow(npMask0,cmap=colormap  , vmin = 0, vmax = summaryR.loc[0, "percent"])
    ax[2,2].imshow(npMask6,cmap=colormap  , vmin = 0, vmax = summaryR.loc[6, "percent"]) 
    ax[2,3].imshow(npMask4,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[4, "percent"]) 

    # align colorbar sizes
    divider1 = make_axes_locatable(ax[0,0])
    divider2 = make_axes_locatable(ax[0,1])
    divider3 = make_axes_locatable(ax[0,2])
    divider4 = make_axes_locatable(ax[0,3])
    divider5 = make_axes_locatable(ax[1,0])
    divider6 = make_axes_locatable(ax[1,1])
    divider7 = make_axes_locatable(ax[1,2])
    divider8 = make_axes_locatable(ax[1,3])

    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cax4 = divider4.append_axes("right", size="5%", pad=0.1)
    cax5 = divider5.append_axes("right", size="5%", pad=0.1)
    cax6 = divider6.append_axes("right", size="5%", pad=0.1)
    cax7 = divider7.append_axes("right", size="5%", pad=0.1)
    cax8 = divider8.append_axes("right", size="5%", pad=0.1)

    global cbar1
    global cbar2
    global cbar3
    global cbar4
    global cbar5
    global cbar6
    global cbar7
    global cbar8

    cbar1 = fig.colorbar(heatmap1, cax=cax1)
    cbar2 = fig.colorbar(heatmap2, cax=cax2)
    cbar3 = fig.colorbar(heatmap3, cax=cax3)
    cbar4 = fig.colorbar(heatmap4, cax=cax4)
    cbar5 = fig.colorbar(heatmap5, cax=cax5)
    cbar6 = fig.colorbar(heatmap6, cax=cax6)
    cbar7 = fig.colorbar(heatmap7, cax=cax7)
    cbar8 = fig.colorbar(heatmap8, cax=cax8)

    # axes([left, bottom, width, height])
    ax_a = plt.axes([0.25, 0.09, 0.50, 0.02], facecolor='gold')
    ax_b = plt.axes([0.25, 0.07, 0.50, 0.02], facecolor='gold')
    ax_c = plt.axes([0.25, 0.05, 0.50, 0.02], facecolor='gold')
    ax_d = plt.axes([0.25, 0.03, 0.50, 0.02], facecolor='gold')
    ax_f = plt.axes([0.85, 0.03, 0.1, 0.05], facecolor='green')


    sli_a = RangeSlider(ax_a, "Increase slope percentile", valmin=0, valmax=100, valinit=[0,100], valstep=0.1, dragging=True)
    sli_b = RangeSlider(ax_b, "Decrease slope percentile", valmin=0, valmax=100, valinit=[0,100], valstep=0.1, dragging=True)
    sli_c = RangeSlider(ax_c, "Duration percentile", valmin=0, valmax=100, valinit=[0,100], valstep=0.1, dragging=True)
    sli_d = RangeSlider(ax_d, 'Time percentile', valmin=0, valmax=100, valinit=[0,100], valstep=0.1, dragging=True)
    button = Button(ax_f, "Done")

    def update(val):
 
        sli_A_min, sli_A_max = sli_a.val
        sli_B_min, sli_B_max = sli_b.val
        sli_C_min, sli_C_max = sli_c.val
        sli_D_min, sli_D_max = sli_d.val
        
        global cbar1
        global cbar2
        global cbar3
        global cbar4
        global cbar5
        global cbar6
        global cbar7
        global cbar8

        cbar1.remove()
        cbar2.remove()
        cbar3.remove()
        cbar4.remove()
        cbar5.remove()
        cbar6.remove()
        cbar7.remove()
        cbar8.remove()
              
        for i in range(axRows):
            for j in range(axCols):
                ax[i, j].clear()
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                    
        ax[0,0].set_title("IncreaseSPD at HalfMax")
        ax[0,1].set_title("DecreaseSPD at HalfMax")
        ax[0,2].set_title("MaxSPD")
        ax[0,3].set_title("MinSPD")
        ax[1,0].set_title("Duration")
        ax[1,1].set_title("Time at threshold")
        ax[1,2].set_title("Time at HalfMax")
        ax[1,3].set_title("Time at Max")
        ax[2,0].set_title("Mean intensity projection")
        ax[2,1].set_title("IncreaseSPD at HalfMax (Overlay)")
        ax[2,2].set_title("Duration (Overlay)")
        ax[2,3].set_title("Time at HalfMax (Overlay)")
        
        percent0 = np.nanpercentile(nanPad[0],     sli_A_max)
        percent1 = np.nanpercentile(nanPad[1], 100-sli_B_max)
        percent2 = np.nanpercentile(nanPad[2],     sli_A_max)
        percent3 = np.nanpercentile(nanPad[3], 100-sli_B_max)
        percent4 = np.nanpercentile(nanPad[4],     sli_D_max)
        percent5 = np.nanpercentile(nanPad[5],     sli_D_max)
        percent6 = np.nanpercentile(nanPad[6],     sli_C_max)
        percent7 = np.nanpercentile(nanPad[7],     sli_D_max)
        percent8 = np.nanpercentile(nanPad[8],     sli_D_max)

        lowPercent0 = np.nanpercentile(nanPad[0],     sli_A_min)
        lowPercent1 = np.nanpercentile(nanPad[1],     100-sli_B_min)
        lowPercent2 = np.nanpercentile(nanPad[2],     sli_A_min)
        lowPercent3 = np.nanpercentile(nanPad[3],     100-sli_B_min)
        lowPercent4 = np.nanpercentile(nanPad[4],     sli_D_min)
        lowPercent5 = np.nanpercentile(nanPad[5],     sli_D_min)
        lowPercent6 = np.nanpercentile(nanPad[6],     sli_C_min)
        lowPercent7 = np.nanpercentile(nanPad[7],     sli_D_min)
        lowPercent8 = np.nanpercentile(nanPad[8],     sli_D_min)

        heatmap1  = ax[0,0].imshow(npMask0,    cmap = colormap  , vmin = lowPercent0, vmax = percent0) 
        heatmap2  = ax[0,1].imshow(npMask1,    cmap = colormap_r, vmax = lowPercent1, vmin = percent1) #positive-negative reverse
        heatmap3  = ax[0,2].imshow(npMask2,    cmap = colormap  , vmin = lowPercent2, vmax = percent2) 
        heatmap4  = ax[0,3].imshow(npMask3,    cmap = colormap_r, vmax = lowPercent3, vmin = percent3) #positive-negative reverse
        heatmap5  = ax[1,0].imshow(npMask6,    cmap = colormap  , vmin = lowPercent6, vmax = percent6) 
        heatmap6  = ax[1,1].imshow(npMask8,    cmap = colormap2 , vmin = lowPercent8, vmax = percent8) 
        heatmap7  = ax[1,2].imshow(npMask4,    cmap = colormap2 , vmin = lowPercent4, vmax = percent4) 
        heatmap8  = ax[1,3].imshow(npMask7,    cmap = colormap2 , vmin = lowPercent7, vmax = percent7) 
        heatmap9  = ax[2,0].imshow(inputNp[4], cmap ="binary") #RAW mean
        heatmap10 = ax[2,1].imshow(inputNp[4], cmap ="binary") #RAW mean
        heatmap11 = ax[2,2].imshow(inputNp[4], cmap ="binary") #RAW mean
        heatmap12 = ax[2,3].imshow(inputNp[4], cmap ="binary") #RAW mean

        ### overlay mask of the threshold
        # binary pixel
        ax[0,0].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[0,1].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[0,2].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[0,3].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,0].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,1].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,2].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,3].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 

        # image overlay
        ax[2,1].imshow(npMask0,cmap=colormap  , vmin = lowPercent0, vmax = percent0)
        ax[2,2].imshow(npMask6,cmap=colormap  , vmin = lowPercent6, vmax = percent6) 
        ax[2,3].imshow(npMask4,cmap=colormap2 , vmin = lowPercent4, vmax = percent4) 

        # align colorbar sizes
        divider1 = make_axes_locatable(ax[0,0])
        divider2 = make_axes_locatable(ax[0,1])
        divider3 = make_axes_locatable(ax[0,2])
        divider4 = make_axes_locatable(ax[0,3])
        divider5 = make_axes_locatable(ax[1,0])
        divider6 = make_axes_locatable(ax[1,1])
        divider7 = make_axes_locatable(ax[1,2])
        divider8 = make_axes_locatable(ax[1,3])

        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        cax4 = divider4.append_axes("right", size="5%", pad=0.1)
        cax5 = divider5.append_axes("right", size="5%", pad=0.1)
        cax6 = divider6.append_axes("right", size="5%", pad=0.1)
        cax7 = divider7.append_axes("right", size="5%", pad=0.1)
        cax8 = divider8.append_axes("right", size="5%", pad=0.1)

        cbar1 = fig.colorbar(heatmap1, cax=cax1)
        cbar2 = fig.colorbar(heatmap2, cax=cax2)
        cbar3 = fig.colorbar(heatmap3, cax=cax3)
        cbar4 = fig.colorbar(heatmap4, cax=cax4)
        cbar5 = fig.colorbar(heatmap5, cax=cax5)
        cbar6 = fig.colorbar(heatmap6, cax=cax6)
        cbar7 = fig.colorbar(heatmap7, cax=cax7)
        cbar8 = fig.colorbar(heatmap8, cax=cax8)
        fig.canvas.blit()
     
    def save_and_close_figure():
        summaryOutDir = f"./{imgName}_manualSummary.png"
        fig.savefig(summaryOutDir, format="png", dpi=300)
        plt.close()

    def doneButton(self):
        thread = threading.Thread(target=save_and_close_figure)
        thread.start()

    fig.canvas.mpl_connect("close_event", doneButton)
    button.on_clicked(doneButton)

    sli_a.on_changed(update)
    sli_b.on_changed(update)
    sli_c.on_changed(update)
    sli_d.on_changed(update)
   
    fig.suptitle(f"{imgName}")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.90)
    plt.show()
    

# %%
### summarizeFitting
def IAsummarizePer(imgPath, colormap, colormap2):
    
    os.chdir(imgPath)
    imgName = os.path.basename(imgPath)
    print("Summarizing...")

    ### import
    #                0     1     2     3     4     5     6     7     8
    # resultArray = [HMS1, HMS2, Smax, Smin, HMT1, HMT2, HWTi, MaxT, ThrT]
    outNp = f"./resultArray/{imgName}_resultStNp.npy"
    resultNpFps = np.load(outNp)
    
    ### calculte summary for vmax and vmin
    nanPad = np.copy(resultNpFps)
    
    # Mask unnecessary elements in the plot
    npMask0 = np.ma.masked_where(resultNpFps[0] <= 0, resultNpFps[0])
    npMask1 = np.ma.masked_where(resultNpFps[1] >= 0, resultNpFps[1])
    npMask2 = np.ma.masked_where(resultNpFps[2] <= 0, resultNpFps[2])
    npMask3 = np.ma.masked_where(resultNpFps[3] >= 0, resultNpFps[3])
    npMask4 = np.ma.masked_where(resultNpFps[4] <= 0, resultNpFps[4])
    npMask5 = np.ma.masked_where(resultNpFps[5] <= 0, resultNpFps[5])
    npMask6 = np.ma.masked_where(resultNpFps[6] <= 0, resultNpFps[6])
    npMask7 = np.ma.masked_where(resultNpFps[7] <= 0, resultNpFps[7])
    npMask8 = np.ma.masked_where(resultNpFps[8] <= 0, resultNpFps[8])


    # Mask unnecessary elements in the plot
    nanPad[0][nanPad[0] <= 0] = np.NaN
    nanPad[1][nanPad[1] >= 0] = np.NaN
    nanPad[2][nanPad[2] <= 0] = np.NaN
    nanPad[3][nanPad[3] >= 0] = np.NaN
    nanPad[4][nanPad[4] <= 0] = np.NaN
    nanPad[5][nanPad[5] <= 0] = np.NaN
    nanPad[6][nanPad[6] <= 0] = np.NaN
    nanPad[7][nanPad[7] <= 0] = np.NaN
    nanPad[8][nanPad[8] <= 0] = np.NaN



    nanSummary0 = pd.DataFrame(pd.Series(nanPad[0].ravel()).describe()).transpose()
    nanSummary1 = pd.DataFrame(pd.Series(nanPad[1].ravel()).describe()).transpose()
    nanSummary2 = pd.DataFrame(pd.Series(nanPad[2].ravel()).describe()).transpose()
    nanSummary3 = pd.DataFrame(pd.Series(nanPad[3].ravel()).describe()).transpose()
    nanSummary4 = pd.DataFrame(pd.Series(nanPad[4].ravel()).describe()).transpose()
    nanSummary5 = pd.DataFrame(pd.Series(nanPad[5].ravel()).describe()).transpose()
    nanSummary6 = pd.DataFrame(pd.Series(nanPad[6].ravel()).describe()).transpose()
    nanSummary7 = pd.DataFrame(pd.Series(nanPad[7].ravel()).describe()).transpose()
    nanSummary8 = pd.DataFrame(pd.Series(nanPad[8].ravel()).describe()).transpose()

    summary = pd.concat([nanSummary0, nanSummary1,nanSummary2,nanSummary3,nanSummary4,nanSummary5,nanSummary6,nanSummary7, nanSummary8])
    summary["mean+SD"] = summary["mean"] +summary["std"]*2
    summary["mean-SD"] = summary["mean"] -summary["std"]*2
    summaryR = summary.reset_index()

    '''
    SD+-1~3.5
    1.0 SD: 68.3 %
    1.5 SD: 86.6 %
    2.0 SD: 95.4 %
    2.5 SD: 98.8 %
    3.0 SD: 99.7 %
    3.5 SD: 100.0 %
    '''

    ### add percentile result to summary
    percent0 = np.nanpercentile(nanPad[0],     100)
    percent1 = np.nanpercentile(nanPad[1], 100-100)
    percent2 = np.nanpercentile(nanPad[2],     100)
    percent3 = np.nanpercentile(nanPad[3], 100-100)
    percent4 = np.nanpercentile(nanPad[4],     100)
    percent5 = np.nanpercentile(nanPad[5],     100)
    percent6 = np.nanpercentile(nanPad[6],     100)
    percent7 = np.nanpercentile(nanPad[7],     100)
    percent8 = np.nanpercentile(nanPad[8],     100)

    percent = pd.DataFrame({'percent': [percent0, percent1, percent2, percent3, percent4, percent5, percent6, percent7, percent8]})
    summaryR = pd.concat([summaryR, percent], axis=1)

    ### plotSummary
    # the RAW intensity/second corrected by fps is shown as the slope.

    # inputNp
    #0        1     2      3     4
    #npF1RAW, npF1, npMip, npThr npRAWmean
    outNp2 = f"./resultArray/{imgName}_inputNp.npy"
    inputNp = np.load(outNp2)
    npThrmask = np.ma.masked_where(inputNp[3] == 255, inputNp[3])

    # backend
    root = tk.Tk()
    plt.switch_backend('Qt5Agg')
    fig, ax=plt.subplots(3, 4, figsize=(25,15), subplot_kw=({"xticks":(), "yticks":()}))
    axRows, axCols = ax.shape
    
    colormap_r = f"{colormap}_r"

    ax[0,0].set_title("IncreaseSPD at HalfMax")
    ax[0,1].set_title("DecreaseSPD at HalfMax")
    ax[0,2].set_title("MaxSPD")
    ax[0,3].set_title("MinSPD")
    ax[1,0].set_title("Duration")
    ax[1,1].set_title("Time at threshold")
    ax[1,2].set_title("Time at HalfMax")
    ax[1,3].set_title("Time at Max")
    ax[2,0].set_title("Mean intensity projection")
    ax[2,1].set_title("IncreaseSPD at HalfMax (Overlay)")
    ax[2,2].set_title("Duration (Overlay)")
    ax[2,3].set_title("Time at HalfMax (Overlay)")

    ## data
    #                0     1     2     3     4     5     6     7     8
    # resultArray = [HMS1, HMS2, Smax, Smin, HMT1, HMT2, HWTi, MaxT, ThrT]
    #
    # from matplotlib.colors import LogNorm
    # heatmap1 = ax[0,0].imshow(resultNpFps[0],cmap=colormap  , norm = LogNorm( vmin = 0, vmax = summaryR.loc[0,"mean+SD"])) 

    ## main plot
    # The scale is aligned with the largest value in the same series.    

    # BG color in plot
    ax[0,0].set_facecolor('#C0C0C0')
    ax[0,1].set_facecolor('#C0C0C0')
    ax[0,2].set_facecolor('#C0C0C0')
    ax[0,3].set_facecolor('#C0C0C0')
    ax[1,0].set_facecolor('#C0C0C0')
    ax[1,1].set_facecolor('#C0C0C0')
    ax[1,2].set_facecolor('#C0C0C0')
    ax[1,3].set_facecolor('#C0C0C0')
    ax[2,0].set_facecolor('#C0C0C0')
    ax[2,1].set_facecolor('#C0C0C0')
    ax[2,2].set_facecolor('#C0C0C0')
    ax[2,3].set_facecolor('#C0C0C0')

    heatmap1  = ax[0,0].imshow(npMask0,cmap=colormap  , vmin = 0, vmax = summaryR.loc[0, "percent"]) 
    heatmap2  = ax[0,1].imshow(npMask1,cmap=colormap_r, vmax = 0, vmin = summaryR.loc[1, "percent"]) # positive-negative reverse
    heatmap3  = ax[0,2].imshow(npMask2,cmap=colormap  , vmin = 0, vmax = summaryR.loc[2, "percent"]) 
    heatmap4  = ax[0,3].imshow(npMask3,cmap=colormap_r, vmax = 0, vmin = summaryR.loc[3, "percent"]) # positive-negative reverse
    heatmap5  = ax[1,0].imshow(npMask6,cmap=colormap  , vmin = 0, vmax = summaryR.loc[6, "percent"]) 
    heatmap6  = ax[1,1].imshow(npMask8,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[8, "percent"]) 
    heatmap7  = ax[1,2].imshow(npMask4,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[4, "percent"]) 
    heatmap8  = ax[1,3].imshow(npMask7,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[7, "percent"]) 
    heatmap9  = ax[2,0].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap10 = ax[2,1].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap11 = ax[2,2].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap12 = ax[2,3].imshow(inputNp[4],cmap="binary") #RAW mean

    ### overlay mask of the threshold
    # binary pixel
    ax[0,0].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[0,1].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[0,2].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[0,3].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,0].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,1].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,2].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,3].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 

    # image overlay
    ax[2,1].imshow(npMask0,cmap=colormap  , vmin = 0, vmax = summaryR.loc[0, "percent"])
    ax[2,2].imshow(npMask6,cmap=colormap  , vmin = 0, vmax = summaryR.loc[6, "percent"]) 
    ax[2,3].imshow(npMask4,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[4, "percent"]) 

    # align colorbar sizes
    divider1 = make_axes_locatable(ax[0,0])
    divider2 = make_axes_locatable(ax[0,1])
    divider3 = make_axes_locatable(ax[0,2])
    divider4 = make_axes_locatable(ax[0,3])
    divider5 = make_axes_locatable(ax[1,0])
    divider6 = make_axes_locatable(ax[1,1])
    divider7 = make_axes_locatable(ax[1,2])
    divider8 = make_axes_locatable(ax[1,3])

    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cax4 = divider4.append_axes("right", size="5%", pad=0.1)
    cax5 = divider5.append_axes("right", size="5%", pad=0.1)
    cax6 = divider6.append_axes("right", size="5%", pad=0.1)
    cax7 = divider7.append_axes("right", size="5%", pad=0.1)
    cax8 = divider8.append_axes("right", size="5%", pad=0.1)

    global cbar1
    global cbar2
    global cbar3
    global cbar4
    global cbar5
    global cbar6
    global cbar7
    global cbar8

    cbar1 = fig.colorbar(heatmap1, cax=cax1)
    cbar2 = fig.colorbar(heatmap2, cax=cax2)
    cbar3 = fig.colorbar(heatmap3, cax=cax3)
    cbar4 = fig.colorbar(heatmap4, cax=cax4)
    cbar5 = fig.colorbar(heatmap5, cax=cax5)
    cbar6 = fig.colorbar(heatmap6, cax=cax6)
    cbar7 = fig.colorbar(heatmap7, cax=cax7)
    cbar8 = fig.colorbar(heatmap8, cax=cax8)

    # axes([left, bottom, width, height])
    ax_a = plt.axes([0.25, 0.09, 0.50, 0.02], facecolor='gold')
    ax_b = plt.axes([0.25, 0.07, 0.50, 0.02], facecolor='gold')
    ax_c = plt.axes([0.25, 0.05, 0.50, 0.02], facecolor='gold')
    ax_d = plt.axes([0.25, 0.03, 0.50, 0.02], facecolor='gold')
    ax_f = plt.axes([0.85, 0.03, 0.1, 0.05], facecolor='green')


    sli_a = RangeSlider(ax_a, "Increase slope percentile", valmin=0, valmax=100, valinit=[0,100], valstep=0.1, dragging=True)
    sli_b = RangeSlider(ax_b, "Decrease slope percentile", valmin=0, valmax=100, valinit=[0,100], valstep=0.1, dragging=True)
    sli_c = RangeSlider(ax_c, "Duration percentile", valmin=0, valmax=100, valinit=[0,100], valstep=0.1, dragging=True)
    sli_d = RangeSlider(ax_d, 'Time percentile', valmin=0, valmax=100, valinit=[0,100], valstep=0.1, dragging=True)
    button = Button(ax_f, "Done")

    def update(val):
 
        sli_A_min, sli_A_max = sli_a.val
        sli_B_min, sli_B_max = sli_b.val
        sli_C_min, sli_C_max = sli_c.val
        sli_D_min, sli_D_max = sli_d.val
        
        global cbar1
        global cbar2
        global cbar3
        global cbar4
        global cbar5
        global cbar6
        global cbar7
        global cbar8

        cbar1.remove()
        cbar2.remove()
        cbar3.remove()
        cbar4.remove()
        cbar5.remove()
        cbar6.remove()
        cbar7.remove()
        cbar8.remove()
              
        for i in range(axRows):
            for j in range(axCols):
                ax[i, j].clear()
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                    
        ax[0,0].set_title("IncreaseSPD at HalfMax")
        ax[0,1].set_title("DecreaseSPD at HalfMax")
        ax[0,2].set_title("MaxSPD")
        ax[0,3].set_title("MinSPD")
        ax[1,0].set_title("Duration")
        ax[1,1].set_title("Time at threshold")
        ax[1,2].set_title("Time at HalfMax")
        ax[1,3].set_title("Time at Max")
        ax[2,0].set_title("Mean intensity projection")
        ax[2,1].set_title("IncreaseSPD at HalfMax (Overlay)")
        ax[2,2].set_title("Duration (Overlay)")
        ax[2,3].set_title("Time at HalfMax (Overlay)")
        
        percent0 = np.nanpercentile(nanPad[0],     sli_A_max)
        percent1 = np.nanpercentile(nanPad[1], 100-sli_B_max)
        percent2 = np.nanpercentile(nanPad[2],     sli_A_max)
        percent3 = np.nanpercentile(nanPad[3], 100-sli_B_max)
        percent4 = np.nanpercentile(nanPad[4],     sli_D_max)
        percent5 = np.nanpercentile(nanPad[5],     sli_D_max)
        percent6 = np.nanpercentile(nanPad[6],     sli_C_max)
        percent7 = np.nanpercentile(nanPad[7],     sli_D_max)
        percent8 = np.nanpercentile(nanPad[8],     sli_D_max)

        lowPercent0 = np.nanpercentile(nanPad[0],     sli_A_min)
        lowPercent1 = np.nanpercentile(nanPad[1],     100-sli_B_min)
        lowPercent2 = np.nanpercentile(nanPad[2],     sli_A_min)
        lowPercent3 = np.nanpercentile(nanPad[3],     100-sli_B_min)
        lowPercent4 = np.nanpercentile(nanPad[4],     sli_D_min)
        lowPercent5 = np.nanpercentile(nanPad[5],     sli_D_min)
        lowPercent6 = np.nanpercentile(nanPad[6],     sli_C_min)
        lowPercent7 = np.nanpercentile(nanPad[7],     sli_D_min)
        lowPercent8 = np.nanpercentile(nanPad[8],     sli_D_min)

        heatmap1  = ax[0,0].imshow(npMask0,    cmap = colormap  , vmin = lowPercent0, vmax = percent0) 
        heatmap2  = ax[0,1].imshow(npMask1,    cmap = colormap_r, vmax = lowPercent1, vmin = percent1) # positive-negative reverse
        heatmap3  = ax[0,2].imshow(npMask2,    cmap = colormap  , vmin = lowPercent2, vmax = percent2) 
        heatmap4  = ax[0,3].imshow(npMask3,    cmap = colormap_r, vmax = lowPercent3, vmin = percent3) # positive-negative reverse
        heatmap5  = ax[1,0].imshow(npMask6,    cmap = colormap  , vmin = lowPercent6, vmax = percent6) 
        heatmap6  = ax[1,1].imshow(npMask8,    cmap = colormap2 , vmin = lowPercent8, vmax = percent8) 
        heatmap7  = ax[1,2].imshow(npMask4,    cmap = colormap2 , vmin = lowPercent4, vmax = percent4) 
        heatmap8  = ax[1,3].imshow(npMask7,    cmap = colormap2 , vmin = lowPercent7, vmax = percent7) 
        heatmap9  = ax[2,0].imshow(inputNp[4], cmap ="binary") #RAW mean
        heatmap10 = ax[2,1].imshow(inputNp[4], cmap ="binary") #RAW mean
        heatmap11 = ax[2,2].imshow(inputNp[4], cmap ="binary") #RAW mean
        heatmap12 = ax[2,3].imshow(inputNp[4], cmap ="binary") #RAW mean

        ### overlay mask of the threshold
        # binary pixel
        ax[0,0].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[0,1].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[0,2].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[0,3].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,0].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,1].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,2].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,3].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 

        # image overlay
        ax[2,1].imshow(npMask0,cmap=colormap  , vmin = lowPercent0, vmax = percent0)
        ax[2,2].imshow(npMask6,cmap=colormap  , vmin = lowPercent6, vmax = percent6) 
        ax[2,3].imshow(npMask4,cmap=colormap2 , vmin = lowPercent4, vmax = percent4) 

        # align colorbar sizes
        divider1 = make_axes_locatable(ax[0,0])
        divider2 = make_axes_locatable(ax[0,1])
        divider3 = make_axes_locatable(ax[0,2])
        divider4 = make_axes_locatable(ax[0,3])
        divider5 = make_axes_locatable(ax[1,0])
        divider6 = make_axes_locatable(ax[1,1])
        divider7 = make_axes_locatable(ax[1,2])
        divider8 = make_axes_locatable(ax[1,3])

        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        cax4 = divider4.append_axes("right", size="5%", pad=0.1)
        cax5 = divider5.append_axes("right", size="5%", pad=0.1)
        cax6 = divider6.append_axes("right", size="5%", pad=0.1)
        cax7 = divider7.append_axes("right", size="5%", pad=0.1)
        cax8 = divider8.append_axes("right", size="5%", pad=0.1)

        cbar1 = fig.colorbar(heatmap1, cax=cax1)
        cbar2 = fig.colorbar(heatmap2, cax=cax2)
        cbar3 = fig.colorbar(heatmap3, cax=cax3)
        cbar4 = fig.colorbar(heatmap4, cax=cax4)
        cbar5 = fig.colorbar(heatmap5, cax=cax5)
        cbar6 = fig.colorbar(heatmap6, cax=cax6)
        cbar7 = fig.colorbar(heatmap7, cax=cax7)
        cbar8 = fig.colorbar(heatmap8, cax=cax8)
        fig.canvas.blit()
       
    def save_and_close_figure():
        summaryOutDir = f"./{imgName}_manualSummary.png"
        fig.savefig(summaryOutDir, format="png", dpi=300)
        plt.close()

    def doneButton(self):
        thread = threading.Thread(target=save_and_close_figure)
        thread.start()

    fig.canvas.mpl_connect("close_event", doneButton)
    button.on_clicked(doneButton)
    
    sli_a.on_changed(update)
    sli_b.on_changed(update)
    sli_c.on_changed(update)
    sli_d.on_changed(update)

    fig.suptitle(f"{imgName}")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.90)
    plt.show()

    
# %%

### summarizeFitting
def IAsummarizeVal(imgPath, rawOrSt, colormap, colormap2, cbarTicks):
    
    os.chdir(imgPath)
    imgName = os.path.basename(imgPath)
    print("Summarizing...")

    ### import
    #                0     1     2     3     4     5     6     7     8
    # resultArray = [HMS1, HMS2, Smax, Smin, HMT1, HMT2, HWTi, MaxT, ThrT]
    if rawOrSt == "raw":
        outNp = f"./resultArray/{imgName}_resultRAWNp.npy"
        resultNpFps = np.load(outNp)
    elif rawOrSt == "st":
        outNp = f"./resultArray/{imgName}_resultStNp.npy"
        resultNpFps = np.load(outNp)        
    else:
        pass        
      
    ### calculte summary for vmax and vmin
    nanPad = np.copy(resultNpFps)
    
    # Mask unnecessary elements in the plot
    npMask0 = np.ma.masked_where(resultNpFps[0] <= 0, resultNpFps[0])
    npMask1 = np.ma.masked_where(resultNpFps[1] >= 0, resultNpFps[1])
    npMask2 = np.ma.masked_where(resultNpFps[2] <= 0, resultNpFps[2])
    npMask3 = np.ma.masked_where(resultNpFps[3] >= 0, resultNpFps[3])
    npMask4 = np.ma.masked_where(resultNpFps[4] <= 0, resultNpFps[4])
    npMask5 = np.ma.masked_where(resultNpFps[5] <= 0, resultNpFps[5])
    npMask6 = np.ma.masked_where(resultNpFps[6] <= 0, resultNpFps[6])
    npMask7 = np.ma.masked_where(resultNpFps[7] <= 0, resultNpFps[7])
    npMask8 = np.ma.masked_where(resultNpFps[8] <= 0, resultNpFps[8])


    # Mask unnecessary elements in the plot
    nanPad[0][nanPad[0] <= 0] = np.NaN
    nanPad[1][nanPad[1] >= 0] = np.NaN
    nanPad[2][nanPad[2] <= 0] = np.NaN
    nanPad[3][nanPad[3] >= 0] = np.NaN
    nanPad[4][nanPad[4] <= 0] = np.NaN
    nanPad[5][nanPad[5] <= 0] = np.NaN
    nanPad[6][nanPad[6] <= 0] = np.NaN
    nanPad[7][nanPad[7] <= 0] = np.NaN
    nanPad[8][nanPad[8] <= 0] = np.NaN



    nanSummary0 = pd.DataFrame(pd.Series(nanPad[0].ravel()).describe()).transpose()
    nanSummary1 = pd.DataFrame(pd.Series(nanPad[1].ravel()).describe()).transpose()
    nanSummary2 = pd.DataFrame(pd.Series(nanPad[2].ravel()).describe()).transpose()
    nanSummary3 = pd.DataFrame(pd.Series(nanPad[3].ravel()).describe()).transpose()
    nanSummary4 = pd.DataFrame(pd.Series(nanPad[4].ravel()).describe()).transpose()
    nanSummary5 = pd.DataFrame(pd.Series(nanPad[5].ravel()).describe()).transpose()
    nanSummary6 = pd.DataFrame(pd.Series(nanPad[6].ravel()).describe()).transpose()
    nanSummary7 = pd.DataFrame(pd.Series(nanPad[7].ravel()).describe()).transpose()
    nanSummary8 = pd.DataFrame(pd.Series(nanPad[8].ravel()).describe()).transpose()

    summary = pd.concat([nanSummary0, nanSummary1,nanSummary2,nanSummary3,nanSummary4,nanSummary5,nanSummary6,nanSummary7, nanSummary8])
    summary["mean+SD"] = summary["mean"] +summary["std"]*2
    summary["mean-SD"] = summary["mean"] -summary["std"]*2
    summaryR = summary.reset_index()

    '''
    SD+-1~3.5
    1.0 SD: 68.3 %
    1.5 SD: 86.6 %
    2.0 SD: 95.4 %
    2.5 SD: 98.8 %
    3.0 SD: 99.7 %
    3.5 SD: 100.0 %
    '''

    ### add percentile result to summary
    percent0 = np.nanpercentile(nanPad[0],     98)
    percent1 = np.nanpercentile(nanPad[1], 100-98)
    percent2 = np.nanpercentile(nanPad[2],     98)
    percent3 = np.nanpercentile(nanPad[3], 100-98)
    percent4 = np.nanpercentile(nanPad[4],     98)
    percent5 = np.nanpercentile(nanPad[5],     98)
    percent6 = np.nanpercentile(nanPad[6],     98)
    percent7 = np.nanpercentile(nanPad[7],     98)
    percent8 = np.nanpercentile(nanPad[8],     98)

    percent = pd.DataFrame({'percent': [percent0, percent1, percent2, percent3, percent4, percent5, percent6, percent7, percent8]})
    summaryR = pd.concat([summaryR, percent], axis=1)

    ### plotSummary
    # the RAW intensity/second corrected by fps is shown as the slope.

    # inputNp
    #0        1     2      3     4
    #npF1RAW, npF1, npMip, npThr npRAWmean
    outNp2 = f"./resultArray/{imgName}_inputNp.npy"
    inputNp = np.load(outNp2)
    npThrmask = np.ma.masked_where(inputNp[3] == 255, inputNp[3])

    # backend
    fig, ax=plt.subplots(3, 4, figsize=(25,15), subplot_kw=({"xticks":(), "yticks":()}))
    axRows, axCols = ax.shape
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()
    canvas = FigureCanvas(fig)

    layout = QtWidgets.QGridLayout()

    label1 = QtWidgets.QLabel('Increase (max): ')
    label2 = QtWidgets.QLabel('Increase (min): ')
    label3 = QtWidgets.QLabel('Decrease (max): ')
    label4 = QtWidgets.QLabel('Decrease (min): ')
    label5 = QtWidgets.QLabel('Duration (max): ')
    label6 = QtWidgets.QLabel('Duration (min): ')
    label7 = QtWidgets.QLabel('Time (max): ')
    label8 = QtWidgets.QLabel('Time (min): ')

    txtInput1 = round(summaryR.loc[0, "percent"], 5)
    txtInput3 = round(summaryR.loc[1, "percent"], 5)
    txtInput5 = round(summaryR.loc[6, "percent"], 5)
    txtInput7 = round(summaryR.loc[4, "percent"], 5)

    text_box1 = QtWidgets.QLineEdit(f"{txtInput1}")
    text_box2 = QtWidgets.QLineEdit("0")
    text_box3 = QtWidgets.QLineEdit(f"{txtInput3}")
    text_box4 = QtWidgets.QLineEdit("0")
    text_box5 = QtWidgets.QLineEdit(f"{txtInput5}")
    text_box6 = QtWidgets.QLineEdit("0")
    text_box7 = QtWidgets.QLineEdit(f"{txtInput7}")
    text_box8 = QtWidgets.QLineEdit("0")

    button9 = QtWidgets.QPushButton('[Export]')

    layout.addWidget(canvas, 0, 0, 1, 9)  # Set rowSpan=1, columnSpan=3 for canvas

    layout.addWidget(label1   , 1, 0)
    layout.addWidget(text_box1, 1, 1)
    layout.addWidget(label2   , 2, 0)
    layout.addWidget(text_box2, 2, 1)
    layout.addWidget(label3   , 1, 2)
    layout.addWidget(text_box3, 1, 3)
    layout.addWidget(label4   , 2, 2)
    layout.addWidget(text_box4, 2, 3)
    layout.addWidget(label5   , 1, 4)
    layout.addWidget(text_box5, 1, 5)
    layout.addWidget(label6   , 2, 4)
    layout.addWidget(text_box6, 2, 5)
    layout.addWidget(label7   , 1, 6)
    layout.addWidget(text_box7, 1, 7)
    layout.addWidget(label8   , 2, 6)
    layout.addWidget(text_box8, 2, 7)

    layout.addWidget(button9  , 1, 8, 2, 1)
    widget.setLayout(layout)
    
    
    colormap_r = f"{colormap}_r"

    ax[0,0].set_title("IncreaseSPD at HalfMax")
    ax[0,1].set_title("DecreaseSPD at HalfMax")
    ax[0,2].set_title("MaxSPD")
    ax[0,3].set_title("MinSPD")
    ax[1,0].set_title("Duration")
    ax[1,1].set_title("Time at threshold")
    ax[1,2].set_title("Time at HalfMax")
    ax[1,3].set_title("Time at Max")
    ax[2,0].set_title("Mean intensity projection")
    ax[2,1].set_title("IncreaseSPD at HalfMax (Overlay)")
    ax[2,2].set_title("Duration (Overlay)")
    ax[2,3].set_title("Time at HalfMax (Overlay)")

    ## data
    #                0     1     2     3     4     5     6     7     8
    # resultArray = [HMS1, HMS2, Smax, Smin, HMT1, HMT2, HWTi, MaxT, ThrT]

    ## main plot
    # The scale is aligned with the largest value in the same series.    

    # BG color in plot
    ax[0,0].set_facecolor('#C0C0C0')
    ax[0,1].set_facecolor('#C0C0C0')
    ax[0,2].set_facecolor('#C0C0C0')
    ax[0,3].set_facecolor('#C0C0C0')
    ax[1,0].set_facecolor('#C0C0C0')
    ax[1,1].set_facecolor('#C0C0C0')
    ax[1,2].set_facecolor('#C0C0C0')
    ax[1,3].set_facecolor('#C0C0C0')
    ax[2,0].set_facecolor('#C0C0C0')
    ax[2,1].set_facecolor('#C0C0C0')
    ax[2,2].set_facecolor('#C0C0C0')
    ax[2,3].set_facecolor('#C0C0C0')

    heatmap1  = ax[0,0].imshow(npMask0,cmap=colormap  , vmin = 0, vmax = summaryR.loc[0, "percent"]) 
    heatmap2  = ax[0,1].imshow(npMask1,cmap=colormap_r, vmax = 0, vmin = summaryR.loc[1, "percent"]) # positive-negative reverse
    heatmap3  = ax[0,2].imshow(npMask2,cmap=colormap  , vmin = 0, vmax = summaryR.loc[0, "percent"]) 
    heatmap4  = ax[0,3].imshow(npMask3,cmap=colormap_r, vmax = 0, vmin = summaryR.loc[1, "percent"]) # posirive-negative reverse
    heatmap5  = ax[1,0].imshow(npMask6,cmap=colormap  , vmin = 0, vmax = summaryR.loc[6, "percent"]) 
    heatmap6  = ax[1,1].imshow(npMask8,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[4, "percent"]) 
    heatmap7  = ax[1,2].imshow(npMask4,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[4, "percent"]) 
    heatmap8  = ax[1,3].imshow(npMask7,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[4, "percent"]) 
    heatmap9  = ax[2,0].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap10 = ax[2,1].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap11 = ax[2,2].imshow(inputNp[4],cmap="binary") #RAW mean
    heatmap12 = ax[2,3].imshow(inputNp[4],cmap="binary") #RAW mean

    ### overlay mask of the threshold
    # binary pixel
    ax[0,0].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[0,1].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[0,2].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[0,3].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,0].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,1].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,2].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 
    ax[1,3].imshow(npThrmask, cmap = "gray", vmin = -10, vmax = 25) 

    # image overlay
    ax[2,1].imshow(npMask0,cmap=colormap  , vmin = 0, vmax = summaryR.loc[0, "percent"])
    ax[2,2].imshow(npMask6,cmap=colormap  , vmin = 0, vmax = summaryR.loc[6, "percent"]) 
    ax[2,3].imshow(npMask4,cmap=colormap2 , vmin = 0, vmax = summaryR.loc[4, "percent"]) 


    increaseMax = round(summaryR.loc[0, "percent"], 5)
    decreaseMax = round(summaryR.loc[1, "percent"], 5)
    durationMax = round(summaryR.loc[6, "percent"], 5)
    thrtimeMax  = round(summaryR.loc[4, "percent"], 5)
    
    # find limit value from log
    global text1
    global text2
    global text3
    global text4
    
    logText1 = f"IncreaseSlope(max): {increaseMax}\nIncreaseSlope(min) : 0"
    logText2 = f"Decrease(max): {decreaseMax}\nDecrease(min) : 0"
    logText3 = f"Duration(max): {durationMax}\nDuration(min) : 0"
    logText4 = f"ThrTime(max): {thrtimeMax}\nThrTime(min) : 0"
    text1 = fig.text(0.05, 0.05, logText1, ha='left')
    text2 = fig.text(0.30, 0.05, logText2, ha='left')
    text3 = fig.text(0.55, 0.05, logText3, ha='left')
    text4 = fig.text(0.80, 0.05, logText4, ha='left')

    if rawOrSt == "raw":
        textRawSt = "RAW array"
    elif rawOrSt == "st":
        textRawSt = "Standardized array"
    else:
        textRawSt = "nothing"
    
    fig.text(0.90, 0.02, f"Dataset: {textRawSt}", ha='right')


    # align colorbar sizes
    divider1 = make_axes_locatable(ax[0,0])
    divider2 = make_axes_locatable(ax[0,1])
    divider3 = make_axes_locatable(ax[0,2])
    divider4 = make_axes_locatable(ax[0,3])
    divider5 = make_axes_locatable(ax[1,0])
    divider6 = make_axes_locatable(ax[1,1])
    divider7 = make_axes_locatable(ax[1,2])
    divider8 = make_axes_locatable(ax[1,3])

    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cax4 = divider4.append_axes("right", size="5%", pad=0.1)
    cax5 = divider5.append_axes("right", size="5%", pad=0.1)
    cax6 = divider6.append_axes("right", size="5%", pad=0.1)
    cax7 = divider7.append_axes("right", size="5%", pad=0.1)
    cax8 = divider8.append_axes("right", size="5%", pad=0.1)

    global cbar1
    global cbar2
    global cbar3
    global cbar4
    global cbar5
    global cbar6
    global cbar7
    global cbar8

    cbar1 = fig.colorbar(heatmap1, cax=cax1, pad = 10)
    cbar2 = fig.colorbar(heatmap2, cax=cax2, pad = 10)
    cbar3 = fig.colorbar(heatmap3, cax=cax3, pad = 10)
    cbar4 = fig.colorbar(heatmap4, cax=cax4, pad = 10)
    cbar5 = fig.colorbar(heatmap5, cax=cax5, pad = 10)
    cbar6 = fig.colorbar(heatmap6, cax=cax6, pad = 10)
    cbar7 = fig.colorbar(heatmap7, cax=cax7, pad = 10)
    cbar8 = fig.colorbar(heatmap8, cax=cax8, pad = 10)
    
    if cbarTicks == False:
        cbar1.ax.yaxis.set_ticks([])
        cbar1.ax.yaxis.set_ticklabels([])
        cbar2.ax.yaxis.set_ticks([])
        cbar2.ax.yaxis.set_ticklabels([])
        cbar3.ax.yaxis.set_ticks([])
        cbar3.ax.yaxis.set_ticklabels([])
        cbar4.ax.yaxis.set_ticks([])
        cbar4.ax.yaxis.set_ticklabels([])
        cbar5.ax.yaxis.set_ticks([])
        cbar5.ax.yaxis.set_ticklabels([])
        cbar6.ax.yaxis.set_ticks([])
        cbar6.ax.yaxis.set_ticklabels([])
        cbar7.ax.yaxis.set_ticks([])
        cbar7.ax.yaxis.set_ticklabels([])
        cbar8.ax.yaxis.set_ticks([])
        cbar8.ax.yaxis.set_ticklabels([])
 
    
 
    def update(val):
      
        a = float(text_box1.text()) #increase max
        b = float(text_box2.text()) #increase min
        c = float(text_box3.text()) #decrease max
        d = float(text_box4.text()) #decrease min
        e = float(text_box5.text()) #duration max
        f = float(text_box6.text()) #duration min
        g = float(text_box7.text()) #thrtime max
        h = float(text_box8.text()) #thrtime min
               
        global cbar1
        global cbar2
        global cbar3
        global cbar4
        global cbar5
        global cbar6
        global cbar7
        global cbar8
        global text1
        global text2
        global text3
        global text4

        cbar1.remove()
        cbar2.remove()
        cbar3.remove()
        cbar4.remove()
        cbar5.remove()
        cbar6.remove()
        cbar7.remove()
        cbar8.remove()
        text1.remove()
        text2.remove()
        text3.remove()
        text4.remove()


        logText1 = f"IncreaseSlope(max): {a}\nIncreaseSlope(min) : {b}"
        logText2 = f"Decrease(max): {c}\nDecrease(min) : {d}"
        logText3 = f"Duration(max): {e}\nDuration(min) : {f}"
        logText4 = f"ThrTime(max): {g}\nThrTime(min) : {h}"
        text1 = fig.text(0.05, 0.05, logText1, ha='left')
        text2 = fig.text(0.30, 0.05, logText2, ha='left')
        text3 = fig.text(0.55, 0.05, logText3, ha='left')
        text4 = fig.text(0.80, 0.05, logText4, ha='left')

              
        for i in range(axRows):
            for j in range(axCols):
                ax[i, j].clear()
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                    
        ax[0,0].set_title("IncreaseSPD at HalfMax")
        ax[0,1].set_title("DecreaseSPD at HalfMax")
        ax[0,2].set_title("MaxSPD")
        ax[0,3].set_title("MinSPD")
        ax[1,0].set_title("Duration")
        ax[1,1].set_title("Time at threshold")
        ax[1,2].set_title("Time at HalfMax")
        ax[1,3].set_title("Time at Max")
        ax[2,0].set_title("Mean intensity projection")
        ax[2,1].set_title("IncreaseSPD at HalfMax (Overlay)")
        ax[2,2].set_title("Duration (Overlay)")
        ax[2,3].set_title("Time at HalfMax (Overlay)")

        heatmap1  = ax[0,0].imshow(npMask0,    cmap = colormap  , vmin = b, vmax = a) 
        heatmap2  = ax[0,1].imshow(npMask1,    cmap = colormap_r, vmax = d, vmin = c) # positive-negative reverse
        heatmap3  = ax[0,2].imshow(npMask2,    cmap = colormap  , vmin = b, vmax = a) 
        heatmap4  = ax[0,3].imshow(npMask3,    cmap = colormap_r, vmax = d, vmin = c) # positive-negative reverse
        heatmap5  = ax[1,0].imshow(npMask6,    cmap = colormap  , vmin = f, vmax = e) 
        heatmap6  = ax[1,1].imshow(npMask8,    cmap = colormap2 , vmin = h, vmax = g) 
        heatmap7  = ax[1,2].imshow(npMask4,    cmap = colormap2 , vmin = h, vmax = g) 
        heatmap8  = ax[1,3].imshow(npMask7,    cmap = colormap2 , vmin = h, vmax = g) 
        heatmap9  = ax[2,0].imshow(inputNp[4], cmap ="binary") #RAW mean
        heatmap10 = ax[2,1].imshow(inputNp[4], cmap ="binary") #RAW mean
        heatmap11 = ax[2,2].imshow(inputNp[4], cmap ="binary") #RAW mean
        heatmap12 = ax[2,3].imshow(inputNp[4], cmap ="binary") #RAW mean

        ### overlay mask of the threshold
        # binary pixel
        ax[0,0].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[0,1].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[0,2].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[0,3].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,0].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,1].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,2].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 
        ax[1,3].imshow(npThrmask, cmap ="gray", vmin = -10, vmax = 25) 

        # image overlay
        ax[2,1].imshow(npMask0,cmap=colormap  , vmin = b, vmax = a)
        ax[2,2].imshow(npMask6,cmap=colormap  , vmin = f, vmax = e) 
        ax[2,3].imshow(npMask4,cmap=colormap2 , vmin = h, vmax = g) 

        # align colorbar sizes
        divider1 = make_axes_locatable(ax[0,0])
        divider2 = make_axes_locatable(ax[0,1])
        divider3 = make_axes_locatable(ax[0,2])
        divider4 = make_axes_locatable(ax[0,3])
        divider5 = make_axes_locatable(ax[1,0])
        divider6 = make_axes_locatable(ax[1,1])
        divider7 = make_axes_locatable(ax[1,2])
        divider8 = make_axes_locatable(ax[1,3])

        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        cax4 = divider4.append_axes("right", size="5%", pad=0.1)
        cax5 = divider5.append_axes("right", size="5%", pad=0.1)
        cax6 = divider6.append_axes("right", size="5%", pad=0.1)
        cax7 = divider7.append_axes("right", size="5%", pad=0.1)
        cax8 = divider8.append_axes("right", size="5%", pad=0.1)

        cbar1 = fig.colorbar(heatmap1, cax=cax1, pad = 0.1)
        cbar2 = fig.colorbar(heatmap2, cax=cax2, pad = 0.1)
        cbar3 = fig.colorbar(heatmap3, cax=cax3, pad = 0.1)
        cbar4 = fig.colorbar(heatmap4, cax=cax4, pad = 0.1)
        cbar5 = fig.colorbar(heatmap5, cax=cax5, pad = 0.1)
        cbar6 = fig.colorbar(heatmap6, cax=cax6, pad = 0.1)
        cbar7 = fig.colorbar(heatmap7, cax=cax7, pad = 0.1)
        cbar8 = fig.colorbar(heatmap8, cax=cax8, pad = 0.1)
        
        if cbarTicks == False:
            cbar1.ax.yaxis.set_ticks([])
            cbar1.ax.yaxis.set_ticklabels([])
            cbar2.ax.yaxis.set_ticks([])
            cbar2.ax.yaxis.set_ticklabels([])
            cbar3.ax.yaxis.set_ticks([])
            cbar3.ax.yaxis.set_ticklabels([])
            cbar4.ax.yaxis.set_ticks([])
            cbar4.ax.yaxis.set_ticklabels([])
            cbar5.ax.yaxis.set_ticks([])
            cbar5.ax.yaxis.set_ticklabels([])
            cbar6.ax.yaxis.set_ticks([])
            cbar6.ax.yaxis.set_ticklabels([])
            cbar7.ax.yaxis.set_ticks([])
            cbar7.ax.yaxis.set_ticklabels([])
            cbar8.ax.yaxis.set_ticks([])
            cbar8.ax.yaxis.set_ticklabels([])
        
        fig.canvas.draw_idle() 
    
    
    def save_and_close_figure():
        summaryOutDir = f"./{imgName}_manualSummary.png"
        fig.savefig(summaryOutDir, format="png", dpi=300)
        plt.close()
        widget.close()


    def doneButton(self):
        save_and_close_figure()

    fig.canvas.mpl_connect("close_event", doneButton)
    button9.clicked.connect(doneButton)

    text_box1.textChanged.connect(update)
    text_box2.textChanged.connect(update)
    text_box3.textChanged.connect(update)
    text_box4.textChanged.connect(update)
    text_box5.textChanged.connect(update)
    text_box6.textChanged.connect(update)
    text_box7.textChanged.connect(update)
    text_box8.textChanged.connect(update)

    fig.suptitle(f"{imgName}")
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.90)
    
    widget.show()
    
    try:
        sys.exit(app.exec_())
    except:
        pass
    
    

# %%
if __name__ == '__main__':
    IAsummarizePer()
    IAsummarizeVal()
    autoSummarize()