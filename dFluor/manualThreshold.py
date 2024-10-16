# %%
import os 
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from matplotlib.widgets import Slider, Button


# %%
def thresholding(imgPath,img):    
    imgName, ext = os.path.splitext(os.path.basename(imgPath))     

    npRAWmean = np.mean(img, axis=0)
    npF1RAW= img[0]
   
    blurStack = []
    for imgSlice in img:
        imgSlice = imgSlice.astype("float64")
        blur = cv2.GaussianBlur(imgSlice, (3, 3), sigmaX=1, sigmaY=1)
        blurStack.append(blur)
   
    
    npF1 = blurStack[0]
    npStack = np.asarray(blurStack) #T, Y(ROW), X(COL)
    npMip = np.amax(blurStack, axis=0)

    imgDim = npStack.shape
    frame = imgDim[0]
    maxRow = imgDim[1]
    maxCol = imgDim[2]

    global ret
    npF1RawMed = cv2.medianBlur(npF1RAW, 5)
    ret, npBin = cv2.threshold(npF1RawMed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   # Get Otsu value

    ### DilateAndErode
    kernel = np.ones((3,3),np.uint8)
    npThr = np.copy(npBin)
    npThr = cv2.dilate(npThr, kernel, iterations = 2)
    npThr =  cv2.erode(npThr, kernel, iterations = 2)
    npThrMask = np.ma.masked_where(npThr == 255, npThr)
    
    npThrBool = np.array(npThr, dtype=bool) #larger than BG as True
    npThrBg = np.logical_not(npThrBool) #reverse bool

    npBg = npStack[:,npThrBg] #bg timecourse[timeseries,pixels]
    npBg[npBg == 0] = np.NaN #0 to nan
    bgSummary = pd.DataFrame(pd.Series(npBg.ravel()).describe()).transpose() #Summarize
    bgStd = bgSummary.loc[0,"std"]       #SD of BG
    BgPlusStd = npF1 + bgStd*2 # BG+2SD
    bgGreaterMip = np.greater_equal(BgPlusStd,npMip) #True if BgPlusStd >= MIP

    bgSdInt = np.array(bgGreaterMip, dtype = "int8")
    bgSdInt255 = np.where(bgSdInt == 1, 255, 0)


    ### plot Figure
    fig, ax=plt.subplots(2, 1, figsize=(5,5), subplot_kw=({"xticks":(), "yticks":()}))

    font_size = int(fig.get_size_inches()[0] * 2)
    ax[0].set_title("BackgroundArea", fontsize=font_size)
    ax[1].set_title("Overlay", fontsize=font_size)

    ax[0].imshow(npThr, cmap = "gray", vmin = -10, vmax = 25)
    ax[1].imshow(npRAWmean,cmap="binary")
    ax[1].imshow(npThrMask, cmap = "gray", vmin = -10, vmax = 25) 

    # axes([Left, Bottom, Width, Height])
    ax_a = plt.axes([0.80, 0.15, 0.05, 0.6], facecolor='gold')
    ax_b = plt.axes([0.90, 0.15, 0.05, 0.6], facecolor='gold')
    ax_c = plt.axes([0.85, 0.03, 0.1, 0.05], facecolor='green')
    sli_a = Slider(ax_a, 'Back\nground',    0, 5.00, valinit=1.00, valstep=0.01, orientation='vertical')
    sli_b = Slider(ax_b, 'SD',    0, 4.00, valinit=2.00, valstep=0.25, orientation='vertical')
    button_c = Button(ax_c, "Done")
  
    global thresholdRatio 
    global sdMagni
    thresholdRatio = 1
    sdMagni = 2
      
    def update(val):
        global thresholdRatio 
        global sdMagni
        global ret

        thresholdRatio = sli_a.val
        sdMagni = sli_b.val

        ret, npBin = cv2.threshold(npF1RawMed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   # Get Otsu value
        ret, npBin = cv2.threshold(npF1RawMed,ret*thresholdRatio ,255,cv2.THRESH_BINARY) # Threshold = OtsuValue*thresholdRatio

        ### DilateAndErode
        kernel = np.ones((3,3),np.uint8)
        npThr = np.copy(npBin)
        npThr = cv2.dilate(npThr, kernel, iterations = 2)
        npThr =  cv2.erode(npThr, kernel, iterations = 2)

        npThrMask = np.ma.masked_where(npThr == 255, npThr)

        ### plot Figure
        ax[0].imshow(npThr, cmap = "gray", vmin = -10, vmax = 25)
        ax[1].imshow(npRAWmean,cmap="binary")
        ax[1].imshow(npThrMask, cmap = "gray", vmin = -10, vmax = 25) 

        fig.canvas.blit()
        
    def update_font_size(event):
        font_size = int(fig.get_size_inches()[0] * 2)
        ax[0].set_title("BackgroundArea", fontsize=font_size)
        ax[1].set_title("OverThresholdArea(BG+SD)", fontsize=font_size)

    fig.canvas.mpl_connect('resize_event', update_font_size)


    sli_a.on_changed(update)
    sli_b.on_changed(update)

    def doneButton(self):
        plt.close()

    fig.canvas.mpl_connect("close_event", doneButton)
    button_c.on_clicked(doneButton)


    fig.suptitle(f"{imgName}")
    plt.subplots_adjust(left=0.05, right=0.7, bottom=0.1, top=0.9)
    plt.show()

    return imgName, ret, sdMagni
