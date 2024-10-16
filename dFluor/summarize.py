#!/usr/bin/env python
# coding: utf-8

import os
import glob

from dFluor import plotSummary
from dFluor import movieExport

def auto(pathDir, rawOrSt = "raw", generateMovie = True, speed = 1.0):   
    '''
    pathDir = "/mnt/c/users/SDH/Desktop/OUT/out_241014_14.38.17___TestRun" # get specified folder
    rawOrSt = "raw" # Choose whether to use raw or standardized data in the summarized plot. Enter 'raw' or 'st'. 
                    # The difference between raw and st is only whether the diff values are divided by the background frame.
    generateMovie = True                 # MovieExport setting
    speed = 1                             # Playback speed of movie
    '''

    os.chdir(pathDir)
    # Delete mp4 files recursively (ffmpeg can't combine the files when the speed changes)
    for mp4Path in glob.glob(f"{pathDir}/*/*.mp4", recursive=True):   # This doesn't work when there are [] in the path.
        if os.path.isfile(mp4Path):
            os.remove(mp4Path)
        
    preproList = plotSummary.globWoBlace(f"{pathDir}/IN")

    ### Movie export1
    if generateMovie == True:
        for moviePath in preproList:
            movieExport.tif_to_mp4(moviePath, speed)

    imgList = plotSummary.globOutWoBlace(pathDir)
    for count, imgPath in enumerate(imgList, 1):
        imgName = os.path.basename(imgPath)
        print(f"file: {count}/{len(imgList)}")
        
        ### autoSummarize
        #variables: path, rawOrSt, cmap1, cmap2, slopePercentileHigh, slopePercentileLow, durationPercentile, timePercentileHigh, timePercentileLow
        plotSummary.autoSummarize(imgPath, rawOrSt, "turbo", "tab20c_r", 99, 0, 99, 99, 0)

        ### Movie export2
        if generateMovie == True:
            movieExport.mp4Summary(imgPath, speed)
    print("Done")


def manual(pathDir, imgPathNum, rawOrSt = "raw", cbarTicks = True):
    '''
    pathDir = "/mnt/c/users/SDH/Desktop/OUT/out_241014_14.38.17___TestRun" # get specified folder
    imgPathNum = 1 # Number of the file to be processed in the specified folder
    rawOrSt = "raw" # Choose whether to use raw or standardized data in the summarized plot. Enter 'raw' or 'st'. 
                    # The difference between raw and st is only whether the diff values are divided by the background frame.
    cbarTicks = False # cbar ticks and labels
    '''
    os.chdir(pathDir)

    ### Specify Image
    imgList = sorted(plotSummary.globOutWoBlace(pathDir))
    imgPath = imgList[imgPathNum]

    #plotSummary.IAsummarizePer(imgPath,"turbo", "tab20c_r")
    plotSummary.IAsummarizeVal(imgPath, rawOrSt, "turbo", "tab20c_r", cbarTicks)
    print("Done")





