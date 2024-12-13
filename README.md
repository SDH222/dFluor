# dFluor

dFluor is a python package for calculate differentiation value from fluorescence time course imaging data  

# Install
```
conda create -n newEnv python=3.9
conda activate newEnv
git clone --recursive https://github.com/SDH222/dFluor
pip install -r ./dFluor/requirements.txt
pip install ./dFluor
```
<br>  

# Usage
Path for input directory should contain folder titled "IN" with input .nd2 or .tif files.  
like
```
/mnt/c/users/SDH/Desktop/IN/sample.tif
```
<br>  

Differential values could be calculated as follows.  
```
import dFluor
inDir = "/mnt/c/users/SDH/Desktop/" # Path for directory should contain folder titled "IN" with input .nd2 or .tif files. This package can process multiple .nd2/.tif files.

dFluor.main.dif(pathDir=inDir)

'''default arguments in dFluor.main.dif()
taskName       = "task"           # Appended to the output folder name
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
```
<br>  

Calculated differential values could be summarized as follows.
```
import dFluor

outDir = "/mnt/c/users/SDH/Desktop/OUT/out_241015_15.07.39___" # path for output directory calculated by dFluor.main.dif()

dFluor.summarize.auto(pathDir=outDir)
dFluor.summarize.manual(pathDir=outDir, imgPathNum=0) # imgPathNum specifies which file to summarize in the folder using a zero-based index


'''Default arguments in dFluor.summarize.auto()
pathDir = "/mnt/c/users/SDH/Desktop/OUT/out_241014_14.38.17___TestRun"  # get specified folder
rawOrSt = "raw"                                                         # Choose whether to use raw or standardized data in the summarized plot. Enter 'raw' or 'st'. 
                                                                        # The difference between raw and st is only whether the diff values are divided by the background frame.
generateMovie = True                                                    # MovieExport setting
speed = 1                                                               # Playback speed of movie
'''

'''Default arguments in dFluor.summarize.manual()
pathDir = "/mnt/c/users/SDH/Desktop/OUT/out_241014_14.38.17___TestRun"  # get specified folder
imgPathNum = 0                                                          # Number of the file to be processed in the specified folder
rawOrSt = "raw"                                                         # Choose whether to use raw or standardized data in the summarized plot. Enter 'raw' or 'st'. 
                                                                        # The difference between raw and st is only whether the diff values are divided by the background frame.
cbarTicks = False                                                       # cbar ticks and labels
'''
```
