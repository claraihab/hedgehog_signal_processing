'''
test the preprocessing of radarLib
'''

from radarLib import *
from radarTools import *
from radarTestDefs import *

maxFrame = None


def runPreprocess(dataPath:str, distance:float, altCalculation:bool=False):
    '''
    run preProcessing for one radar recording and create one file with phase vaues and one with the norm distance values
    :param dataPath:
    :param distance:
    :param altCalculation: True -> use alternate calculation for signal phase
    :return:
    '''

    # preprocess radar data: get phase values for the distance to target (+/-1)
    radarPhaseData = preProcessRadarData(dataPath=dataPath,
                                         distance=distance,
                                         frameMax=maxFrame,
                                         altCalculation=altCalculation)
    if altCalculation:
        np.save(dataPath + "/radar_phase2.npy", radarPhaseData)
    else:
        np.save(dataPath + "/radar_phase.npy", radarPhaseData)



def runDistCalc(dataPath:str, altCalculation:bool=False):
    radarConfig = importRadarConfig(dataPath)

    if altCalculation:
        radarPhaseData = np.load(dataPath + "/radar_phase2.npy")
    else:
        radarPhaseData = np.load(dataPath + "/radar_phase.npy")

    # convert phaseData to distance-data
    dataShape = radarPhaseData.shape #??????
    frameCount = dataShape[0]

    # create empty results array to speed up later work
    distanceData = np.zeros([frameCount-1,4], dtype=float) 
    distanceData[:,0] = radarPhaseData[:-1,0]   # copy timestamps from phase data
    # calculate the distance data from the phase data
    for col in [1,2,3]:
        distanceCol = calcDisplacement(radarConfig, radarPhaseData[:,col])
        distanceData[:, col] = distanceCol - np.mean(distanceCol)

    if altCalculation:
        np.save(dataPath + "/radar_distance2.npy", distanceData)
    else:
        np.save(dataPath + "/radar_distance.npy", distanceData)

    # alternate calc of distances :
    # create empty results array to speed up later work
    distanceData = np.zeros([frameCount,4], dtype=float)
    distanceData[:,0] = radarPhaseData[:,0]   # copy timestamps from phase data
    # calculate the distance data from the phase data
    for col in [1,2,3]:
        distanceCol = calcDisplacementAlt(radarConfig, radarPhaseData[:,col])
        distanceData[:, col] = distanceCol - np.mean(distanceCol)

    if altCalculation:
        np.save(dataPath + "/radar_distance2_alt.npy", distanceData)
    else:
        np.save(dataPath + "/radar_distance_alt.npy", distanceData)




## PRE-PROCESSING STARTS HERE

#processing for dog 
# if __name__ == '__main__':
#     radarLib.radarLib_verbose    = True  

#     dataPath = radarFilePaths3[0]
#     distance = radarDistances3[0]
#     runPreprocess(dataPath, distance, altCalculation=False)
#     runDistCalc(dataPath, altCalculation=False)
    
#     runPreprocess(dataPath, distance, altCalculation=True)
#     runDistCalc(dataPath, altCalculation=True)


#     '''
#     for i in  range(1,len(radarFilePaths)):
#         dataPath = radarFilePaths[i]
#         distance = radarDistances[i]
#         runPreprocess(dataPath, distance)
#         runDistCalc(dataPath)
#     '''



#processing for awake hedgehogs

# for file_path, distance in zip(radarFilePaths2, radarDistances2):
    
#     print(f"Processing hedgehog data at: {file_path}")
#     print(f"Using radar distance: {distance}")
#     runPreprocess(file_path, distance, altCalculation=False)
#     runDistCalc(file_path, altCalculation=False)
    
#     runPreprocess(file_path, distance, altCalculation=True)
#     runDistCalc(file_path, altCalculation=True)
    


# loop for hibernating + arousal hedgehogs 
for dataPath, distance in zip(radarFilePaths1, radarDistances1): 
    print(f"\nProcessing file: {dataPath}")  
        
# Run standard calculation (False)
    try: 
        runPreprocess(dataPath, distance, altCalculation=False)
        runDistCalc(dataPath, altCalculation=False)
    except: 
        print("Error in dir:", dataPath)
    
# Run alternative calculation (True)
    try:
        runPreprocess(dataPath, distance, altCalculation=True)
        runDistCalc(dataPath, altCalculation=True)
    except: 
        print("Error in dir:", dataPath)


