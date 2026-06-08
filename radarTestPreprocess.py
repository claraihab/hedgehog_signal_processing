'''
    test the preprocessing of radarLib
'''

import radarLib

from radarLib import *
from radarTools import *
from radarTestDefs import *

maxFrame    = None
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
    np.save(dataPath + "/radar_phase.npy", radarPhaseData)



def runDistCalc(dataPath:str):
    radarPhaseData = np.load(dataPath + "/radar_phase.npy")
    radarConfig = importRadarConfig(dataPath)

    # convert phaseData to distance-data
    dataShape = radarPhaseData.shape
    frameCount = dataShape[0]

    # create empty results array to speed up later work
    distanceData = np.zeros([frameCount-1,4], dtype=float)
    distanceData[:,0] = radarPhaseData[:-1,0]   # copy timestamps from phase data
    # calculate the distance data from the phase data
    for col in [1,2,3]:
        distanceCol = calcDisplacement(radarConfig, radarPhaseData[:,col])
        distanceData[:, col] = distanceCol - np.mean(distanceCol)

    np.save(dataPath + "/radar_distance.npy", distanceData)

    # alternate calc of distances :
    # create empty results array to speed up later work
    distanceData = np.zeros([frameCount,4], dtype=float)
    distanceData[:,0] = radarPhaseData[:,0]   # copy timestamps from phase data
    # calculate the distance data from the phase data
    for col in [1,2,3]:
        distanceCol = calcDisplacementAlt(radarConfig, radarPhaseData[:,col])
        distanceData[:, col] = distanceCol - np.mean(distanceCol)

    np.save(dataPath + "/radar_distance_alt.npy", distanceData)



if __name__ == '__main__':
    radarLib.radarLib_verbose    = True     # radarLib should print propgress info


    for i in  range(1,len(radarFilePathes)):
        dataPath = radarFilePathes[i]
        distance = radarDistances[i]
        runPreprocess(dataPath, distance)
        runDistCalc(dataPath)
