from pathlib import Path

#hibernating hedgehogs

base = Path("D:/igel/Radar-Test")
radarFilePaths1 = sorted(
    str(p)
    for p in base.glob("recording_*/RadarIfxBGT60TR13C_*")
) 
radarDistances1 = [0.49] * len(radarFilePaths1) 


#awake hedgehogs
base2 = Path("D:/Igelstation")
radarFilePaths2 = sorted(
    str(p)
    for p in base2.glob("*/RadarIfxAvian_00")
)
radarDistances2 = [0.38, 0.38, 0.27, 0.33, 0.46, 0.38, 0.38, 0.38, 0.38, 0.38] 


# dog 
base3 = Path("D:/Radar-Test-mit-hund")
radarFilePaths3 = sorted(
    str(p)
    for p in base3.glob("BGT60TR13C_record_20260415-144425/RadarIfxAvian_00")
)
radarDistances3 = [0.30] * len(radarFilePaths3)


radarFilePaths4 = ["D:/igel/Radar-Test/recording_2026_02_18_10_17_15/RadarIfxBGT60TR13C_00",
                   "D:/igel/Radar-Test/recording_2026_02_26_11_42_11/RadarIfxBGT60TR13C_00"]

radarDistances4 = [0.49, 0.49] 


