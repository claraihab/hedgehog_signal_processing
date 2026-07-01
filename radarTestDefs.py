
# filepath for the test-files
#radarFilePathes = [
       # "D:/Projekte/Igel-Box/Radar/mit_hund/RadarIfxBGT60TR13C_00",
        #"D:/Projekte/Igel-Box/Radar/mit_Igel/recording_2026_02_26_11_42_11",
        #"D:/Projekte/Igel-Box/Radar/Igelstation - 25.5.2026/BGT60TR13C_record_20260522-164318/RadarIfxAvian_00",
        #"D:/Projekte/Igel-Box/Radar/Igelstation - 25.5.2026/BGT60TR13C_record_20260522-164716/RadarIfxAvian_00",
        #"D:/Projekte/Igel-Box/Radar/Igelstation - 25.5.2026/BGT60TR13C_record_20260522-165559/RadarIfxAvian_00",
        #"D:/Projekte/Igel-Box/Radar/Igelstation - 25.5.2026/BGT60TR13C_record_20260522-170049/RadarIfxAvian_00",
        #"D:/Projekte/Igel-Box/Radar/Igelstation - 25.5.2026/BGT60TR13C_record_20260522-170600/RadarIfxAvian_00",
           #         ]
#radarDistances  = [ 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40]






from pathlib import Path


base = Path("D:/igel/Radar-Test")

radarFilePaths1 = sorted(
    str(p)
    for p in base.glob("recording_*/RadarIfxBGT60TR13C_*")
)

radarDistances = [0.40] * len(radarFilePaths1)



base2 = Path("D:/Igelstation")

radarFilePaths2 = sorted(
    str(p)
    for p in base2.glob("*/RadarIfxAvian_00")
)

radarDistances2 = [0.40] * len(radarFilePaths2)




base3 = Path("D:/Radar-Test-mit-hund")

radarFilePaths3 = sorted(
    str(p)
    for p in base3.glob("BGT60TR13C_record_20260415-144425/RadarIfxAvian_00")
)

radarDistances3 = [0.40] * len(radarFilePaths3)




