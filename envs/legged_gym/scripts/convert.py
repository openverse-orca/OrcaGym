import numpy as np

# real action space
hipXmin = -24
hipXmax = 24
hipYmin = -200
hipYmax = 20
kneemin = 34.5
kneemax = 156

degree2radian = np.pi / 180

for i in [hipXmin, hipXmax, hipYmin, hipYmax, kneemin, kneemax]:
    print(i * degree2radian)