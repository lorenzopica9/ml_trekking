import gpxpy
import pandas as pd
import os
import math

def compute_distance(lat1, lon1, lat2, lon2):
    R = 6378.137; #Radius of earth in KM
    dLat = lat2 * math.pi / 180. - lat1 * math.pi / 180.
    dLon = lon2 * math.pi / 180. - lon1 * math.pi / 180.
    a = math.sin(dLat/2) * math.sin(dLat/2) +\
        math.cos(lat1 * math.pi / 180.) * math.cos(lat2 * math.pi / 180.) *\
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c

    return d * 1000; #meters


# # Load gpx.
# gpx_path = 'gpx_apuane/anello-della-pania-della-croce.gpx'
# with open(gpx_path) as f:
#     gpx = gpxpy.parse(f)

gpx_list = []
gpx_path = "./gpx_apuane/"
filelist = os.listdir(gpx_path)
for i in filelist:
    if i.endswith(".gpx"):
        with open(gpx_path + i, 'r') as f:
            #gpx = gpxpy.parse(f)
            gpx_list.append(gpxpy.parse(f))

#print(gpx_list[0].tracks[0].segments[0].points[0])

# Convert to a dataframe one point at a time.
trails = []

for gpx_i in gpx_list:
    points = []
    for segment in gpx_i.tracks[0].segments:
        for p in segment.points:
            points.append({
                'time': p.time,
                'latitude': p.latitude,
                'longitude': p.longitude,
                'elevation': p.elevation,
            })
    df = pd.DataFrame.from_records(points)

    #extract info about trail from datapoints:
    #time
    total_time = df['time'][df.index[-1]] - df['time'][df.index[0]]
    # length
    lat1 = lon1 = lat2 = lon2 = 0.
    length = 0.

    for p in range(0, len(df.index)-1):
        lat1 = df['latitude'][p]
        lon1 = df['longitude'][p]
        lat2 = df['latitude'][p+1]
        lon2 = df['longitude'][p+1]

        length += compute_distance(lat1, lon1, lat2, lon2)

    #uphill, descent
    up = down = 0.
    for p in range(0, len(df.index)-1):
        elev1 = df['elevation'][p]
        elev2 = df['elevation'][p+1]

        diff = elev2 - elev1

        if diff >=0:
            up += diff
        else:
            down += diff

    trails.append({
        'time': total_time,
        'length': length,
        'uphill': up,
        'downhill': -1*down
    })

    df_trails = pd.DataFrame.from_records(trails)


print(df_trails)

df.to_csv("./gpx_apuane/trails.csv", sep='\t')