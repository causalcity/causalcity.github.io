#########################################################################################
# Logging script to run a scenario, save frames and log car positions.
#########################################################################################

import sys
import airsim
import logging
import argparse
import pandas as pd
import numpy as np
import asyncio
import time
import os 
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
import json
import pycocotools._mask as _mask
from scipy import ndimage

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--output', type=str, default='')
parser.add_argument('--scenario', type=str, default='Scenario_1')
parser.add_argument('--scenarioRange', nargs='+')
parser.add_argument('--dataDir', type=str)
args = parser.parse_args()

print(args.scenarioRange)
print(args.dataDir)

## For Traffic Lights:
import asyncio

LIGHT_DURATION = 15
TURN_LIGHT_DURATION = LIGHT_DURATION // 2
YELLOW_DURATION = 4

# Set up car client:
c = airsim.client.CarClient()

## Initialize the client:
camera_pose = airsim.Pose(airsim.Vector3r(20, -15, -40), airsim.to_quaternion(-3.142/2, 0, 3.142/2))  #RPY in radians
c.simSetCameraPose(0, camera_pose)

## Define weather/road wetness and correlate with the scenario:
c.simEnableWeather(True)
#c.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.25)

## Time of Day:
#c.simSetTimeOfDay(True, '17-33-31', False, 1, 60, True)
#c.simSetTimeOfDay(True, '17-33-31', False, 1, 60, True)

## LOGGING:
NCARS = 12  # TBD
NATTRIBUTES = 8  # x, y, z, roll, pitch, yaw
cars = ["Car_" + str(i) for i in range(NCARS)]                         # List of all car names
car_list = [car_name for car_name in cars for i in range(NATTRIBUTES)] # Repeated list to match list of attributes
attributes = ["x", "y", "z", "x_pixel", "y_pixel", "pitch", "roll", "yaw"]                    # What we want to log
attributes_all_cars = attributes * NCARS

## Make folder for logs:
saveDir = os.path.join('D:', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(saveDir)
trainSaveDir = os.path.join(saveDir, 'Train')
os.mkdir(trainSaveDir)
valSaveDir = os.path.join(saveDir, 'Val')
os.mkdir(valSaveDir)
testSaveDir = os.path.join(saveDir, 'Test')
os.mkdir(testSaveDir)
# Save dir for just locations:
os.mkdir(os.path.join(saveDir, "Locations"))

saveSegMask = True
saveRGBImage = True

## Load scenario json to get:
with open('scenarioOutput_100000_V1.json') as f:
  scenarioList = json.load(f)

Nscenarios = 10000

W = 1024
H = 640
(cx, cy) = W / 2, H / 2
f = W / 2  # AirSim camera has an FOV of 90 deg and FOV = 2 * arctan(W/2f)
f = 512
T = np.array([20, -15, -40]).reshape(3, 1)
R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
RT = np.vstack((np.hstack((R, T)), np.array([0, 0, 0, 1])))

async def main():
  
  future = asyncio.ensure_future(LoopTrafficLightsOriginal())
  await asyncio.sleep(2)

  for i in range(1,5001):
    await RunScenarios("Scenario_"+str(i), True, True)
  future.cancel() 

  # Wait for a bit for thread to get cancelled
  await asyncio.sleep(2)

async def LoopTrafficLights():
  global LIGHT_DURATION, TURN_LIGHT_DURATION, YELLOW_DURATION
  EastBoundStops = set(['A0y', 'A1y', 'B0y', 'B1y'])
  WestBoundStops = set(['A1w', 'A2w', 'B1w', 'B2w'])
  NorthBoundStops = set(['C1y', 'C2y', 'D1y', 'D2y'])
  SouthBoundStops = set(['C0w', 'C1w', 'D0w', 'D1w'])
  
  EW = EastBoundStops.union(WestBoundStops)
  NS = NorthBoundStops.union(SouthBoundStops)
  SetLights(EW, 'Left', 'Red')
  SetLights(NS, 'Left', 'Red')
  #Loop
  SetLights(EW, 'Straight', 'Green')
  SetLights(NS, 'Straight', 'Green')

async def LoopTrafficLightsOriginal():
  global LIGHT_DURATION, TURN_LIGHT_DURATION, YELLOW_DURATION
  EastBoundStopsLeft = set(['A0y', 'A1y', 'B0y', 'B1y'])
  EastBoundStopsRight = set(['A0z', 'A1z', 'B0z', 'B1z'])
  WestBoundStopsRight = set(['A1w', 'A2w', 'B1w', 'B2w'])
  WestBoundStopsLeft = set(['A1x', 'A2x', 'B1x', 'B2x'])
  NorthBoundStopsLeft = set(['C1y', 'C2y', 'D1y', 'D2y'])
  NorthBoundStopsRight = set(['C1z', 'C2z', 'D1z', 'D2z'])
  SouthBoundStopsRight = set(['C0w', 'C1w', 'D0w', 'D1w'])
  SouthBoundStopsLeft = set(['C0x', 'C1x', 'D0x', 'D1x'])
  
  EastBoundStops = EastBoundStopsLeft.union(EastBoundStopsRight)
  WestBoundStops = WestBoundStopsLeft.union(WestBoundStopsRight)
  NorthBoundStops = NorthBoundStopsLeft.union(NorthBoundStopsRight)
  SouthBoundStops = SouthBoundStopsLeft.union(SouthBoundStopsRight)

  EW = EastBoundStops.union(WestBoundStops)
  NS = NorthBoundStops.union(SouthBoundStops)
  #Loop
  while True:
    #east/west lights green; north/south lights red.
    SetLights(EW, 'Straight', 'Green')
    #SetLights(EW, 'Left', 'Green')
    SetLights(NS, 'Straight', 'Red')
    await asyncio.sleep(LIGHT_DURATION)
    
    #east/west goes yellow, then red.
    SetLights(EW, 'Straight', 'Yellow')
    SetLights(EW, 'Left', 'Yellow')
    await asyncio.sleep(YELLOW_DURATION)
    SetLights(EW, 'Straight', 'Red')
    SetLights(EW, 'Left', 'Red')
    
    #NSTURNS: north/south left turns activate for a short time, then go yellow, then red.
    SetLights(NS, 'Left', 'Green')
    await asyncio.sleep(TURN_LIGHT_DURATION)
    SetLights(NS, 'Left', 'Yellow')
    await asyncio.sleep(YELLOW_DURATION)
    SetLights(NS, 'Left', 'Red')
    
    #north/south green.
    SetLights(NS, 'Straight', 'Green')
    #SetLights(NS, 'Left', 'Green')
    await asyncio.sleep(LIGHT_DURATION)
    
    #Turn north/south yellow, then red.
    SetLights(NS, 'Straight', 'Yellow')
    SetLights(NS, 'Left', 'Yellow')
    await asyncio.sleep(YELLOW_DURATION)
    SetLights(NS, 'Straight', 'Red')
    SetLights(NS, 'Left', 'Red')
    
    #EWTURNS: east/west left turns activate for a short time, then go yellow, then red.
    SetLights(EW, 'Left', 'Green')
    await asyncio.sleep(TURN_LIGHT_DURATION)
    SetLights(EW, 'Left', 'Yellow')
    await asyncio.sleep(YELLOW_DURATION)
    SetLights(EW, 'Left', 'Red')
    
  
def SetLights(startingLanes, direction, color):
  global c #airsim client
  for lane in startingLanes:
    c.simRunConsoleCommand('ce SetLightState ' + lane + ' ' + direction + ' ' + color)

def create_pose_dict(car_name, pose):
    position = pose.position.to_numpy_array().tolist()                  # Create list of [x, y, z]
    orientation = list(airsim.to_eularian_angles(pose.orientation))     # Create list of [pitch, roll, yaw]
    pixel_position = [pose.position.x_pixel, pose.position.y_pixel]

    dict_keys = zip([car_name] * len(attributes), attributes)           # Create tuples of (car_name, attribute)
    pose_dict = dict(zip(dict_keys, position + pixel_position + orientation))            # Create dictionary of tuple : value

    return pose_dict

async def RunScenarios(scenarioName, recordingOn, loggingOn):

    scenarioNumber = int(scenarioName[9:]) 
    if scenarioNumber<np.round(Nscenarios*0.4):
      saveFolder = os.path.join(trainSaveDir, scenarioName)
    elif scenarioNumber>=400 and scenarioNumber<500:
      saveFolder = os.path.join(valSaveDir, scenarioName)
    elif scenarioNumber>=500:
      saveFolder = os.path.join(testSaveDir, scenarioName)
    os.mkdir(saveFolder)

    ## Logging:

    df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(zip(car_list, attributes_all_cars)))
    df.insert(0, "timestamp", 0)

    ## Initialize logging: 
    logging.basicConfig(level=logging.DEBUG, filename='sample.log')

    ## Run a scenario:
    print(scenarioName)
    c.simRunConsoleCommand('ce RunScenario ' + scenarioName)

    ## Set all elements black:
    found = c.simSetSegmentationObjectID("[\w]*", 0, True)

    ## Loop through vehicles and get IDs:
    cars = []
    for i in range(1,NCARS+1):
        id = 'Car_'+str(i)
        cars.append(c.simGetNpcCarObjectName(id))
        success = c.simSetSegmentationObjectID(cars[i-1], 20*(i+1), True)

    ## Log the pose and position of each vehicle:
    t_end = time.time() + 80
    ind = 0
    results = []
    frames = []
    while ind<150:#150:#time.time() < t_end:

        # GET images:
        segResponse = c.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])

        # GET time.time and append to filename:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.png')
        rgbFilename = ''
        if saveSegMask:
          seg = np.fromstring(segResponse[1].image_data_uint8, dtype=np.uint8) #get numpy array
          if segResponse[1].height!=0:
            img_seg = seg.reshape(segResponse[1].height, segResponse[1].width, 3) #reshape array to 3 channel image array H X W X 3
          else:
            continue
          segFilename = "seg_frame" + str(ind) + "_"+timestamp
          seg = Image.fromarray(img_seg)
          seg.save(os.path.join(saveFolder, segFilename))
        if saveRGBImage:
          rgb = np.fromstring(segResponse[0].image_data_uint8, dtype=np.uint8) #get numpy array
          img_rgb = rgb.reshape(segResponse[0].height, segResponse[0].width, 3) #reshape array to 3 channel image array H X W X 3
          rgbFilename = "rgb_frame" + str(ind) + "_"+timestamp
          rgb = Image.fromarray(img_rgb)
          rgb.save(os.path.join(saveFolder, rgbFilename))

        # Assign different colors to the cars:
        objects = []
        colors = [28, 48, 68, 138, 200, 212, 227, 232, 240, 243, 245, 254]
        car_pose_dicts = []
        car_pixel = dict(x_pixel =  np.nan, y_pixel =  np.nan)
        for i in range(1,NCARS+1):

          car_pose = c.simGetObjectPose(cars[i-1])   

          # Calculate car position in X and Y:
          position_H = np.array([car_pose.position.y_val, car_pose.position.x_val, car_pose.position.z_val, 1]).T.reshape(
              4, 1
          )
          car_xyz_cam = np.matmul(RT, position_H)
          Z = T[2]
          X = car_xyz_cam[0] * f / Z
          Y = car_xyz_cam[1] * f / Z
          X = cx - X
          Y = cy - Y
          car_pose.position.x_pixel = X[0]
          car_pose.position.y_pixel = Y[0]

          # Add car position to dictionary:
          car_pose_dicts.append(create_pose_dict("Car_" + str(i-1), car_pose))
          
          # Create separate masks for each car (necessary for CLEVRER code):
          if saveSegMask:
            pix = np.array(seg)
            pix = 1*(pix[:,:,1]==colors[i-1])
            pix = pix.reshape((pix.shape[0], pix.shape[1], 1))
            pix = pix.astype(np.uint8)

            msk = _mask.encode(np.asfortranarray(pix))
            mask = ({"size" : msk[0]['size'],
                    "counts" : str(msk[0]['counts'].decode("utf-8"))
            })
            objects.append({"car" : "Car_"+str(i),
                    "x_pixel" : car_pose.position.x_pixel,
                    "y_pixel" : car_pose.position.y_pixel,
                    "mask" : mask})
          else:
            objects.append({"car" : "Car_"+str(i),
                    "x_pixel" : car_pose.position.x_pixel,
                    "y_pixel" : car_pose.position.y_pixel})

        car_pose_dict_combined = {k: v for d in car_pose_dicts for k, v in d.items()}       # Convert list of dicts to dict
        car_pose_dict_combined[('frame', 'n')] = ind

        # Add the Mode and Order to the CSV:
        car_pose_dict_combined[('Mode', 'Type')] = scenarioList[scenarioNumber-1]['Mode']
        car_pose_dict_combined[('Mode', 'Order')] = scenarioList[scenarioNumber-1]['Order']

        df = df.append(car_pose_dict_combined, ignore_index=True)    

        frames.append({"frame_filename" : scenarioName+'/'+rgbFilename,
                "frame_index" : ind,
                "objects" : objects}
                )
        
        ind += 1
        await asyncio.sleep(0.01)

    for car in cars:
      c.simDestroyObject(car)

    results.append({"video_index" : scenarioName,
                "mode" : scenarioList[scenarioNumber-1]['Mode'],
                "order" : scenarioList[scenarioNumber-1]['Order'],
                "frames" : frames}
                )
    with open(os.path.join(saveFolder, "json.json"), 'w') as outfile:
        json.dump(results, outfile, indent=4)

    df.to_csv(os.path.join(saveFolder, "locations.csv"))

    df.to_csv(os.path.join(saveDir, "Locations", scenarioName+".csv"))


if __name__ == "__main__":
  asyncio.get_event_loop().run_until_complete(main())
