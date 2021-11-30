import os
import json
import time
import argparse
import random

parser = argparse.ArgumentParser('Scenarios generator for AirSim causal reasoning project.')
parser.add_argument('-nv', '--number-vehicles', type=int, required=True,
                    help='Number of vehicles in the scenarios.')
parser.add_argument('-ns', '--number-stages', type=int, required=True,
                    help='Number of stages for each vehicle.')
parser.add_argument('--seed', type=int, default=time.time(), 
                    help='Random seed')
parser.add_argument('-s', '--save_dir', type=str, default='./jsons',
                    help='Location to save the generated json.')

possibleStages = ["left","right","straight"]    # List of possible car actions.
possibleMerges = ["mergeL","mergeR"]            # List of possible car merges.

possibleSpawnPoints = ["A0w","A0x","A0y","A0z","A1w","A1x","A1y","A1z","A2w","A2x","A2y","A2z", "B0w","B0x","B0y","B0z","B1w","B1x","B1y","B1z","B2w","B2x","B2y","B2z", "C0w","C0x","C0y","C0z","C1w","C1x","C1y","C1z","C2w","C2x","C2y","C2z", "D0w","D0x","D0y","D0z","D1w","D1x","D1y","D1z","D2w","D2x","D2y","D2z"]                   # List of possible car spawn points.

def main():
    random.seed(args.seed)
    print("Creating json for scenarios.")

    scenarioJSON = {}
    scenarioJSON["Name"] = "Scenario"

    # List of vehicles:
    vehicles = []

    # Loop through vehicles and create them:
    for i in range(args.number_vehicles):

        stagesList = []
        mergesList = []

        # Loop through stages (all vehicles have the same number of stages and merge list)
        for j in range(args.number_stages):

            stagesList.append( random.choice(possibleStages) )
            mergesList.append( random.choice(possibleMerges) )

        # Append vehicle to the list:
        vehicles.append({"Id" : "Car_"+str(i+1),
                        "Spawn" : random.choice(possibleSpawnPoints),
                        "SpawnAtDistance" : 0,
                        "DriveQueue" : stagesList,
                        "MergeAtDistances" : mergesList}
                        )

    scenarioJSON["Vehicles"] = vehicles

    # Dump to JSON:
    print(scenarioJSON)
    #json_dump = json.dumps(scenarioJSON,indent=2)

    with open('scenarioOutput.json', 'w') as outfile:
        json.dump(scenarioJSON, outfile, indent=4)


if __name__ == "__main__":

    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args
    
    main()
