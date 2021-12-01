# CausalCity: Complex Simulations with Agency for Causal Discovery and Reasoning

This is the official code repository to accompany the paper "CausalCity: Complex Simulations with Agency for Causal Discovery and Reasoning".
Here we provide python code for generating and logging scenarios using the simulation environment as well as links to the baseline code used for running our experiments.

## Data Download:

Location Files (size - 435MB) - https://simulation.blob.core.windows.net/publicrelease/CausalCityLocations.zip

Image, Segmentation and Location Files (size - 82GB) - https://simulation.blob.core.windows.net/publicrelease/Test.zip

Location Files (size - 19MB) - https://simulation.blob.core.windows.net/publicrelease/ToyLocations.zip

## Generate Your Own Data:

To generate your own scenarios you need to create a JSON configuration file that defines those scenarios.

code/scenarioGenerator.py provides an example of how to programmatically create scenarios - which is useful if you want to define 100s or 1000s of scenarios.

Scenario configurations have the following form:

```json
[
    {
        "Name": "Scenario_1",
        "Vehicles": [
            {
                "Id": "Car_1",
                "Spawn": "C0w",
                "SpawnAtDistance": 0,
                "DriveQueue": [
                    "right",
                    "straight",
                    "right",
                    "mergeR",
                    "left"
                ],
                "MergeAtDistances": [
                    0,
                    0,
                    0,
                    50,
                    0
                ]
            }
        ]
   }
]
```

## Controlling Environment Variables

### Weather:
https://microsoft.github.io/AirSim/apis/#weather-apis

```python
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.25);

class WeatherParameter:
    Rain = 0
    Roadwetness = 1
    Snow = 2
    RoadSnow = 3
    MapleLeaf = 4
    RoadLeaf = 5
    Dust = 6
    Fog = 7
```

### Time of Day:

```python
simSetTimeOfDay(self, is_enabled, start_datetime = "", is_start_datetime_dst = False, celestial_clock_speed = 1, update_interval_secs = 60, move_sun = True)
```


