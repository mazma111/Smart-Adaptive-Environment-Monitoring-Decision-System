import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta

np.random.seed(42)

start_time= datetime.now()
timestamps = [start_time + timedelta(minutes=i) for i in range (1000)]

temperature = np.random.normal(loc=24.0, scale=0.5, size=1000)
humidity = np.random.normal(loc=50.0, scale=2.0, size=1000)   
gas_level = np.random.normal(loc=120.0, scale=5.0, size=1000) 
light_intensity = np.random.normal(loc=300.0, scale=10.0, size=1000)

hvac_status = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
motion_alert = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])

for i in range(500, 520):
    temperature[i] += 15.0 
    gas_level[i] += 400.0

df = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': temperature,
    'humidity': humidity,
    'gas_level': gas_level,
    'light_intensity': light_intensity,
    'hvac_status': hvac_status,
    'motion_alert': motion_alert
})

df.to_csv('data/raw_sensor_stream.csv', index=False)
print("raw_sensor_stream.csv successfully created in data/ directory!")


import json

nodes = [
    {"id": 0, "name": "temp_sensor_1", "type": "temperature"},
    {"id": 1, "name": "humidity_sensor_1", "type": "humidity"},
    {"id": 2, "name": "gas_sensor_1", "type": "gas"},
    {"id": 3, "name": "light_sensor_1", "type": "light"}
]

edges = [
    {"source": 0, "target": 1, "distance_meters": 0.5},
    {"source": 0, "target": 2, "distance_meters": 1.2},
    {"source": 1, "target": 2, "distance_meters": 1.0},
    {"source": 2, "target": 3, "distance_meters": 4.5}
]

graph_data = {
    "nodes": nodes,
    "edges": edges
}

with open('data/topology_graph.json', 'w') as json_file:
    json.dump(graph_data, json_file, indent=4)

print("topology_graph.json successfully created in data/ directory!")