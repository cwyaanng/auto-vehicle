import os
import sys
# 현재 디렉토리에서 carla-*.egg 파일 경로를 자동으로 찾아서 sys.path에 추가
sys.path.append('/home/jingjingee198/auto-vehicle/PythonAPI/carla-0.9.8-py3.5-linux-x86_64.egg')
import carla

from agents.pid_control import run_pid_drive_with_log
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from env.env_set import attach_camera_sensor, attach_collision_sensor, connect_to_carla, spawn_vehicle
from env.route import generate_route, visualize_all_waypoints
from utils.visualize import generate_actual_path_plot, plot_carla_map
import itertools

def calculate_spawn_transform(route_waypoints):
    """
      경로의 첫 번째 구간을 기준으로 차량 스폰 위치 및 방향을 계산
    """
    if len(route_waypoints) < 2:
        raise ValueError("웨이포인트가 충분하지 않습니다.")

    wp1 = route_waypoints[0].transform.location
    wp2 = route_waypoints[1].transform.location

    dx = wp2.x - wp1.x
    dy = wp2.y - wp1.y
    yaw = math.degrees(math.atan2(dy, dx))

    spawn_transform = carla.Transform(
        location=carla.Location(x=wp1.x, y=wp1.y, z=wp1.z + 0.5),
        rotation=carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
    )
  
    return spawn_transform

def pid_control(start_coords, end_coords , filename):
    """
      시뮬레이션 실행 로직 
    """
    """
      PID 기반 주행 시뮬레이션 로직    
    """
    # 이산적으로 쪼갠 parameter 값들
    param_combinations = itertools.product(
        [1.0, 3.0, 5.0],  # kp_s
        [0.0, 0.01, 0.02],         # ki_s
        [0.0, 0.3, 0.6],           # kd_s
        [0.3, 0.6],                # ke
        [0.5, 1.0],           # kp_t
        [0.0, 0.05],              # ki_t
        [0.0, 0.05]               # kd_t
    )

    a = 0
    for i, (kp_s, ki_s, kd_s, ke, kp_t, ki_t, kd_t) in enumerate(param_combinations):
        
        client, world, carla_map = connect_to_carla()
        blueprint_library = world.get_blueprint_library()
        # visualize_all_waypoints(carla_map)
    
        start = start_coords 
        end = end_coords 
        
        route_waypoints = generate_route(carla_map, start, end)
        if not route_waypoints:
            print("경로 생성 실패")
            return

        spawn_transform = calculate_spawn_transform(route_waypoints)
        spawn_transform = carla_map.get_waypoint(spawn_transform.location, project_to_road=True).transform
        vehicle = spawn_vehicle(world, blueprint_library, spawn_transform)
        collision_sensor, collision_event = attach_collision_sensor(world, vehicle)
        # camera_sensor = attach_camera_sensor(world, vehicle, save_path="logs/driving_scene")
        
        actual_path_x = []
        actual_path_y = []
        
        run_pid_drive_with_log(world, vehicle, route_waypoints, actual_path_x, actual_path_y, collision_event,kp_s, ki_s, kd_s, ke, kp_t, ki_t, kd_t, a ,filename)
        a += 1
        vehicle.destroy()
        collision_sensor.destroy()
        generate_actual_path_plot(route_waypoints, actual_path_x, actual_path_y,  filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("wrong argument : python run.py [PID|MPC|BEHAVIOR|...]\nexample : python run.py PID")
        sys.exit(1)

    mode = sys.argv[1].upper()
    print(mode)
    if mode == "PID":
        start_coords = (150, 50)
        end_coords = (200, 50)
        pid_control(start_coords, end_coords, "route_1")
        
        start_coords = (-125, 0)
        end_coords = (0, -70)
        pid_control(start_coords, end_coords, "route_2")
        
        start_coords = (150, 50)
        end_coords = (50, 125) 
        pid_control(start_coords, end_coords, "route_3")
        
        start_coords = (0, 20)
        end_coords = (-50, 150) 
        pid_control(start_coords, end_coords, "route_4")
        
        start_coords = (-150, 100)
        end_coords = (-100, 0) 
        pid_control(start_coords, end_coords, "route_5")
    if mode == "COLLECTION":
        # start_coords = (-100,-120)
        # end_coords = (-100,-20)
        # pid_control(start_coords, end_coords, "route_6")
        
        start_coords = (100,58)
        end_coords = (100,80)
        pid_control(start_coords, end_coords, "route_7")
    else:
        print(f"❌ wrong argument")
