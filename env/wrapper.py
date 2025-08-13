import gym
import numpy as np
from gym import spaces
from agents.pid_control import compute_reward, find_closest_waypoint_index, find_closest_waypoints_ahead, make_obs_from_waypoints,compute_errors
import carla
import os 
from datetime import datetime
from env.env_set import attach_collision_sensor, connect_to_carla, spawn_vehicle
from env.route import generate_long_route, generate_route
from run import calculate_spawn_transform
from utils.visualize import generate_actual_path_plot 

# 외부에서 carla 환경에 연결하고 경로 생성 후 env 클래스 객체 생성 
class CarlaWrapperEnv(gym.Env):
    def __init__(self, client, world, carla_map, points, simulation, target_speed=22.0):
        super(CarlaWrapperEnv, self).__init__()
        self.client = client
        self.world = world
        self.map = carla_map
        self.blueprint_library = world.get_blueprint_library()
        self.route_waypoints = None
        self.target_speed = target_speed
        self.vehicle = None
        self.collision_sensor = None
        self.collision_event = {"collided": False}
        
        # 정지라고 판단할 속도, 정지한 스텝 수
        self.speed_stop_threshold = 1
        self.stop_count = 0
        
        # [변경됨] 관측 공간: 15개 waypoint의 x좌표 + 각도 차이 + lateral offset
        self.observation_space = spaces.Box(
            low=np.array([-100.0,-100.0]*15 + [-np.pi, -10.0]),
            high=np.array([100.0, 100.0]*15 + [np.pi, 10.0]),
            dtype=np.float32
        )
        # 행동 공간: steering [-1, 1], throttle [0, 1] . brake [0,1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.current_idx = 0
        self.timestamp =  datetime.now().strftime("%Y%m%d_%H%M%S")
        self.simulation_category = simulation
        self.points = points 
        
        self.route_waypoints = None 
        
        self.actual_path_x = []
        self.actual_path_y = []
        
        # 경로 생성
        self.route_waypoints = generate_long_route(self.map, self.points)
        if not self.route_waypoints:
            print("경로 생성 실패")
            return
        self.max_index = len(self.route_waypoints) - 1
        # carla  동기 모드 
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        # wandb로 로깅 해야함 # 
        
    # 에피소드 시작 부분 
    def reset(self):
        self._cleanup()
        self.stop_count = 0 
        self.actual_path_x = []
        self.actual_path_y = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 5번의 차량 SPAWN 시도 
        for attempt in range(5):
            try:
                # 차량 스폰 및 충돌 센서 부착 
                spawn_transform = calculate_spawn_transform(self.route_waypoints)
                spawn_transform = self.map.get_waypoint(spawn_transform.location, project_to_road=True).transform
                self.vehicle = spawn_vehicle(self.world, self.blueprint_library, spawn_transform)
                self.collision_sensor, self.collision_event = attach_collision_sensor(self.world, self.vehicle)
                
                self.world.tick()
                
                # 초기 관측값 계산 
                loc = self.vehicle.get_location()
                vel = self.vehicle.get_velocity()
                speed = np.linalg.norm([vel.x, vel.y, vel.z])  
                waypoints_ahead = find_closest_waypoints_ahead(loc, self.route_waypoints)
                idx = find_closest_waypoint_index(loc, self.route_waypoints)
                target_idx = min(idx + 5, len(self.route_waypoints) - 1)
                target_wp = self.route_waypoints[target_idx]
                e, theta_e = compute_errors(self.vehicle, target_wp)
                
                self.actual_path_x.append(loc.x)
                self.actual_path_y.append(loc.y)
                
                obs = make_obs_from_waypoints(self.vehicle, waypoints_ahead, e, theta_e).astype(np.float32, copy=False)
                return obs

            except Exception as e:
                print(f"[RESET ERROR] 차량 스폰 실패 (시도 {attempt+1}): {e}")
                import traceback
                traceback.print_exc()
            

        raise RuntimeError("[RESET ERROR] 차량 스폰 5회 모두 실패. 환경 초기화 중단.")



    def step(self, action):
        info = {}
        steer , throttle , brake = float(action[0]) , float(action[1]) , float(action[2])
        brake = 0
        
        # --- control 적용 ---
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.vehicle.apply_control(control)
        self.world.tick()
        
        speed_vec = self.vehicle.get_velocity()  # 차량 속도 벡터 (x, y, z 구성)
        speed = np.linalg.norm([speed_vec.x, speed_vec.y, speed_vec.z])
        
        if speed <= self.speed_stop_threshold:
            self.stop_count += 1
        else :
            self.stop_count = 0

        # 현재 상태 관측 
        loc = self.vehicle.get_location()
        waypoints_ahead = find_closest_waypoints_ahead(loc, self.route_waypoints)
        idx = find_closest_waypoint_index(loc, self.route_waypoints)
        target_idx = min(idx + 5, len(self.route_waypoints) - 1)
        target_wp = self.route_waypoints[target_idx]
        e, theta_e = compute_errors(self.vehicle, target_wp)
        obs = make_obs_from_waypoints(self.vehicle, waypoints_ahead, e, theta_e).astype(np.float32, copy=False)

        # 주행 경로 시각화용 저장
        self.actual_path_x.append(loc.x)
        self.actual_path_y.append(loc.y)

        collided = self.collision_event['collided']
        reached = (target_idx >= len(self.route_waypoints)-1)  # 타겟이 마지막 웨이포인트에 도달
        done = collided or reached
        
        reward = compute_reward(obs, self.vehicle, collided=collided, reached=reached)
        # print(f"reward : {reward} , target_idx : { target_idx} ")
        if self.stop_count >= 2000:
            reward -= 5
            done = True 
        
        if done:
            try:
                generate_actual_path_plot(
                    route_waypoints=self.route_waypoints,
                    actual_path_x=self.actual_path_x,
                    actual_path_y=self.actual_path_y,
                    simulation_category=self.simulation_category,
                    timestamp=self.timestamp,
                )
                info["plot_saved"] = True
            except Exception as ex:
                print(f"[PLOT ERROR] {ex}")
                info["plot_saved"] = False

        return obs, reward, done , info

    def _cleanup(self):
        try:
            if self.collision_sensor is not None:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
        except Exception:
            pass
        self.collision_sensor = None

        try:
            if self.vehicle is not None:
                self.vehicle.destroy()
        except Exception:
            pass
        self.vehicle = None