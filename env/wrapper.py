import gym
import numpy as np
from gym import spaces
from agents.pid_control import compute_reward, find_closest_waypoint_index, find_closest_waypoints_ahead, make_obs_from_waypoints,compute_errors
import carla
import os 
from datetime import datetime
from env.env_set import attach_camera_sensor, attach_collision_sensor, connect_to_carla, spawn_vehicle
from env.route import generate_forward_route, generate_long_route, generate_route
from run import calculate_spawn_transform
import utils.visualize as viz
from torch.utils.tensorboard import SummaryWriter

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
        
        self.writer = SummaryWriter(log_dir=simulation)
        self.speed_stop_threshold = 1
        self.stop_count = 0
        self.camera_save_root = simulation+"/driving_scene" 
        
        # 수정 : pi 수정
        self.observation_space = spaces.Box(
            low=np.array([-100.0,-100.0]*15 + [-np.pi/2, -10.0]),
            high=np.array([100.0, 100.0]*15 + [np.pi/2, 10.0]),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.current_idx = 0
        self.simulation_category = simulation
        self.points = points 
        
        self.route_waypoints = None 
        
        self.actual_path_x = []
        self.actual_path_y = []
        
        self.route_waypoints = generate_forward_route(self.map, self.points)
        # viz.generate_planned_path_plot(self.route_waypoints,  self.simulation_category)
        if not self.route_waypoints:
            print("경로 생성 실패")
            return
        self.max_index = len(self.route_waypoints) - 1
    
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        self._global_step = 0
        self._episode_len = 0
        self._episode_return = 0.0
        self.episode_num = 0
        self.count_step = 0
       
    def reset(self):
        self._cleanup()
        self.stop_count = 0 
        self.count_step = 0
        self.actual_path_x = []
        self.actual_path_y = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._episode_return = 0.0
        self._episode_len = 0
        
        # 5회에 걸쳐서 차량 spawn 및 시뮬레이션 초기화 시도 
        for attempt in range(5):
            try: 
                spawn_transform = calculate_spawn_transform(self.route_waypoints)
                spawn_transform = self.map.get_waypoint(spawn_transform.location, project_to_road=True).transform
                self.vehicle = spawn_vehicle(self.world, self.blueprint_library, spawn_transform)
                self.collision_sensor, self.collision_event = attach_collision_sensor(self.world, self.vehicle)
                self.world.tick()
                
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
                print(f"[RESET ERROR] 차량 스폰 실패 시도한 횟수{attempt+1} : {e}")
                import traceback
                traceback.print_exc()

        raise RuntimeError("[RESET ERROR] 차량 스폰 5회 모두 실패.")

    # 한 스텝 이동 
    def step(self, action):
        self.count_step += 1
        
        info = {}
        steer, throttle, brake = float(action[0]) , float(action[1]) , float(action[2]) 
        # 진행 편의를 위해 brake는 0으로 설정 
        brake = 0

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.vehicle.apply_control(control)
        self.world.tick()
        
        speed_vec = self.vehicle.get_velocity()  
        speed = np.linalg.norm([speed_vec.x, speed_vec.y, speed_vec.z])
        
        if speed <= self.speed_stop_threshold:
            self.stop_count += 1
        else :
            self.stop_count = 0
 
        loc = self.vehicle.get_location() 
        waypoints_ahead = find_closest_waypoints_ahead(loc, self.route_waypoints)
        idx = find_closest_waypoint_index(loc, self.route_waypoints)
        target_idx = min(idx + 5, len(self.route_waypoints) - 1) 
        target_wp = self.route_waypoints[target_idx]
        e, theta_e = compute_errors(self.vehicle, target_wp)
        obs = make_obs_from_waypoints(self.vehicle, waypoints_ahead, e, theta_e).astype(np.float32, copy=False)

        self.actual_path_x.append(loc.x)
        self.actual_path_y.append(loc.y)

        collided = self.collision_event['collided']
        reached = (target_idx >= len(self.route_waypoints)-1) 
        done = collided or reached
        
        reward = compute_reward(obs, self.vehicle, collided=collided, reached=reached)
        
        self._episode_return += float(reward)
        self._global_step += 1

        if self.stop_count >= 2000:
            reward -= 5
            done = True 
        
        if self.count_step > 20000:
            done = True
        
        info["metrics"] = {
            "reward/step": float(reward),
            "vehicle/speed_mps": float(speed),
            "progress/target_idx": int(target_idx),
            "status/stop_count": int(self.stop_count),
            "status/collided": int(bool(collided)),
            "status/reached": int(bool(reached)),
            "control/steer": float(steer),
            "control/throttle": float(throttle),
            "control/brake": float(brake),
        }
        
        if "metrics" in info:
            for k, v in info["metrics"].items():
                self.writer.add_scalar(k, v, self._global_step)
        
        
        if done:
            self.count_step = 0
            self.episode_num += 1
            try:
                viz.generate_actual_path_plot(
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
                
            info["episode"] = {
                "reward/episode_return": self._episode_return,
                "episode/done_collided": int(bool(collided)),
                "episode/done_reached": int(bool(reached)),
                "episode/waypoint_ahead" : target_idx 
            }
            
            for k, v in info["episode"].items():
                self.writer.add_scalar(k, v, self.episode_num) 
                
            self._cleanup()
        return obs, reward, done , info

    def _cleanup(self):
        try:
            if self.camera is not None:
                self.camera.stop()
                self.camera.destroy()
        except Exception:
            pass
        self.camera = None

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
        
    def close(self):
        self.writer.close()
        super().close()
