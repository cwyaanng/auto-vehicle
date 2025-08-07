import glob
import os
import sys
import carla
import random
import time
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def generate_route(carla_map, start_coords, end_coords, max_dist=1000, wp_separation=2.0):
  
    print(f"경로 생성 시작: {start_coords} → {end_coords}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1) 시작점 설정 - 지정된 좌표에서 가장 가까운 도로 위 지점
    start_raw = carla.Location(
        x=float(start_coords[0]),
        y=float(start_coords[1]),
        z=0.0
    )
    
    try:
        start_wp = carla_map.get_waypoint(
            start_raw,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if start_wp:
            start_location = start_wp.transform.location
            print(f"시작점 설정: 도로 위 좌표 ({start_location.x:.1f}, {start_location.y:.1f})")
        else:
            print(f"좌표 ({start_raw.x:.1f}, {start_raw.y:.1f}) 근처에 도로를 찾지 못함")
            return
          
    except Exception as e:
        print(f"시작점 설정 중 오류 발생: {e}")
        return
    
    # 2) 목적지 설정 - 지정된 좌표에서 가장 가까운 도로 위 지점
    end_raw = carla.Location(
        x=float(end_coords[0]),
        y=float(end_coords[1]),
        z=0.0
    )
    
    try:
        end_wp = carla_map.get_waypoint(
            end_raw,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if end_wp:
            end_location = end_wp.transform.location
            print(f"목적지 설정: 도로 위 좌표 ({end_location.x:.1f}, {end_location.y:.1f})")
        else:
            print(f"좌표 ({end_raw.x:.1f}, {end_raw.y:.1f}) 근처에 도로를 찾지 못함")
            return
    except Exception as e:
        print(f"목적지 설정 중 오류 발생: {e}")
        return
    
    # 3) 경로 생성 - 시작점에서 웨이포인트를 따라가며 목적지에 가까워지는 경로 생성
    route_waypoints = []
    wp_separation = 2.0  # 웨이포인트 간격 (미터)

    # 시작 웨이포인트 추가
    current_wp = start_wp
    route_waypoints.append(current_wp)
    initial_lane_id = current_wp.lane_id  # 시작 차선 ID 저장
    print(f"경로 생성 시작 - 차선 ID: {initial_lane_id}")

    # 목적지까지의 경로 탐색
    min_distance_to_goal = float('inf')
    closest_wp_to_goal = None
    iterations = 0

    while iterations < max_dist:
        # 현재 웨이포인트에서 다음 웨이포인트들 가져오기
        next_waypoints = current_wp.next(wp_separation)
        
        if not next_waypoints:
            print("경로 탐색 중 다음 웨이포인트가 없습니다. 탐색 종료.")
            break
        
        # 같은 차선의 웨이포인트만 필터링
        same_lane_waypoints = [wp for wp in next_waypoints if wp.lane_id == initial_lane_id]
        
        # 같은 차선에 웨이포인트가 없으면 차선 변경 없이 계속 진행
        if not same_lane_waypoints:
            print(f"경고: 차선 {initial_lane_id}에 다음 웨이포인트가 없습니다. 사용 가능한 웨이포인트 사용.")
            candidate_waypoints = next_waypoints
        else:
            candidate_waypoints = same_lane_waypoints
        
        # 가장 목적지에 가까워지는 웨이포인트 선택
        best_wp = None
        best_distance = float('inf')
        
        for wp in candidate_waypoints:
            # 이 웨이포인트에서 목적지까지의 거리 계산
            dist_to_goal = wp.transform.location.distance(end_location)
            
            if dist_to_goal < best_distance:
                best_distance = dist_to_goal
                best_wp = wp
        
        if best_wp:
            # 경로에 웨이포인트 추가
            route_waypoints.append(best_wp)
            current_wp = best_wp
            
            # 목적지까지의 최소 거리 업데이트
            if best_distance < min_distance_to_goal:
                min_distance_to_goal = best_distance
                closest_wp_to_goal = best_wp
                
            # 목적지에 충분히 가까워지면 종료
            if best_distance < 5.0:  # 5미터 이내면 목적지 도달로 간주
                print(f"목적지에 도달했습니다! (거리: {best_distance:.2f}m)")
                break
        else:
            print("적절한 다음 웨이포인트를 찾지 못함")
            break
        
        iterations += 1
        
    if iterations >= max_dist:
        print(f"최대 탐색 거리({max_dist})에 도달")
    
    print(f"경로 생성 완료: {len(route_waypoints)}개 웨이포인트, 목적지까지 최소 거리: {min_distance_to_goal:.2f}m")
    
    
    # 경로 시각화 및 저장
    import matplotlib.pyplot as plt
    
    # 플롯 생성
    plt.figure(figsize=(15, 12))
    
    # 경로 좌표 추출
    route_x = [wp.transform.location.x for wp in route_waypoints]
    route_y = [wp.transform.location.y for wp in route_waypoints]
    total = [route_x,route_y]
    
    # 경로 시각화 (파란 선)
    plt.plot(route_x, route_y, 'b-', linewidth=3, label='Generated Route')
    
    # 시작점 방향 표시 (화살표)
    start_loc = route_waypoints[0].transform.location
    yaw_deg = route_waypoints[0].transform.rotation.yaw
    yaw_rad = np.deg2rad(yaw_deg)
    arrow_length = 5.0
    dx = arrow_length * np.cos(yaw_rad)
    dy = arrow_length * np.sin(yaw_rad)
    plt.arrow(start_loc.x, start_loc.y, dx, dy,
              head_width=1.0, head_length=1.5, fc='magenta', ec='magenta', label='Spawn Direction')
    
    # 시작점 (초록색 별)
    plt.scatter(route_waypoints[0].transform.location.x, route_waypoints[0].transform.location.y, color='green', s=200, marker='*', label='Start')
    
    # 목적지 (빨간색 별)
    plt.scatter(route_waypoints[-1].transform.location.x, route_waypoints[-1].transform.location.y, color='red', s=200, marker='*', label='Destination')
    
    # 웨이포인트 표시 (10개마다 하나씩 표시)
    wp_indices = list(range(0, len(route_waypoints), 10))
    if len(route_waypoints) - 1 not in wp_indices:
        wp_indices.append(len(route_waypoints) - 1)
    
    for i in wp_indices:
        wp = route_waypoints[i]
        plt.scatter(wp.transform.location.x, wp.transform.location.y, color='orange', s=50)
        # 인덱스 표시
        plt.text(wp.transform.location.x + 2, wp.transform.location.y + 2, str(i), fontsize=9)
    
    # 그래프 설정
    plt.title('Generated Route from Start to Destination', fontsize=16)
    plt.xlabel('X (meters)', fontsize=14)
    plt.ylabel('Y (meters)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 축 비율 동일하게 설정
    plt.axis('equal')
    
    # 저장 디렉토리 생성
    route_viz_dir = "route_visualization"
    os.makedirs(route_viz_dir, exist_ok=True)
    
    # 저장
    plt.savefig(os.path.join(route_viz_dir, f'route_map_{timestamp}.png'), dpi=300)
    print(f"경로 시각화가 저장되었습니다: {os.path.join(route_viz_dir, f'route_map_{timestamp}.png')}")
    
    plt.close()
    if len(route_waypoints) < 2:
        print("유효한 경로 생성 못함")
        return
    
    
    if len(route_waypoints) >= 2:
        # 첫 번째에서 두 번째 웨이포인트로의 벡터
        direction_vector = carla.Vector3D(
            x=route_waypoints[1].transform.location.x - route_waypoints[0].transform.location.x,
            y=route_waypoints[1].transform.location.y - route_waypoints[0].transform.location.y,
            z=0.0
        )
        
        # 벡터의 방향을 yaw 각도로 변환 (라디안에서 도로 변환)
        yaw = math.degrees(math.atan2(direction_vector.y, direction_vector.x))
        
        # 스폰 트랜스폼에 새 방향 설정
        spawn_transform = route_waypoints[0].transform
        spawn_transform.rotation = carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
        spawn_transform.location.z += 0.5
        
        print(f"경로의 첫 번째 지점에 차량 스폰 준비: ({spawn_transform.location.x:.1f}, {spawn_transform.location.y:.1f}, {yaw:.1f}°)")
        
        return route_waypoints

    return route_waypoints
        

        
def calculate_spawn_transform(route_waypoints):
    """
    경로의 첫 번째 구간을 기준으로 차량 스폰 위치 및 방향을 계산합니다.
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
  
def generate_actual_path_plot(route_waypoints, actual_path_x, actual_path_y, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(15, 12))

    # 경로 좌표
    route_x = [wp.transform.location.x for wp in route_waypoints]
    route_y = [wp.transform.location.y for wp in route_waypoints]

    # 경로 및 주행 데이터 시각화
    # 생성된 경로 (파란 선)
    plt.plot(route_x, route_y, 'b-', linewidth=2, label='Planned Route')

    # 실제 차량 주행 경로 (빨간 선, 굵게)
    plt.plot(actual_path_x, actual_path_y, 'r-', linewidth=3, label='Actual Trajectory')

    # 시작점과 도착점
    plt.scatter(route_x[0], route_y[0], color='green', s=300, marker='*', label='Start')  # 별 크기도 키움
    plt.scatter(route_x[-1], route_y[-1], color='red', s=300, marker='*', label='End')

    # 축 라벨 및 제목
    plt.xlabel('X (m)', fontsize=30)
    plt.ylabel('Y (m)', fontsize=30)
    plt.title('Planned Route vs Actual Trajectory', fontsize=30)

    # 기타 스타일
    plt.grid(True)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.axis('equal')

    # 저장
    os.makedirs("route_visualization", exist_ok=True)
    save_path = f"route_visualization/actual_vs_planned_{timestamp}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"실제 주행 경로가 시각화되어 저장됨: {save_path}")



# Carla 서버 연결
def connect_to_carla(host='localhost', port=2000):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    return client, world, carla_map

# 차량 생성
def spawn_vehicle(world, blueprint_library, transform):
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    vehicle = world.spawn_actor(vehicle_bp, transform)
    return vehicle


# PID 제어 및 주행 시작
def run_pid_drive_with_log(vehicle, route_waypoints, actual_path_x, actual_path_y, collision_event, max_steps=1000):
    previous_error = 0.0
    previous_steer = 0.0
    integral = 0.0
    dt = 0.1

    def find_closest_waypoint_index(loc):
        min_dist = float('inf')
        closest_idx = 0
        for i, wp in enumerate(route_waypoints):
            dist = wp.transform.location.distance(loc)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        return closest_idx

    def compute_errors(vehicle, target_wp):
        vehicle_transform = vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

        forward_vec = np.array([np.cos(vehicle_yaw), np.sin(vehicle_yaw)])
        target_loc = target_wp.transform.location
        to_target = np.array([target_loc.x - vehicle_loc.x, target_loc.y - vehicle_loc.y])
        dist = np.linalg.norm(to_target)
        if dist < 1e-6:
            return 0.0, 0.0

        target_vec = to_target / dist
        dot = np.clip(np.dot(forward_vec, target_vec), -1.0, 1.0)
        heading_error = np.arccos(dot)
        cross = forward_vec[0] * target_vec[1] - forward_vec[1] * target_vec[0]
        if cross < 0:
            heading_error *= -1

        normal_vec = np.array([-target_vec[1], target_vec[0]])
        offset_vec = np.array([vehicle_loc.x - target_loc.x, vehicle_loc.y - target_loc.y])
        lateral_error = np.dot(offset_vec, normal_vec)

        return lateral_error, heading_error

    def pid_steering_control(e, theta_e, previous_error, integral, dt):
        kp = 3
        ki = 0.01
        kd = 0.5
        ke = 0.25

        prop = kp * e
        integral += ki * e * dt
        integral = np.clip(integral, -0.5, 0.5)
        derivative = kd * (e - previous_error) / dt
        steer = prop + integral + derivative - ke * theta_e
        steer = -np.clip(steer, -1.0, 1.0)
        return steer, integral

    for t in range(max_steps):
        loc = vehicle.get_location()
        speed_vec = vehicle.get_velocity()
        speed = np.linalg.norm([speed_vec.x, speed_vec.y, speed_vec.z])

        idx = find_closest_waypoint_index(loc)
        target_idx = min(idx + 5, len(route_waypoints) - 1)
        target_wp = route_waypoints[target_idx]

        e, theta_e = compute_errors(vehicle, target_wp)
        steer, integral = pid_steering_control(e, theta_e, previous_error, integral, dt)
        previous_error = e
        steer = 0.7 * steer + 0.3 * previous_steer

        control = carla.VehicleControl(throttle=0.5, steer=steer)
        vehicle.apply_control(control)
        actual_path_x.append(loc.x)
        actual_path_y.append(loc.y)
        
        if collision_event['collided']:
            print(f"[{t}] 🚨 충돌 감지됨, 주행 중단")
            break
          
        print(f"[{t}] steer={steer:.2f}, e={e:.2f}, θe={np.degrees(theta_e):.2f}°")
        time.sleep(dt)
        
def attach_collision_sensor(world, vehicle):
    blueprint = world.get_blueprint_library().find('sensor.other.collision')
    sensor_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
    collision_sensor = world.spawn_actor(blueprint, sensor_transform, attach_to=vehicle)
    collision_event = {'collided': False}

    def _on_collision(event):
        print(f"💥 충돌 발생! 충돌 대상: {event.other_actor.type_id}")
        collision_event['collided'] = True

    collision_sensor.listen(_on_collision)
    return collision_sensor, collision_event



def plot_carla_map(carla_map, save_path="route_visualization/carla_map.png"):
    topology = carla_map.get_topology()
    waypoints = []
    for seg in topology:
        wp_start = seg[0].transform.location
        wp_end = seg[1].transform.location
        waypoints.append((wp_start, wp_end))

    plt.figure(figsize=(15, 12))
    for start, end in waypoints:
        plt.plot([start.x, end.x], [start.y, end.y], 'k-', linewidth=0.5)

    plt.title("CARLA Map View", fontsize=16)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[맵 저장 완료] 전체 맵 이미지 저장됨: {save_path}")


# 메인 진입점
def main():
    client, world, carla_map = connect_to_carla()
    blueprint_library = world.get_blueprint_library()

    # 1. 전체 맵 저장
    plot_carla_map(carla_map)

    # 2. 시작 / 끝 좌표 정의
    start_coords = (-100, -20)
    end_coords = (0, -70)

    # 3. 경로 생성 및 저장
    route_waypoints = generate_route(carla_map, start_coords, end_coords)
    if not route_waypoints:
        print("경로 생성 실패!")
        return

    # 4. 차량 스폰
    spawn_transform = calculate_spawn_transform(route_waypoints)
    spawn_transform = carla_map.get_waypoint(spawn_transform.location, project_to_road=True).transform
    vehicle = spawn_vehicle(world, blueprint_library, spawn_transform)
    collision_sensor, collision_event = attach_collision_sensor(world, vehicle)
  
    print("차량 스폰 및 충돌 센서 부착 완료!")

    # 5. PID 주행 + 실제 경로 저장용 리스트
    actual_path_x = []
    actual_path_y = []
    run_pid_drive_with_log(vehicle, route_waypoints, actual_path_x, actual_path_y,collision_event)
    print("정리 중...")
    vehicle.destroy()
    collision_sensor.destroy()
    print("시뮬레이션 종료 및 리소스 정리 완료")
    # 6. 실제 주행 경로 시각화
    generate_actual_path_plot(route_waypoints, actual_path_x, actual_path_y)


if __name__ == "__main__":
    main()
