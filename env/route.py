import os
import carla
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def generate_route(carla_map, start_coords, end_coords, max_dist=500, wp_separation=2.0):
  
    """
      차량이 주행할 target route 생성 및 시각화 
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
        else:
            return
          
          
    except Exception as e:
        print(f"시작점 설정 중 오류 발생: {e}")
        return
    
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
        else:
            print(f"좌표 ({end_raw.x:.1f}, {end_raw.y:.1f}) 근처에 도로를 찾지 못함")
            return
    except Exception as e:
        print(f"목적지 설정 중 오류 발생: {e}")
        return
    
    route_waypoints = []
    wp_separation = 2.0  

    current_wp = start_wp
    route_waypoints.append(current_wp)
    initial_lane_id = current_wp.lane_id 

    min_distance_to_goal = float('inf')
    closest_wp_to_goal = None
    iterations = 0

    while iterations < max_dist:
        next_waypoints = current_wp.next(wp_separation)
        
        if not next_waypoints:
            print("경로 탐색 중 다음 웨이포인트가 없습니다. 탐색 종료.")
            break

        same_lane_waypoints = [wp for wp in next_waypoints if wp.lane_id == initial_lane_id]
        
        if not same_lane_waypoints:
            print(f"경고: 차선 {initial_lane_id}에 다음 웨이포인트가 없습니다. 사용 가능한 웨이포인트 사용.")
            candidate_waypoints = next_waypoints
        else:
            candidate_waypoints = same_lane_waypoints
        
        best_wp = None
        best_distance = float('inf')
        
        for wp in candidate_waypoints:
            dist_to_goal = wp.transform.location.distance(end_location)
            
            if dist_to_goal < best_distance:
                best_distance = dist_to_goal
                best_wp = wp
        
        if best_wp:
            route_waypoints.append(best_wp)
            current_wp = best_wp
            
            if best_distance < min_distance_to_goal:
                min_distance_to_goal = best_distance
                closest_wp_to_goal = best_wp
                
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
  
    # plt.figure(figsize=(15, 12))
    
    # route_x = [wp.transform.location.x for wp in route_waypoints]
    # route_y = [wp.transform.location.y for wp in route_waypoints]
    # total = [route_x,route_y]
  
    # plt.plot(route_x, route_y, 'b-', linewidth=3, label='Generated Route')
    
    # start_loc = route_waypoints[0].transform.location
    # yaw_deg = route_waypoints[0].transform.rotation.yaw
    # yaw_rad = np.deg2rad(yaw_deg)
    # arrow_length = 5.0
    # dx = arrow_length * np.cos(yaw_rad)
    # dy = arrow_length * np.sin(yaw_rad)
    # plt.arrow(start_loc.x, start_loc.y, dx, dy,
    #           head_width=1.0, head_length=1.5, fc='magenta', ec='magenta', label='Spawn Direction')
    
    # plt.scatter(route_waypoints[0].transform.location.x, route_waypoints[0].transform.location.y, color='green', s=200, marker='*', label='Start')
    
    # plt.scatter(route_waypoints[-1].transform.location.x, route_waypoints[-1].transform.location.y, color='red', s=200, marker='*', label='Destination')
    
    # wp_indices = list(range(0, len(route_waypoints), 10))
    # if len(route_waypoints) - 1 not in wp_indices:
    #     wp_indices.append(len(route_waypoints) - 1)
    
    # for i in wp_indices:
    #     wp = route_waypoints[i]
    #     plt.scatter(wp.transform.location.x, wp.transform.location.y, color='orange', s=50)
    #     plt.text(wp.transform.location.x + 2, wp.transform.location.y + 2, str(i), fontsize=9)
        
    # plt.title('Generated Route from Start to Destination', fontsize=16)
    # plt.xlabel('X (meters)', fontsize=14)
    # plt.ylabel('Y (meters)', fontsize=14)
    # plt.grid(True, alpha=0.3)
    # plt.legend(fontsize=12)
    
    # # 축 비율 동일하게 설정
    # plt.axis('equal')
    
    # # 저장 디렉토리 생성
    # route_viz_dir = "logs/route_visualization"
    # os.makedirs(route_viz_dir, exist_ok=True)
    
    # # 저장
    # plt.savefig(os.path.join(route_viz_dir, f'route_map_{timestamp}.png'), dpi=300)
    # print(f"경로 시각화가 저장되었습니다: {os.path.join(route_viz_dir, f'route_map_{timestamp}.png')}")
    
    # plt.close()
    # if len(route_waypoints) < 2:
    #     print("유효한 경로 생성 못함")
    #     return
    
    
    if len(route_waypoints) >= 2:
 
        direction_vector = carla.Vector3D(
            x=route_waypoints[1].transform.location.x - route_waypoints[0].transform.location.x,
            y=route_waypoints[1].transform.location.y - route_waypoints[0].transform.location.y,
            z=0.0
        )
        
        yaw = math.degrees(math.atan2(direction_vector.y, direction_vector.x))
        
        spawn_transform = route_waypoints[0].transform
        spawn_transform.rotation = carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
        spawn_transform.location.z += 0.5
        
        print(f"CAR SPAWN : ({spawn_transform.location.x:.1f}, {spawn_transform.location.y:.1f}, {yaw:.1f}°)")
        
        return route_waypoints

    return route_waypoints
        
