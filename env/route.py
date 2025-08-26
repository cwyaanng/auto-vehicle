import os
import carla
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def generate_route(carla_map, start_coords, end_coords, max_dist=1000, wp_separation=2.0):
  
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
            # print(f"경고: 차선 {initial_lane_id}에 다음 웨이포인트가 없습니다. 사용 가능한 웨이포인트 사용.")
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

def generate_forward_route(
    carla_map,
    start_coords,
    wp_separation: float = 2.0,
    steps: int = 1000,   # 3000 스텝 전진             
    prefer_same_lane: bool = True,    
    max_steps_cap: int = 100000,      
    verbose: bool = True,
):
    """
    start_coords에서 시작해 '앞으로' 진행하며 웨이포인트 리스트를 생성.
    - steps번 앞으로 진행


    반환: List[carla.Waypoint]
    """
    start_raw = carla.Location(float(start_coords[0]), float(start_coords[1]), 0.0)
    start_wp = carla_map.get_waypoint(start_raw, project_to_road=True, lane_type=carla.LaneType.Driving)
    if start_wp is None:
        if verbose: print(f"[forward] 시작점 근처 도로 없음: ({start_raw.x:.1f}, {start_raw.y:.1f})")
        return []

    route = [start_wp]
    cur = start_wp
    lane_id0 = cur.lane_id

    # 전진 조건
    remain_steps = steps if steps is not None else None
    traveled = 0.0

    it = 0
    while it < max_steps_cap:
        nxts = cur.next(wp_separation)
        if not nxts:
            if verbose: print("[forward] 다음 웨이포인트 없음. 종료")
            break

        # 후보 선정: 같은 차선 선호(없으면 전체)
        cand = ([wp for wp in nxts if wp.lane_id == lane_id0] or nxts) if prefer_same_lane else nxts

        # 진행 방향(헤딩) 연속성 + 전진성 점수화
        best, best_score = None, float("inf")
        fwd = cur.transform.get_forward_vector()
        fwd_vec = np.array([fwd.x, fwd.y], dtype=np.float32)
        fwd_norm = np.linalg.norm(fwd_vec) + 1e-9

        for wp in cand:
            f2 = wp.transform.get_forward_vector()
            f2_vec = np.array([f2.x, f2.y], dtype=np.float32)
            cos_th = float(np.dot(fwd_vec, f2_vec) / (fwd_norm * (np.linalg.norm(f2_vec) + 1e-9)))
            heading_penalty = 1.0 - np.clip(cos_th, -1.0, 1.0)  
          
            dx = wp.transform.location.x - cur.transform.location.x
            dy = wp.transform.location.y - cur.transform.location.y
            forward_proj = float((dx * fwd_vec[0] + dy * fwd_vec[1]) / (fwd_norm + 1e-9))
            forward_penalty = -forward_proj  

            score = 2.0 * heading_penalty + 0.1 * forward_penalty
            if score < best_score:
                best_score, best = score, wp

        if best is None:
            if verbose: print("[forward] 적절한 다음 웨이포인트 없음. 종료")
            break

        route.append(best)
        moved = cur.transform.location.distance(best.transform.location)
        traveled += moved
        cur = best
        it += 1

        if remain_steps is not None:
            remain_steps -= 1
            if remain_steps <= 0:
                if verbose: print(f"[forward] steps 완료: traveled={traveled:.1f} m, steps={steps}")
                break

    if it >= max_steps_cap and verbose:
        print(f"[forward] max_steps_cap({max_steps_cap}) 도달")

    if verbose:
        print(f"[forward] 경로 생성: {len(route)} wp, 총 거리 {traveled:.1f} m")

    return route



def generate_long_route(carla_map, points, wp_separation=2.0, max_dist=500):
    full_route = []
    last_wp = None

    for i in range(len(points) - 1):
        if i == 0:
            # 첫 구간은 기존대로
            seg = generate_route(
                carla_map,
                start_coords=points[0],
                end_coords=points[1],
                max_dist=max_dist,
                wp_separation=wp_separation,
            )
        else:
            # 다음 구간 시작을 '직전 세그먼트 마지막 WP 위치'로
            lw = full_route[-1].transform.location
            start_coords = (lw.x, lw.y, lw.z)
            seg = generate_route(
                carla_map,
                start_coords=start_coords,   
                end_coords=points[i + 1],
                max_dist=max_dist,
                wp_separation=wp_separation,
            )

        if not seg or len(seg) < 2:
            raise RuntimeError(f"경로 실패: {points[i]} → {points[i+1]}")

        if i == 0:
            full_route.extend(seg)
        else:
            full_route.extend(seg[1:]) 

    return full_route

    

def visualize_all_waypoints(carla_map, separation=2.0, save_dir="./plots"):
    """
    CARLA 맵의 모든 waypoint를 시각화하여 plot으로 저장하는 함수

    Args:
        carla_map: carla.Map 객체
        separation: waypoint 간격 (미터)
        save_dir: 이미지 저장 디렉토리
    """
    try:
        # Waypoint 생성
        all_wps = carla_map.generate_waypoints(separation)

        xs = [wp.transform.location.x for wp in all_wps]
        ys = [wp.transform.location.y for wp in all_wps]

        # 시각화
        plt.figure(figsize=(10, 10))
        plt.scatter(xs, ys, s=5, c="blue", alpha=0.6, label="Waypoints")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.title(f"CARLA Map Waypoints (Separation={separation}m)")
        plt.legend()
        plt.axis("equal")

        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(save_dir, f"carla_map_waypoints_{timestamp}.png")

        plt.savefig(file_path, dpi=300)
        plt.close()

        print(f"Waypoint 시각화 저장 완료: {file_path}")
        return file_path

    except Exception as e:
        print(f"Waypoint 시각화 중 오류 발생: {e}")
        return None