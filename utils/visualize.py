from datetime import datetime
import os
import matplotlib.pyplot as plt

def plot_carla_map(carla_map, save_path="logs/route_visualization/carla_map.png"):
    topology = carla_map.get_topology()
    waypoint_locations = set()

    for seg in topology:
        wp_start = seg[0].transform.location
        wp_end = seg[1].transform.location
        waypoint_locations.add((wp_start.x, wp_start.y))
        waypoint_locations.add((wp_end.x, wp_end.y))

    x_vals = [pt[0] for pt in waypoint_locations]
    y_vals = [pt[1] for pt in waypoint_locations]

    plt.figure(figsize=(15, 12))
    plt.scatter(x_vals, y_vals, c='k', s=1)  

    plt.title("CARLA Map View (Dots)", fontsize=16)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

  
def generate_actual_path_plot(route_waypoints, actual_path_x, actual_path_y, simulation_category, timestamp=None):
    """
      시뮬레이션 동안의 차량의 주행 경로를 시각화하여 저장 
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(15, 12))

    route_x = [wp.transform.location.x for wp in route_waypoints]
    route_y = [wp.transform.location.y for wp in route_waypoints]
    
    plt.plot(route_x, route_y, 'b-', linewidth=2, label='Planned Route')

    plt.plot(actual_path_x, actual_path_y, 'r-', linewidth=3, label='Actual Trajectory')

    plt.scatter(route_x[0], route_y[0], color='green', s=300, marker='*', label='Start')  # 별 크기도 키움
    plt.scatter(route_x[-1], route_y[-1], color='red', s=300, marker='*', label='End')

    plt.xlabel('X (m)', fontsize=30)
    plt.ylabel('Y (m)', fontsize=30)
    plt.title('Planned Route vs Actual Trajectory', fontsize=30)

    plt.grid(True)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.axis('equal')

    os.makedirs(f"logs/route_visualization/{simulation_category}/_{timestamp}", exist_ok=True)
    save_path = f"logs/route_visualization/{simulation_category}/_{timestamp}/actual_vs_planned_{timestamp}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"실제 주행 경로가 시각화되어 저장됨: {save_path}")

