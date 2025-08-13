import carla 
import os 
from datetime import datetime

def connect_to_carla(host='localhost', port=2000):
    """
      carla 시뮬레이션 환경에 연결 
      
      returns :
        client : carla 클라이언트 객체 
        world : 현재 로드된 시뮬레이션 월드 
        carla_map : 현재 월드에 대한 맵 정보
    """
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    return client, world, carla_map

def spawn_vehicle(world, blueprint_library, transform):
    """
      차량 객체를 시뮬레이션 월드에 spawn 
      
      return :
        vehicle : 차량 
    """
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    vehicle = world.spawn_actor(vehicle_bp, transform)
    return vehicle

def attach_collision_sensor(world, vehicle):
    """
      출동 감지 센서를 차량에 부착하고 충돌 이벤트를 추적  
    """
    blueprint = world.get_blueprint_library().find('sensor.other.collision')
    sensor_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
    collision_sensor = world.spawn_actor(blueprint, sensor_transform, attach_to=vehicle)
    collision_event = {'collided': False}

    def _on_collision(event):
        print(f"충돌 발생 - 충돌 대상: {event.other_actor.type_id}")
        collision_event['collided'] = True

    collision_sensor.listen(_on_collision)
    return collision_sensor, collision_event
  
def attach_camera_sensor(world, vehicle, image_width=800, image_height=600, fov=90, sensor_tick=0.1, save_path='logs/driving_scene'):
    
    """
      카메라 센서를 차량에 부착 
      카메라 센서가 새로운 이미지 프레임을 캡처할 때마다 콜백 함수 자동 호출 
    """
    os.makedirs(save_path, exist_ok=True)
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(image_width))
    camera_bp.set_attribute('image_size_y', str(image_height))
    camera_bp.set_attribute('fov', str(fov))
    camera_bp.set_attribute('sensor_tick', str(sensor_tick))  

    camera_transform = carla.Transform(
      carla.Location(x=-6.0, z=3.5),  
      carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0)  
    )
    
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    def _on_image(image):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}/{timestamp}_{image.frame:06d}.png"
        image.save_to_disk(filename)
        print(f"이미지 저장됨: {filename}")

    camera.listen(_on_image)
    return camera
