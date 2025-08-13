# 요구사항

# 파일 구조
- agents : 제어 모델 
  - pid_control.py : pid 기반 제어 로직 
  - run_sac.py : sac 모델 학습 루프

- env : carla 환경 세팅 및 경로 세팅
  - route.py : 경로 생성 
  - env_set.py : carla 환경, 차량 에이전트 설정 
  - wrapper.py : 모델 학습 env 설정 

- logs : 실험 로그, 결과 저장

- utils : 시각화 등 도구 기능

- run.py : 시뮬레이션 실행 스크립트 

# TO-DO 
- [X] 주행 경로 생성 및 시각화 기능
- [X] carla 시뮬레이터와 차량 환경 세팅 
- [X] PID Controller 구현
- [X] 차량 이동 경로 시각화 
- [X] PID 제어 파라미터 변경하며 데이터 수집 
      - 수집 데이터 : 매 step에서의 앞선 15개 waypoint 위치 정보, 그 행동의 reward 값(강화학습과 동일한 reward function), steer & throttle 값 
- [X] pid 제어 파라미터 로그, 결과 plot 구조적으로 저장
- [X] 강화학습 학습 루프 설정 - offline model pretrain, SAC 학습
- [ ] 학습 로깅 설정 - wandb
