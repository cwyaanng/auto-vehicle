import glob
import os
import numpy as np
import torch as th
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

"""
  SAC로 offline to online 학습하는 루프 구현 
"""
class SACOfflineOnline:
  def __init__(self,env, tensorboard_log=None, buffer_size=5_000_000, batch_size=1024, tau=0.005, verbose=1):
    self.env = env
    self.model = SAC(
      "MlpPolicy",
      env,
      learning_starts=0,
      buffer_size=buffer_size,
      tau=tau,
      verbose=verbose,
      tensorboard_log=tensorboard_log
    )
    self.actor = self.model.policy.actor 
    self.critic = self.model.policy.critic  
    self.critic_target = self.model.policy.critic_target
    self.actor_opt = self.actor.optimizer 
    self.critic_opt = self.critic.optimizer 
    
    self.gamma = float(self.model.gamma)
    self.tau = float(self.model.tau)
    self.batch_size = self.model.batch_size
    self.device = self.model.device 
  
    self.auto_alpha = self.model.ent_coef_optimizer is not None
    if self.auto_alpha:
      self.log_ent_coef = self.model.log_ent_coef
      self.ent_coef_opt = self.model.ent_coef_optimizer
      self.target_entropy = self.model.target_entropy 
  
    self.auto_alpha = self.model.ent_coef_optimizer is not None  
    if self.auto_alpha:
        self.log_ent_coef = self.model.log_ent_coef     
        self.ent_coef_opt = self.model.ent_coef_optimizer
        self.target_entropy = self.model.target_entropy  
        
  """
    오프라인 critic 학습 루프 
  """
  
  def _alpha(self):
      if self.auto_alpha:
          with th.no_grad():
              return self.log_ent_coef.exp().detach()
    
      val = float(self.model.ent_coef) if isinstance(self.model.ent_coef, (int, float)) else 0.2
      return th.tensor(val, device=self.device)

  def prefill_from_npz_folder(self, data_dir, clip_actions=True):
      files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
      if not files:
          raise FileNotFoundError("No .npz files in {}".format(data_dir))

      act_low  = getattr(self.env.action_space, "low", None)
      act_high = getattr(self.env.action_space, "high", None)

      n_added, n_files = 0, 0

      for path in files:
          with np.load(path, allow_pickle=False) as d:
              obs  = d["observations"].astype(np.float32)
              acts = d["actions"].astype(np.float32)
              rews = d["rewards"].astype(np.float32).reshape(-1, 1)
              nobs = d["next_observations"].astype(np.float32)
              dones = d["terminals"].astype(np.float32).reshape(-1, 1)

          N = min(len(obs), len(acts), len(rews), len(nobs), len(dones))
          if N == 0:
              continue
          obs, acts, rews, nobs, dones = obs[:N], acts[:N], rews[:N], nobs[:N], dones[:N]

          if clip_actions and act_low is not None and act_high is not None:
              acts = np.clip(acts, act_low, act_high)

          for o, no, a, r, d in zip(obs, nobs, acts, rews, dones):
            self.model.replay_buffer.add(
                o[None, :],            # (1, obs_dim)
                no[None, :],           # (1, obs_dim)
                a[None, :],            # (1, act_dim)
                np.array([float(r)], dtype=np.float32),   # (1,)
                np.array([bool(d)], dtype=np.float32),    # (1,)
                [ {"TimeLimit.truncated": False} ]        # ★ 리스트 안에 dict 하나
            )


          n_added += N
          n_files += 1

      print("Prefilled {} transitions from {} files.".format(n_added, n_files))
      return n_added

  """
    ---------- Critic만 오프라인 사전학습 ----------
  """
  def pretrain_critic(self, steps=5000, polyak_every=2):
      self.critic.train()
      self.actor.eval()
      for step in range(steps): # 오프라인 배치로 steps번 업데이트 
          batch = self.model.replay_buffer.sample(self.batch_size)

          with th.no_grad():
              next_actions, next_logp = self.actor.action_log_prob(batch.next_observations)
              # 다음 상태 s'에서 현재 정책 π로 행동 a' 샘플 + log π(a'|s') 계산
              # SAC는 확률정책이라 action과 log_prob을 함께 얻습니다.
              if next_logp.dim() == 1:
                next_logp = next_logp.view(-1, 1)
                
              tq1, tq2 = self.critic_target(batch.next_observations, next_actions)
              # 타겟 크리틱 추정값 
              
              tmin = th.min(tq1, tq2)
              # 두 Q 중 더 작은 값 사용 -> overestimation 줄이기
              
              target_q = batch.rewards + self.gamma * (1.0 - batch.dones) * (tmin - self._alpha() * next_logp)
              # SAC 타깃:
              # y = r + γ(1-d) * [ min(Q'(s',a')) - α * log π(a'|s') ]
              #  - (1 - d)로 에피소드 종료 시 부트스트랩 차단
              #  - 엔트로피 보너스: - α * logπ (탐험을 장려)


          q1, q2 = self.critic(batch.observations, batch.actions)
          # 현재 크리틱 Q1(s,a), Q2(s,a) 예측
          
          critic_loss = th.nn.functional.mse_loss(q1, target_q) + th.nn.functional.mse_loss(q2, target_q)
          # 크리틱 손실: (Q1 - y)^2 + (Q2 - y)^2, 두 네트워크를 모두 타깃 y에 맞춤

          self.critic_opt.zero_grad() # 크리틱 옵티마이저 기울기 초기화
          critic_loss.backward() # 손실에 대한 역전파 
          self.critic_opt.step() # 한 스텝 최적화 
          print(f"[CRITIC PRETRAIN] critic loss : {critic_loss}")
          if (step + 1) % polyak_every == 0:
              polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

  """
    ---------- Actor만 오프라인 사전학습(critic 고정) ----------
  """
  def pretrain_actor(self, steps=3000):
      # critic을 고정해서, actor 업데이트 때 Q 네트워크가 바뀌지 않도록 함
      for p in self.critic.parameters():
          p.requires_grad = False
      self.critic.eval()
      self.actor.train()
      
      # 오프라인 버퍼에서 배치를 뽑아 actor만 'steps'번 업데이트
      for _ in range(steps):
          batch = self.model.replay_buffer.sample(self.batch_size)
          # 리플레이 버퍼에서 (s, a, r, s', done) 중 's'만 필요 (actor는 s→행동 분포 학습)

          actions_pi, logp_pi = self.actor.action_log_prob(batch.observations)
          # 현재 정책 π(a|s)로부터 행동 a_π와 log π(a_π|s)를 계산
          # (SAC의 squashed Gaussian 정책: tanh 변환 포함해 올바른 log prob 반환)
          
          q1_pi, q2_pi = self.critic(batch.observations, actions_pi)
          # 고정된 critic으로 Q1(s, a_π), Q2(s, a_π) 값을 평가
          q_pi = th.min(q1_pi, q2_pi).detach() 
          # Double Q trick: 더 작은 Q를 사용해 과대추정 방지
          # detach(): actor 업데이트 중에 critic 쪽으로 그라디언트가 흐르지 않게 차단(critic은 고정 상태 유지)

          alpha = self._alpha()
          # 현재 사용할 엔트로피 온도 α 획득 (자동 튜닝이면 exp(logα), 아니면 고정값 텐서)
          actor_loss = (alpha * logp_pi - q_pi).mean()
          # SAC actor 목적: J_π = E[ α·logπ(a|s) - Q(s,a) ] 를 최소화
          # = 엔트로피(무작위성)를 키우되(Q가 큰 행동을 선호), Q가 큰 행동을 더 선택하도록 학습
          print(f"[ACTOR PRETRAIN] actor loss : {actor_loss}")
          self.actor_opt.zero_grad() # 이전 step의 기울기 초기화(누적 방지)
          actor_loss.backward()  # 역전파: actor 파라미터에 대한 grad 계산
          self.actor_opt.step()  # 옵티마이저로 actor 파라미터 한 스텝 업데이트

          if self.auto_alpha:
              # α 자동 튜닝: L(α) = E[ - logα · (logπ + target_entropy) ]
              ent_coef_loss = -(self.log_ent_coef * (logp_pi + self.target_entropy).detach()).mean()
              # detach(): α 업데이트가 actor로 역전파되지 않도록 차단 (정책 학습과 분리)
              self.ent_coef_opt.zero_grad() # logα 옵티마이저의 기울기 초기화
              ent_coef_loss.backward() # logα에 대한 grad 계산
              self.ent_coef_opt.step()  # logα 업데이트 (→ α = exp(logα) 변경) 

      # (3) actor 사전학습이 끝났으니, critic을 다시 학습 가능 상태로 풀고 모드 복귀
      for p in self.critic.parameters():
          p.requires_grad = True
      self.critic.train()
      self.actor.train()

  """ 
    ---------- Online 파인튜닝(표준 SB3 루프) ---------- 
  """
  def online_learn(self, total_timesteps, tb_log_name="phase_online"):
      self.model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)

  def save(self, path):
      self.model.save(path)

  def load(self, path):
      self.model = SAC.load(path, env=self.env)
