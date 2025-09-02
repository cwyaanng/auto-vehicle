# agents/sacrnd_model.py
import glob
import os
from typing import Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update 


class RunningMeanStd:
    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
    def update(self, x: th.Tensor):
        # x: (B,1)
        batch_mean = x.mean().item()
        batch_var  = x.var(unbiased=False).item()
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count

class SACOfflineOnline(SAC): # SAC 상속한 커스텀 에이전트 

    def __init__(
        self,
        env, # 학습 환경 
        policy: str = "MlpPolicy", # SB3 정책 이름
        learning_starts: int = 0, # 오프라인 prefill [의문]
        **sac_kwargs,
    ):
        if "learning_starts" not in sac_kwargs:
            sac_kwargs["learning_starts"] = learning_starts

        super().__init__(policy, env, **sac_kwargs) # 부모 SAC 모델 초기화 
        self._nov_rms = RunningMeanStd()
        self.rnd = None # rnd 모듈 정의 
        self.rnd_update_every = 5
        self.rnd_update_steps = 1
     
        # 몬테 카를로 return 캐시를 위한 변수 
        self._mc_cached_size: int = -1  # 유효 버퍼 길이 [의문]
        self._mc_cached_pos  = -1
        self._mc_cached_full = False

        self._mc_returns: Optional[np.ndarray] = None # mc 리턴 캐시 
        

    def _alpha(self) -> th.Tensor: # 현재 엔트로피 계수 얻기 
        if self.ent_coef_optimizer is not None:
            with th.no_grad():
                return self.log_ent_coef.exp().detach() 
        if isinstance(self.ent_coef, (int, float)):
            return th.tensor(float(self.ent_coef), device=self.device)
        return self.ent_coef_tensor  

    def attach_rnd(self, rnd) -> None: # 외부 RND 네트워크를 연결 
        self.rnd = rnd
        self.rnd.device = str(self.device)
        self._mc_returns = None
        self._mc_cached_size = -1

    # 직선 도로에 해당하는 오프라인 파일만 버퍼에 집어넣기 
    def prefill_from_npz_folder(self, data_dir: str, clip_actions: bool = True) -> int:
   
        files = sorted(glob.glob(os.path.join(data_dir, "route_6*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        act_low = getattr(self.action_space, "low", None)
        act_high = getattr(self.action_space, "high", None)

        n_added, n_files = 0, 0

        for path in files:
            with np.load(path, allow_pickle=False) as d:
                obs = d["observations"].astype(np.float32)
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

            for o, no, a, r, d in zip(obs, nobs, acts, rews, dones): # 전이 하나씩 삽입이라는데 [의문]
                self.replay_buffer.add(
                    o[None, :],                           # (1, obs_dim)
                    no[None, :],                          # (1, obs_dim)
                    a[None, :],                           # (1, act_dim)
                    np.array([float(r)], np.float32),     # (1,)
                    np.array([bool(d)], np.float32),      # (1,)
                    [{"TimeLimit.truncated": False}],     # info list 길이 = n_envs(=1)
                )
            n_added += N
            n_files += 1

        self._mc_returns = None
        self._mc_cached_size = -1
        return n_added

    # 다양한 경로에 해당하는 오프라인 파일만 버퍼에 집어넣기 
    def prefill_from_npz_folder_mclearn(self, data_dir: str, clip_actions: bool = True) -> int: 
   
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        act_low = getattr(self.action_space, "low", None)
        act_high = getattr(self.action_space, "high", None)

        n_added, n_files = 0, 0

        for path in files:
            with np.load(path, allow_pickle=False) as d:
                obs = d["observations"].astype(np.float32)
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

            for o, no, a, r, d in zip(obs, nobs, acts, rews, dones): # 전이 하나씩 삽입이라는데 [의문]
                self.replay_buffer.add(
                    o[None, :],                           # (1, obs_dim)
                    no[None, :],                          # (1, obs_dim)
                    a[None, :],                           # (1, act_dim)
                    np.array([float(r)], np.float32),     # (1,)
                    np.array([bool(d)], np.float32),      # (1,)
                    [{"TimeLimit.truncated": False}],     # info list 길이 = n_envs(=1)
                )
            n_added += N
            n_files += 1

        self._mc_returns = None
        self._mc_cached_size = -1
        return n_added

    # 리플레이 버퍼 안에 있는 값들 중 observations 만 가져와서 학습함 
    def update_rnd_with_critic_batch(self, batch: ReplayBufferSamples) -> th.Tensor:
        if self.rnd is None:
            raise RuntimeError("RND is not attached. Call attach_rnd(rnd) first.")
        # observation 텐서를 현재 학습 장치에 맞게 옮김
        obs = batch.observations # batch_size * obs_dim
        if obs.device != self.device:
            obs = obs.to(self.device)
        # predictor 네트워크 파라미터만 업데이트 
        loss = self.rnd.update(obs)
        return loss.detach()

    # 버퍼 안의 데이터에 대해서 monte carlo return 계산 
    def compute_mc_returns_from_buffer(self, gamma: Optional[float] = None) -> np.ndarray:
        
        rb = self.replay_buffer
        # 현재 버퍼에 실제로 들어있는 transition의 개수 
        size = rb.buffer_size if rb.full else rb.pos
        
        # reward와 done 신호를 버퍼 안에 들어있는 데이터에 대해 가져오기
        rewards = rb.rewards[:size]
        dones = rb.dones[:size]

        if rewards.ndim == 2 and rewards.shape[1] == 1:
            rewards = rewards[:, 0]
            dones = dones[:, 0]

        # 몬테 카를로 리턴에서 감가율 설정 
        if gamma is None:
            gamma = float(self.gamma)

        # 몬테 카를로 리턴 배열 초기화 
        # mc[i] 라면 i번째 스텝에서 시작했을 때의 MC return 
        mc = np.zeros_like(rewards, dtype=np.float32)
        
        # 뒤에서부터 리턴값을 누적해서 MC return을 구해준다
        G = 0.0
        for i in reversed(range(size)):
            if bool(dones[i]):
                G = float(rewards[i])
            else:
                G = float(rewards[i]) + gamma * G
            mc[i] = G

        # 결과 캐싱 
        self._mc_returns = mc
        self._mc_cached_size = size
        self._mc_cached_pos  = rb.pos
        self._mc_cached_full = rb.full
        return mc

    # monte carlo return이 최신 상태인지 확인하고 필요하면 새로 계산 
    def _ensure_mc_returns_current(self) -> None:
        rb = self.replay_buffer
        size = rb.buffer_size if rb.full else rb.pos
        must_recompute = (
            self._mc_returns is None or
            size != self._mc_cached_size or
            rb.pos != self._mc_cached_pos or
            rb.full != self._mc_cached_full
        )
        if must_recompute:
            self.compute_mc_returns_from_buffer(gamma=float(self.gamma))

    # 정책이 뽑은 행동 & 로그 확률 뽑기 
    @th.no_grad()
    def _actor_log_prob(self, obs: th.Tensor):
        a, logp = self.policy.actor.action_log_prob(obs)
        if logp.dim() == 1:
            logp = logp.view(-1, 1)
        return a, logp

    # critic을 offline data로 사전학습하는 함수 
    def pretrain_critic(self, steps: int = 5000, polyak_every: int = 2, update_rnd: bool = True) -> None:
        # actor은 평가 모드로 키고 파라미터 업데이트 안함
        self.policy.actor.eval()
        # Critic은 학습 모드로 파라미터 업데이트 함
        self.policy.critic.train()
    
        for step in range(steps):
            # 리플레이 버퍼에서 배치 뽑기 
            batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
            # 같은 배치로 RND predictor 만 1스텝 업데이트 
            if update_rnd and self.rnd is not None:
                self.update_rnd_with_critic_batch(batch)

            # 역전파 안 흘러가게 막음 
            with th.no_grad():
                # 다음 상태에서 현재 actor 가 내는 행동, 로그 확률 
                next_actions, next_logp = self._actor_log_prob(batch.next_observations)
                # target Q network로 다음 상태의 Q1 , Q2 추정 
                tq1, tq2 = self.policy.critic_target(batch.next_observations, next_actions)
                # 더 보수적인 타겟 사용 
                tmin = th.min(tq1, tq2)
                # sac 타겟 q 계산
                target_q = batch.rewards + (1.0 - batch.dones) * float(self.gamma) * (tmin - self._alpha() * next_logp)

            # 현재 critic으로 q 값 계산 
            q1, q2 = self.policy.critic(batch.observations, batch.actions)
            # target q에 맞추도록 두개의 Q network 학습 
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            if(step % 10000 == 0) :
                print(f"[SAC&RND CRITIC pretrain] critic loss : {critic_loss}")

            # 그냥 최적화 루틴 
            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.critic.optimizer.step()

            if (step + 1) % max(1, polyak_every) == 0:
                polyak_update(self.policy.critic.parameters(), self.policy.critic_target.parameters(), self.tau)

        self.policy.actor.train()
        self.policy.critic.train()

    # behavior cloning 기반 pretraining 
    def pretrain_actor(self, steps: int = 5000) -> None:

        self.policy.actor.train()
        self.policy.critic.eval()

        for p in self.policy.critic.parameters():
            p.requires_grad = False

        for step in range(steps):
            batch = self.replay_buffer.sample(self.batch_size)

            pred_actions = self.policy.actor(batch.observations)
            bc_loss = F.mse_loss(pred_actions, batch.actions)

            self.policy.actor.optimizer.zero_grad()
            bc_loss.backward()
            self.policy.actor.optimizer.step()

            if step % 500 == 0:
                print(f"[BC pretrain] step {step} | loss: {bc_loss.item():.4f}")

        for p in self.policy.critic.parameters():
            p.requires_grad = True

        self.policy.critic.train()
        self.policy.actor.train()  



    def train(self, gradient_steps: int, batch_size: int = 64) -> None:

        # 학습 세팅 
        self.policy.set_training_mode(True)
        optimizers = [self.policy.actor.optimizer, self.policy.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        for gradient_step in range(gradient_steps):
            global_update = self._n_updates + gradient_step
            
            # 1. 리플레이 버퍼에서 배치를 뽑는다. 
            rb = self.replay_buffer
            size = rb.buffer_size if rb.full else rb.pos
            batch_inds = np.random.randint(0, size, size=batch_size)  
            replay_data = rb._get_samples(batch_inds)  

            # Actor 출력값 
            actions_pi, log_prob = self.policy.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.view(-1, 1)
            ent_coef_loss = None
            
            # 엔트로피 계수 업데이트 
            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()  # 수정: 이중 detach 제거
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.detach().cpu().numpy()))
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                
            if self.rnd is None:
                raise RuntimeError("RND is not attached. Call attach_rnd(rnd) before learn().")  
            
            self.update_rnd_with_critic_batch(replay_data)
            
            # target Q값 계산 
            with th.no_grad():
                next_actions, next_log_prob = self.policy.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(
                    self.policy.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_sac = replay_data.rewards + (1.0 - replay_data.dones) * float(self.gamma) * (
                    next_q_values - ent_coef * next_log_prob.view(-1, 1)
                )                
 
                obs_n = self._normalize_tensor(replay_data.observations, self.obs_rms.mean, self.obs_rms.var)
                act_n = self._normalize_tensor(replay_data.actions,       self.act_rms.mean, self.act_rms.var)
                mc_in = th.cat([obs_n, act_n], dim=1)
                
                nov = self.rnd.novelty(mc_in) 
                self._nov_rms.update(nov.detach().cpu().numpy())
                mean_t = th.tensor(self._nov_rms.mean, device=self.device)
                std_t  = th.tensor(self._nov_rms.var ** 0.5, device=self.device)
                nov_z = (nov - mean_t) / (std_t + 1e-6)
                rnd_norm = nov_z.sigmoid()

                # 수정: 안정성 위해 clamp 방식 사용
                w = rnd_norm.clamp(0.05, 0.95)  
                if nov_z.std() < 1e-4:
                    w = w * 0 + 0.5
                
            # critic 업데이트 
            current_q1, current_q2 = self.policy.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * (F.mse_loss(current_q1, target_q_sac) + F.mse_loss(current_q2, target_q_sac))
            critic_losses.append(float(critic_loss.detach().cpu().numpy()))

            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.critic.optimizer.step()

            # actor 업데이트 
            q1_pi, q2_pi = self.policy.critic(replay_data.observations, actions_pi) 
            min_q_pi = th.min(q1_pi, q2_pi)

            # 수정: 정식 deterministic action 계산 방식
            with th.no_grad():
                deterministic_action = self.policy.actor.get_action(
                    replay_data.observations, deterministic=True
                )

            # (3) normalization 후 mcnet 입력
            obs_n = self._normalize_tensor(replay_data.observations, self.obs_rms.mean, self.obs_rms.var)
            act_n = self._normalize_tensor(deterministic_action, self.act_rms.mean, self.act_rms.var)
            g_pi = self.mcnet(th.cat([obs_n, act_n], dim=1)).detach()
            
            # novelty 계산
            w = w.detach()
            if w.dim() == 1:
                w = w.view(-1, 1)

            if float((self._nov_rms.var ** 0.5).mean()) < 1e-8:
                w = w * 0.0 + 0.5
        
            min_q_pi_n = (min_q_pi - min_q_pi.mean()) / (min_q_pi.std() + 1e-6)
            g_pi_n = (g_pi - g_pi.mean()) / (g_pi.std() + 1e-6)
            calibrated_q = (1 - w) * min_q_pi_n + w * g_pi_n

            if global_update % 50 == 0:
                print(f"[Update {global_update}]")
                print("w:", self.tstats(w, "w"))
                print("q_pi:", self.tstats(min_q_pi, "q_pi"))
                print("mc_q:", self.tstats(g_pi, "mc_q"))
                print("logp:", self.tstats(log_prob, "logp"))
                print("alpha:", float(ent_coef.detach().cpu().numpy()))
                print("-" * 50)

            actor_loss = (ent_coef * log_prob - calibrated_q).mean()
            actor_losses.append(float(actor_loss.detach().cpu().numpy()))

            self.policy.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.policy.actor.optimizer.step()

            if global_update % self.target_update_interval == 0:
                polyak_update(self.policy.critic.parameters(), self.policy.critic_target.parameters(), self.tau)

                if hasattr(self, "batch_norm_stats") and hasattr(self, "batch_norm_stats_target"):
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
                # ① NaN 발생 시 즉시 보고
                self._nan_report(global_update, "online",
                    target_q_sac=target_q_sac, current_q1=current_q1, current_q2=current_q2,
                    critic_loss=critic_loss, log_prob=log_prob, min_q_pi=min_q_pi,
                    g_pi=g_pi, calibrated_q=calibrated_q, actor_loss=actor_loss)

                # ② 1000스텝마다 간단 요약
                self._tick_log(global_update, 1000,
                    f"[onl] up {global_update} | crit:{critic_loss.item():.4f} | actor:{actor_loss.item():.4f}")

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", float(np.mean(ent_coef_losses)))

    def online_learn(self, total_timesteps=300_000, tb_log_name: str = "", log_interval: int = 10, callback=None):
  
        return self.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            log_interval=log_interval,
            callback=callback,
        )
