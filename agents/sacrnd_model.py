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
import torch.nn as nn
import pickle

class MCNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        print(f"dims:{dims}")
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        
        layers.append(nn.Linear(dims[-1], 1))  # output: MC return
        self.model = nn.Sequential(*layers)
        self.optimizer = th.optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x):
        return self.model(x)


class RunningMeanStd:
    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
    def update(self, x: th.Tensor):
        # x: (B,1)
        
        if isinstance(x, th.Tensor):
            x_np = x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            x_np = x
        else:
            raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(x)}")
        
        batch_mean = x_np.mean()
        batch_var = x_np.var(ddof=0)
        batch_count = x_np.shape[0]
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
        policy: str = "MlpPolicy", 
        learning_starts: int = 0, 
        target_entropy = -3,
        **sac_kwargs,
    ):
        if "learning_starts" not in sac_kwargs:
            sac_kwargs["learning_starts"] = learning_starts

        super().__init__(policy, env, target_entropy=target_entropy, **sac_kwargs) 
        print(f"target entropy : {target_entropy}")
        self._nov_rms = RunningMeanStd()
        self.rnd = None # rnd 모듈 정의 
        self.rnd_update_every = 5
        self.rnd_update_steps = 1
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        print(self.device)
        # 몬테 카를로 return 캐시를 위한 변수 
        self._mc_cached_size: int = -1  # 유효 버퍼 길이 
        self._mc_cached_pos  = -1
        self._mc_cached_full = False
        self.ent_coef = 0.5
        self.mc_targets = []
        self.mcnet = MCNet(input_dim=self.observation_space.shape[0] + self.action_space.shape[0]).to(self.device)
        checkpoint = th.load("mcnet/mcnet_pretrained.pth")
        self.mcnet.load_state_dict(checkpoint["model_state_dict"])
        self.mcnet.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        
        self.obs_rms = RunningMeanStd()
        self.act_rms = RunningMeanStd()
        self.gamma = 0.99
        self.calibrated = 0

    def _alpha(self) -> th.Tensor: # 현재 엔트로피 계수 얻기 
        if self.ent_coef_optimizer is not None:
            with th.no_grad():
                return self.log_ent_coef.exp().detach() 
        if isinstance(self.ent_coef, (int, float)):
            return th.tensor(float(self.ent_coef), device=self.device)
        return self.ent_coef_tensor  
   
    # 몬테 카를로 리턴 계산
    def compute_mc_returns(self, gamma: float, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        mc = np.zeros_like(rewards, dtype=np.float32)
        G = 0.0
        for i in reversed(range(len(rewards))):
            if bool(dones[i]):
                G = float(rewards[i])
            else:
                G = float(rewards[i]) + gamma * G
            mc[i] = G
        return mc

    
    # SACOfflineOnline 클래스에 메서드 추가
    def _normalize_tensor(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> th.Tensor:
        eps = 1e-8

        def to_tensor_safe(val):
            if isinstance(val, th.Tensor):
                return val.clone().detach().to(dtype=th.float32, device=self.device)
            else:
                return th.tensor(val, dtype=th.float32, device=self.device)

        x_t    = to_tensor_safe(x)
        mean_t = to_tensor_safe(mean)
        var_t  = to_tensor_safe(var)
        std_t  = th.sqrt(var_t + eps)

        return (x_t - mean_t) / std_t


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
                self.obs_rms.update(th.tensor(o).unsqueeze(0))  # shape (1, obs_dim)
                self.act_rms.update(th.tensor(a).unsqueeze(0))  # shape (1, act_dim)
            n_added += N
            n_files += 1
     
        # mc_returns = self.compute_mc_returns(
        #     gamma=self.gamma,
        #     rewards=rews.flatten(),
        #     dones=dones.flatten()
        # )
        # self.mc_targets.extend(mc_returns.tolist())
        return n_added

    # 다양한 경로에 해당하는 오프라인 파일만 버퍼에 집어넣기 
    def prefill_from_npz_folder_mclearn(self, data_dir: str, clip_actions: bool = True) -> int: 
        self.mc_targets = [] 
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        act_low = getattr(self.action_space, "low", None)
        act_high = getattr(self.action_space, "high", None)

        n_added, n_files = 0, 0
        all_rews = []
        all_dones = []
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

            for o, no, a, r, d in zip(obs, nobs, acts, rews, dones): # 전이 하나씩 삽입이라는데 
                self.replay_buffer.add(
                    o[None, :],                           # (1, obs_dim)
                    no[None, :],                          # (1, obs_dim)
                    a[None, :],                           # (1, act_dim)
                    np.array([float(r)], np.float32),     # (1,)
                    np.array([bool(d)], np.float32),      # (1,)
                    [{"TimeLimit.truncated": False}],     # info list 길이 = n_envs(=1)
                )
                
                
                all_rews.append(float(r))
                all_dones.append(bool(d))
                self.obs_rms.update(th.tensor(o).unsqueeze(0))  # shape (1, obs_dim)
                self.act_rms.update(th.tensor(a).unsqueeze(0))  # shape (1, act_dim)
            n_added += N
            n_files += 1

        mc_returns = self.compute_mc_returns(
            gamma=self.gamma,
            rewards=np.array(all_rews),
            dones=np.array(all_dones)
        )
        self.mc_targets = mc_returns.tolist()
        return n_added

    # 리플레이 버퍼 안에 있는 값들 중 observations 만 가져와서 학습함 
    def update_rnd_with_critic_batch(self, batch: ReplayBufferSamples) -> th.Tensor:
        if self.rnd is None:
            raise RuntimeError("RND is not attached. Call attach_rnd(rnd) first.")

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device)

        #[!! 배치별 정규화 중 !!]
        obs_n = self._normalize_tensor(obs, self.obs_rms.mean, self.obs_rms.var)
        act_n = self._normalize_tensor(act, self.act_rms.mean, self.act_rms.var)
        
        x = th.cat([obs_n, act_n], dim=1)
        loss = self.rnd.update(x)

        return loss.detach()


    def train_mcnet_from_buffer(self, epochs=5, batch_size=512):
        self.mcnet.train()
        dataset_size = len(self.mc_targets)
        # print(f"dataset size : {dataset_size}")
        buffer_len = self.replay_buffer.pos if not self.replay_buffer.full else self.replay_buffer.buffer_size
        # print(f"buffer len : {buffer_len}")
        assert dataset_size == buffer_len, f"mc_targets size mismatch ({dataset_size} vs {buffer_len})"

        for epoch in range(epochs):
            perm = np.random.permutation(dataset_size)
            for i in range(0, dataset_size, batch_size):
                idxs = perm[i:i + batch_size]
                obs = self.replay_buffer.observations[idxs].squeeze(1)
                act = self.replay_buffer.actions[idxs].squeeze(1)

                target = th.tensor(np.array(self.mc_targets)[idxs], dtype=th.float32).unsqueeze(-1).to(self.device)
                # print("obs.shape:", obs.shape)
                # print("act.shape:", act.shape)
                # print("target.shape:", target.shape)
                obs_n = self._normalize_tensor(obs, self.obs_rms.mean, self.obs_rms.var)
                act_n = self._normalize_tensor(act, self.act_rms.mean, self.act_rms.var)
                x = th.cat([obs_n, act_n], dim=1)
                # print("obs.shape:", obs.shape)
                # print("act.shape:", act.shape)
                # print("target.shape:", target.shape)

                pred = self.mcnet(x)
                loss = F.mse_loss(pred, target)

                self.mcnet.optimizer.zero_grad()
                loss.backward()
                self.mcnet.optimizer.step()

            print(f"[MCNet] Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
            
    def save_mcnet_pth(self, path="mcnet_pretrained.pth"):
        th.save({
            'model_state_dict': self.mcnet.state_dict(),
            'optimizer_state_dict': self.mcnet.optimizer.state_dict()
        }, path)
        print(f"[MCNet] Saved as PyTorch .pth at {path}")
        

    def save_mcnet_pickle(self, path="mcnet_pretrained.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self.mcnet, f)
        print(f"[MCNet] Saved as Pickle .pkl at {path}")
        
            
    def load_mcnet(self, path: str):
        checkpoint = th.load(path, map_location=self.device)
        self.mcnet.load_state_dict(checkpoint['model_state_dict'])
        self.mcnet.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[MCNet] Loaded from {path}")

    def tstats(self, tensor: th.Tensor, name: str = "") -> str:
        """
        텐서의 요약 통계를 문자열로 반환.
        Args:
            tensor (th.Tensor): 통계를 출력할 텐서
            name (str): 변수 이름(옵션)
        Returns:
            str: 정리된 통계 문자열
        """
        if not isinstance(tensor, th.Tensor):
            return f"{name}: Not a tensor"

        tensor = tensor.detach().cpu()
        stats = {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "shape": list(tensor.shape)
        }

        return (
            f"{name} | shape: {stats['shape']} | "
            f"mean: {stats['mean']:.4f} | std: {stats['std']:.4f} | "
            f"min: {stats['min']:.4f} | max: {stats['max']:.4f}"
        )

    def _nan_report(self, step: int, phase: str = "train", **tensors):
        """
        텐서들 중 NaN이 있는지 탐지하고 있으면 관련 정보를 출력.

        Args:
            step (int): 현재 스텝 수
            phase (str): 학습 단계 이름 (예: "train", "online", "pretrain" 등)
            tensors (dict): 검사할 텐서들 (키=이름, 값=텐서)
        """
        for name, tensor in tensors.items():
            if not isinstance(tensor, th.Tensor):
                continue
            if th.isnan(tensor).any() or th.isinf(tensor).any():
                print(f"[NaN DETECTED] step {step} | phase: {phase} | tensor: '{name}'")
                print(f"→ Shape: {tuple(tensor.shape)}")
                print(f"→ Values (sample): {tensor.flatten()[:5].tolist()}")
                print(f"→ Stats: mean={tensor.float().mean().item():.4f}, std={tensor.float().std().item():.4f}")
                # Optional: raise error to halt training
                raise ValueError(f"NaN detected in tensor '{name}' during {phase} at step {step}.")

    
    def _tick_log(self, step: int, interval: int, message: str):
        """
        주어진 step이 interval의 배수일 때만 message를 출력하는 헬퍼 함수.
        
        Args:
            step (int): 현재 step 또는 업데이트 수
            interval (int): 로그 출력 간격
            message (str): 출력할 메시지
        """
        if step % interval == 0:
            print(message)

    # 정책이 뽑은 행동 & 로그 확률 뽑기 
    @th.no_grad()
    def _actor_log_prob(self, obs: th.Tensor):
        a, logp = self.policy.actor.action_log_prob(obs)
        if logp.dim() == 1:
            logp = logp.view(-1, 1)
        return a, logp

    # critic을 offline data로 사전학습하는 함수 
    def pretrain_critic(self, epochs: int = 60, batch_size: int = 256) -> None:
        """
        Critic을 Monte Carlo return (self.mc_targets)에 맞춰 supervised pretrain.
        - Actor는 freeze
        - Critic만 학습
        """
        self.policy.actor.eval()
        self.policy.critic.train()

        dataset_size = len(self.mc_targets)
        buffer_len = self.replay_buffer.pos if not self.replay_buffer.full else self.replay_buffer.buffer_size
        assert dataset_size == buffer_len, \
            f"Size mismatch: mc_targets({dataset_size}) vs buffer({buffer_len})"

        mc_targets_np = np.array(self.mc_targets, dtype=np.float32)

        for epoch in range(epochs):
            perm = np.random.permutation(dataset_size)

            for i in range(0, dataset_size, batch_size):
                idxs = perm[i:i + batch_size]

                # (s,a) from buffer
                obs = th.tensor(
                    self.replay_buffer.observations[idxs].squeeze(1),
                    dtype=th.float32,
                    device=self.device
                )
                act = th.tensor(
                    self.replay_buffer.actions[idxs].squeeze(1),
                    dtype=th.float32,
                    device=self.device
                )

                # Monte Carlo return as supervised target
                target_q = th.tensor(mc_targets_np[idxs], dtype=th.float32).unsqueeze(-1).to(self.device)

                # Q(s,a) from critic
                q1, q2 = self.policy.critic(obs, act)

                # supervised loss
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                self.policy.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.policy.critic.optimizer.step()

            print(f"[Critic-MC] Epoch {epoch+1}/{epochs} | Loss: {critic_loss.item():.4f}")

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
                w = rnd_norm.clamp(0.05, 0.9)  
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
                deterministic_action = self.policy._predict(replay_data.observations, deterministic=True)

            # (3) normalization 후 mcnet 입력
            obs_n = self._normalize_tensor(replay_data.observations, self.obs_rms.mean, self.obs_rms.var)
            act_n = self._normalize_tensor(deterministic_action, self.act_rms.mean, self.act_rms.var)
            g_pi = self.mcnet(th.cat([obs_n, act_n], dim=1)).detach()
            
            calibrate_q = th.minimum(min_q_pi, g_pi)
            if calibrate_q == g_pi: 
                self.calibrated += 1
                print(f"calibrated_count : {self.calibrated}")
            
             
            # novelty 계산
            w = w.detach()
            if w.dim() == 1:
                w = w.view(-1, 1)

            # if float((self._nov_rms.var ** 0.5).mean()) < 1e-8:
            #     w = w * 0.0 + 0.5
        
            # min_q_pi_n = (min_q_pi - min_q_pi.mean()) / (min_q_pi.std() + 1e-6)
            # g_pi_n = (g_pi - g_pi.mean()) / (g_pi.std() + 1e-6)
         
            calibrated_q = (1 - w) * min_q_pi + w * calibrated_q

            if global_update % 5000 == 0:
                print(f"[Update {global_update}]")
                print("w:", self.tstats(w, "w"))
                print("q_pi:", self.tstats(min_q_pi, "q_pi"))
                print("mc_q:", self.tstats(g_pi, "mc_q"))
                print("logp:", self.tstats(log_prob, "logp"))
                print("alpha:", float(ent_coef.detach().cpu().numpy()))
                print("cal_q:", self.tstats(calibrated_q, "calibrated_q"))
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
