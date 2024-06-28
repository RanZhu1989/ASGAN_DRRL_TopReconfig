from modules import *
from utils.misc import get_time, event_logger, data_logger
from utils.wrappers import obs_wrapper, action_wrapper

from tqdm import tqdm


class PPO_Clip_Agent():
    """ Since the discrete actions have been redefined as {0,1} by env, we can simply represent the action by a number. """

    def __init__(
        self,
        episode_recorder: object,
        actor_network: torch.nn,
        critic_network: torch.nn,
        actor_optimizer: torch.optim,
        critic_optimizer: torch.optim,
        gamma: float = 0.9,
        advantage_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        train_iters: int = 10,
        device: torch.device = torch.device("cpu")) -> None:

        self.device = device
        self.episode_recorder = episode_recorder
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.clip_epsilon = clip_epsilon
        self.train_iters = train_iters
        self.gamma = gamma
        self.advantage_lambda = advantage_lambda

    def get_action(self, obs: np.ndarray) -> int:
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        dist = self.actor_network(
            obs)  # NN_outputs -softmax-> Action distribution
        dist = F.softmax(dist, dim=0)
        action_probs = torch.distributions.Categorical(dist)
        picked_action = action_probs.sample()

        return picked_action.item()

    def calculate_log_prob(self, obs: torch.tensor,
                           action: torch.tensor) -> torch.tensor:
        dist = F.softmax(self.actor_network(obs), dim=1)
        log_prob = torch.log(dist.gather(1, action))

        return log_prob

    def calculate_advantage(self, td_error: torch.tensor) -> torch.tensor:
        """ The advantage function is calculated by the TD error. """
        td_error = td_error.cpu().detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_error[::-1]:
            advantage = self.gamma * self.advantage_lambda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.tensor(np.array(advantage_list),
                                 dtype=torch.float32).to(self.device)

        return advantage

    def train_policy(self) -> None:
        obs, action, reward, next_obs, done, = self.episode_recorder.get_trajectory(
        )
        TD_target = reward + self.gamma * self.critic_network(next_obs) * (1 -
                                                                           done)
        TD_error = TD_target - self.critic_network(obs)
        advantage = self.calculate_advantage(TD_error)

        old_log_prob = self.calculate_log_prob(obs, action).detach(
        )  # Freeze the log_prob obtained by the current policy

        for _ in range(self.train_iters):
            critic_loss = torch.mean(
                F.mse_loss(TD_target.detach(), self.critic_network(obs)))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            log_prob = self.calculate_log_prob(obs, action)
            ratio = torch.exp(log_prob -
                              old_log_prob)  # pi_theta / pi_theta_old
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon,
                                        1 + self.clip_epsilon)
            actor_loss = torch.mean(
                -torch.min(ratio * advantage, clipped_ratio * advantage))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


class Actor_Network(torch.nn.Module):

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(Actor_Network, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 64)
        self.fc2 = torch.nn.Linear(64, action_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Critic_Network(torch.nn.Module):

    def __init__(self, obs_dim: int) -> None:
        super(Critic_Network, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Episode_Recorder():

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device = device
        self.reset()

    def append(self, obs: np.ndarray, action: int, reward: float,
               next_obs: np.ndarray, done: bool) -> None:
        obs = torch.tensor(np.array([obs]), dtype=torch.float32).to(self.device)
        action = torch.tensor(np.array([[action]]),
                              dtype=torch.int64).to(self.device)
        reward = torch.tensor(np.array([[reward]]),
                              dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(np.array([next_obs]),
                                dtype=torch.float32).to(self.device)
        done = torch.tensor(np.array([[done]]),
                            dtype=torch.float32).to(self.device)
        self.trajectory["obs"] = torch.cat((self.trajectory["obs"], obs))
        self.trajectory["action"] = torch.cat(
            (self.trajectory["action"], action))
        self.trajectory["reward"] = torch.cat(
            (self.trajectory["reward"], reward))
        self.trajectory["next_obs"] = torch.cat(
            (self.trajectory["next_obs"], next_obs))
        self.trajectory["done"] = torch.cat((self.trajectory["done"], done))

    def get_trajectory(
        self
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor,
               torch.tensor, torch.tensor]:

        return self.trajectory["obs"], self.trajectory["action"], \
            self.trajectory["reward"], self.trajectory["next_obs"], self.trajectory["done"]

    def reset(self) -> None:
        """ Clear the trajectory when begin a new episode."""
        self.trajectory = {
            "obs": torch.tensor([], dtype=torch.float32).to(self.device),
            "action": torch.tensor([], dtype=torch.int64).to(self.device),
            "reward": torch.tensor([], dtype=torch.float32).to(self.device),
            "next_obs": torch.tensor([], dtype=torch.float32).to(self.device),
            "done": torch.tensor([], dtype=torch.float32).to(self.device)
        }


class RL_Manager():

    def __init__(self,
                 env_args,
                 rl_args,
                 seed=0,
                 logger_output_path=None,
                 saving_folder=None,
                 device='cpu',
                 log_event=True):
        '''
        episode_num:int = 1000,
        actor_lr:float = 1e-4,
        critic_lr:float = 1e-3,
        gamma:float = 0.9,
        advantage_lambda:float = 0.95,
        clip_epsilon:float = 0.2,
        train_iters:int = 10,
        
        '''
        self.rl_args = rl_args
        self.seed = seed
        self.device = torch.device(device)
        self.saving_folder = saving_folder
        self.last_saving_policy_path = None
        self.env = gym.make(**env_args)
        self.obs_dim = gym.spaces.utils.flatdim(self.env.observation_space)
        self.action_dim = 2**self.env.action_space.n  # 2^num_switch

        self.seed = seed
        _, _ = self.env.reset(seed=self.seed)

        self.episode_num = self.rl_args['episode_num']
        self.log_output_path = logger_output_path
        self.log_event = log_event
        if self.log_event:
            self.event_logger = event_logger(
                log_output_path=self.log_output_path)
        self.data_logger = data_logger(log_output_path=self.log_output_path)

        self.eval_iterations = self.rl_args['eval_iterations']

        self.reset()
        self.index_episode = 0

        self.round = 0

        pass

    def reset(self):
        episode_recorder = Episode_Recorder(device=self.device)
        actor_network = Actor_Network(self.obs_dim,
                                      self.action_dim).to(self.device)
        actor_optimizer = torch.optim.Adam(actor_network.parameters(),
                                           lr=self.rl_args['actor_lr'])
        critic_network = Critic_Network(self.obs_dim).to(self.device)
        critic_optimizer = torch.optim.Adam(critic_network.parameters(),
                                            lr=self.rl_args['critic_lr'])
        self.agent = PPO_Clip_Agent(
            episode_recorder=episode_recorder,
            actor_network=actor_network,
            critic_network=critic_network,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            gamma=self.rl_args['gamma'],
            advantage_lambda=self.rl_args['advantage_lambda'],
            clip_epsilon=self.rl_args['clip_epsilon'],
            train_iters=self.rl_args['train_iters'],
            device=self.device)
        pass

    def train_episode(self, train_scenarios_id_list) -> float:
        options = {
            "Candidate_Scenario_ID_List": train_scenarios_id_list,
            "Specific_Scenario_ID": None,
            "External_RNG": None
        }
        total_reward = 0
        self.agent.episode_recorder.reset()
        obs, info = self.env.reset(options=options)
        scn_id = info["Chosen_Scenario_ID"]
        if self.log_event:
            self.event_logger.logger.info(
                "The training Scenario_ID is {}".format(scn_id))
        total_reward = 0

        step = 0
        train_log_name = 'Round_' + str(self.round) + '_Train'
        self.data_logger.reset_user_label(train_log_name)

        while True:
            action = self.agent.get_action(obs_wrapper(obs))
            if self.log_event:
                self.event_logger.logger.info(
                    "Given profiles @S[{}] = {}".format(step,
                                                        info['ENV_Profile']))
                self.event_logger.logger.info(
                    "Agent action at S[{}] is A[{}] = {}".format(
                        step, step + 1,
                        action_wrapper(action, self.env.action_space.n)))

            next_obs, reward, done, _, info = self.env.step(
                action_wrapper(action, self.env.action_space.n))
            if self.log_event:
                self.event_logger.logger.info("Event at S[{}] is '{}'".format(
                    step + 1, info["Event_Log"]))
                self.event_logger.logger.info(
                    "Reward R[{}] is {}. Results @S[{}] = {}.".format(
                        step + 1, reward, step + 1, info['Penalties']))
            self.agent.episode_recorder.append(obs_wrapper(obs), action, reward,
                                               obs_wrapper(next_obs), done)
            total_reward += reward
            obs = next_obs
            step += 1
            '''
            log : [#episode, #scn_id, #step, reward, #event_log, v_violation, p_loss, q_loss, p_balance, q_balance]
            '''
            log = [
                self.index_episode, scn_id, step, reward, info['Event_Code'],
                info['Penalties']['V_Violation'], info['Penalties']['P_Loss'],
                info['Penalties']['Q_Loss'], info['Penalties']['P_Unbalance'],
                info['Penalties']['Q_Unbalance']
            ]
            self.data_logger.save_to_csv(log)
            if done:
                break

        self.agent.train_policy()

        return total_reward

    def train(self, scenarios_id_list, save_policy: bool = True) -> None:
        log_name = 'Round_' + str(self.round) + '_Train'

        if self.log_event:
            self.event_logger.reset_user_label(log_name)

        self.index_episode = 0  #NOTE For continuous training
        self.data_logger.reset_user_label(log_name)
        self.data_logger.save_to_csv([
            '#episode', '#scn_id', '#action_step', 'reward', '#event_log',
            'v_violation', 'p_loss', 'q_loss', 'p_unbalance', 'q_unbalance'
        ])

        reward_list = []
        with tqdm(total=self.episode_num, desc='Training') as pbar:
            for idx_episode in range(self.episode_num):
                if self.log_event:
                    self.event_logger.logger.info(
                        f"=============== Episode {idx_episode+1:d} of {self.episode_num:d} ================="
                    )
                episode_reward = self.train_episode(scenarios_id_list)
                pbar.set_postfix({"Reward": (episode_reward)})
                pbar.update(1)
                reward_list.append(episode_reward)
                self.index_episode += 1
                pass

        if save_policy:
            self.last_saving_policy_path = self.save_policy(self.saving_folder)

        return np.mean(reward_list)

    def eval(self,
             scenarios_id_list,
             multi_eval=False,
             policy_path=None,
             user_label='Eval'):
        # Evaluate the reward in all environments specified in reval_scenarios_id_list
        eval_log_name = 'Round_' + str(self.round) + '_' + user_label
        self.data_logger.reset_user_label(eval_log_name)
        self.data_logger.save_to_csv([
            '#env_id', '#iter', '#action_step', 'reward', '#event_log',
            'v_violation', 'p_loss', 'q_loss', 'p_unbalance', 'q_unbalance'
        ])

        # load the policy
        if policy_path is not None:
            self.load_policy(policy_path)

        # Test each scenario in eval_scenarios_id_list, test each scenario eval_iterations times, and save the average reward
        eval_id_list = scenarios_id_list

        saved_reward = []

        if multi_eval:
            num_eval_iter = self.eval_iterations
        else:
            num_eval_iter = 1

        with tqdm(total=len(eval_id_list), desc=user_label) as pbar:
            for eval_id in eval_id_list:
                total_reward = 0
                for eval_iter_idx in range(num_eval_iter):
                    obs, info = self.env.reset(
                        options={
                            "Candidate_Scenario_ID_List": eval_id_list,
                            "Specific_Scenario_ID": eval_id,
                            "External_RNG": None
                        })
                    agent_step_reward = []
                    for step in range(self.env.T - 1):
                        s0 = obs_wrapper(obs)
                        a = self.agent.get_action(s0)
                        s, reward, _, _, info = self.env.step(
                            action_wrapper(a, self.env.action_space.n))
                        agent_step_reward.append(reward)
                        s0 = s
                        log = [
                            eval_id, eval_iter_idx, step + 1, reward,
                            info['Event_Code'],
                            info['Penalties']['V_Violation'],
                            info['Penalties']['P_Loss'],
                            info['Penalties']['Q_Loss'],
                            info['Penalties']['P_Unbalance'],
                            info['Penalties']['Q_Unbalance']
                        ]
                        self.data_logger.save_to_csv(log)

                    total_reward += np.sum(agent_step_reward)
                average_reward = total_reward / self.eval_iterations

                pbar.set_postfix({'Avg.R': average_reward})
                pbar.update(1)
                saved_reward.append(average_reward)
                pass

        # Return two lists for updating labels in the database
        return saved_reward, np.mean(saved_reward)

    def save_policy(self, saving_folder) -> None:
        file_name = 'Round_' + str(
            self.round) + '_PPO_Actor_' + get_time() + '.pth'
        torch.save(self.agent.actor_network.state_dict(),
                   saving_folder / file_name)
        return saving_folder / file_name

    def load_policy(self, loading_path) -> None:
        self.agent.actor_network.load_state_dict(torch.load(loading_path))
        pass
