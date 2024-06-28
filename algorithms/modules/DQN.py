from modules import *
from utils.misc import get_time, event_logger, data_logger
from utils.wrappers import obs_wrapper, action_wrapper

from tqdm import tqdm


class DQN_Agent():
    """Vanilla DQN Agent"""

    def __init__(
        self,
        Q_func: torch.nn.Module,
        action_dim: int,
        optimizer: torch.optim.Optimizer,
        replay_buffer: collections.deque,
        replay_start_size: int,
        batch_size: int,
        replay_frequent: int,
        target_sync_frequent: int,  # The frequency of synchronizing the parameters of the two Q networks
        epsilon: float = 0.1,  # Initial epsilon
        mini_epsilon: float = 0.01,  # Minimum epsilon
        explore_decay_rate: float = 0.0001,  # Decay rate of epsilon
        gamma: float = 0.9,
        device: torch.device = torch.device("cpu")
    ) -> None:

        self.device = device
        self.action_dim = action_dim

        self.exp_counter = 0

        self.replay_buffer = replay_buffer
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.replay_frequent = replay_frequent

        self.target_sync_frequent = target_sync_frequent
        """Two Q functions (mian_Q and target_Q) are used to stabilize the training process. 
            Since they share the same network structure, we can use copy.deepcopy to copy the main_Q to target_Q for initialization."""
        self.main_Q_func = Q_func
        self.target_Q_func = copy.deepcopy(Q_func)

        self.optimizer = optimizer

        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.gamma = gamma
        self.explore_decay_rate = explore_decay_rate
        pass

    def get_target_action(self, obs: np.ndarray) -> torch.tensor:
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        Q_list = self.target_Q_func(obs)
        action = torch.argmax(Q_list)
        return action

    def get_behavior_action(self, obs: np.ndarray) -> int:
        """Here, a simple epsilon decay is used to balance the exploration and exploitation.
            The epsilon is decayed from epsilon_init to mini_epsilon."""
        self.epsilon = max(self.mini_epsilon,
                           self.epsilon - self.explore_decay_rate)

        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.get_target_action(obs).item()

        return action

    """Here, we defined a function to synchronize the parameters of the main_Q and target_Q."""

    def sync_target_Q_func(self) -> None:
        for target_params, main_params in zip(self.target_Q_func.parameters(),
                                              self.main_Q_func.parameters()):
            target_params.data.copy_(main_params.data)

    def batch_Q_approximation(self, batch_obs: torch.tensor,
                              batch_action: torch.tensor,
                              batch_reward: torch.tensor,
                              batch_next_obs: torch.tensor,
                              batch_done: torch.tensor) -> None:

        batch_current_Q = torch.gather(self.main_Q_func(batch_obs), 1,
                                       batch_action).squeeze(1)
        batch_TD_target = batch_reward + (
            1 - batch_done
        ) * self.gamma * self.target_Q_func(batch_next_obs).max(1)[0]
        loss = torch.mean(F.mse_loss(batch_current_Q, batch_TD_target))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def Q_approximation(self, obs: np.ndarray, action: int, reward: float,
                        next_obs: np.ndarray, done: bool) -> None:

        self.exp_counter += 1
        self.replay_buffer.append((obs, action, reward, next_obs, done))

        if len(
                self.replay_buffer
        ) > self.replay_start_size and self.exp_counter % self.replay_frequent == 0:
            self.batch_Q_approximation(
                *self.replay_buffer.sample(self.batch_size))

        # Synchronize the parameters of the two Q networks every target_update_frequent steps
        if self.exp_counter % self.target_sync_frequent == 0:
            self.sync_target_Q_func()


class Q_Network(torch.nn.Module):
    """You can define your own network structure here."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(Q_Network, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 64)
        self.fc2 = torch.nn.Linear(64, action_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class ReplayBuffer():

    def __init__(
        self, capacity: int,
        device: torch.device = torch.device("cpu")) -> None:
        self.capacity = capacity
        self.device = device
        self.buffer = collections.deque(maxlen=self.capacity)

    def reset(self):
        self.buffer = collections.deque(maxlen=self.capacity)

    def append(self, exp_data: tuple) -> None:
        self.buffer.append(exp_data)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor,
               torch.tensor]:
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(
            *mini_batch)

        obs_batch = torch.tensor(np.array(obs_batch),
                                 dtype=torch.float32,
                                 device=self.device)

        action_batch = torch.tensor(action_batch,
                                    dtype=torch.int64,
                                    device=self.device)
        action_batch = action_batch.unsqueeze(1)

        reward_batch = torch.tensor(reward_batch,
                                    dtype=torch.float32,
                                    device=self.device)
        next_obs_batch = torch.tensor(np.array(next_obs_batch),
                                      dtype=torch.float32,
                                      device=self.device)
        done_batch = torch.tensor(done_batch,
                                  dtype=torch.float32,
                                  device=self.device)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self) -> int:
        return len(self.buffer)


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
        lr_args = {
        episode_num:int = 1000,
        test_iterations:int = 5,
        eval_iterations:int = 5,
        lr:float = 1e-3,
        gamma:float = 0.9,
        epsilon:float = 0.1,
        mini_epsilon:float = 0.01,
        explore_decay_rate:float = 0.0001,
        buffer_capacity:int = 2000,
        replay_start_size:int = 200,
        replay_frequent:int = 4,
        target_sync_frequent:int = 200,
        batch_size:int = 32
        }
        '''
        self.rl_args = rl_args
        self.seed = seed
        self.device = torch.device(device)
        self.saving_folder = saving_folder
        self.last_saving_policy_path = None
        self.env = gym.make(**env_args)
        self.obs_dim = gym.spaces.utils.flatdim(self.env.observation_space)
        self.action_dim = 2**self.env.action_space.n  # 2^num_switch

        self.episode_num = self.rl_args['episode_num']
        self.log_output_path = logger_output_path
        self.log_event = log_event
        if self.log_event:
            self.event_logger = event_logger(
                log_output_path=self.log_output_path)
        self.data_logger = data_logger(log_output_path=self.log_output_path)

        self.eval_iterations = 1  # Due to the deterministic policy of Q-learning

        self.seed = seed
        _, _ = self.env.reset(seed=self.seed)

        self.reset()

        self.index_episode = 0

        self.round = 0
        pass

    def reset(self):
        Q_func = Q_Network(self.obs_dim, self.action_dim).to(self.device)
        optimizer = torch.optim.Adam(Q_func.parameters(), lr=self.rl_args['lr'])
        self.buffer = ReplayBuffer(capacity=self.rl_args['buffer_capacity'],
                                   device=self.device)

        #NOTE! The agent can be inherited from the previous training process
        # Set a large capacity and the buffer can be automatically inherited from the previous training process
        self.agent = DQN_Agent(
            Q_func=Q_func,
            action_dim=self.action_dim,
            optimizer=optimizer,
            replay_buffer=self.buffer,
            replay_start_size=self.rl_args['replay_start_size'],
            batch_size=self.rl_args['batch_size'],
            replay_frequent=self.rl_args['replay_frequent'],
            target_sync_frequent=self.rl_args['target_sync_frequent'],
            epsilon=self.rl_args['epsilon'],
            mini_epsilon=self.rl_args['mini_epsilon'],
            explore_decay_rate=self.rl_args['explore_decay_rate'],
            gamma=self.rl_args['gamma'],
            device=self.device)

    def train_episode(self, train_scenarios_id_list) -> float:
        options = {
            "Candidate_Scenario_ID_List": train_scenarios_id_list,
            "Specific_Scenario_ID": None,
            "External_RNG": None
        }
        total_reward = 0
        obs, info = self.env.reset(options=options)
        scn_id = info["Chosen_Scenario_ID"]
        if self.log_event:
            self.event_logger.logger.info(
                "The training Scenario_ID is {}".format(scn_id))
        obs = obs_wrapper(obs)
        step = 0
        train_log_name = 'Round_' + str(self.round) + '_Train'
        self.data_logger.reset_user_label(train_log_name)
        while True:
            #NOTE: Use wrapper, dict obs -> reshape
            action = self.agent.get_behavior_action(obs)
            if self.log_event:
                self.event_logger.logger.info(
                    "Given profiles @S[{}] = {}".format(step,
                                                        info['ENV_Profile']))
                self.event_logger.logger.info(
                    "Agent action at S[{}] is A[{}] = {}".format(
                        step, step + 1,
                        action_wrapper(action, self.env.action_space.n)))
            #NOTE: Use wrapper, int action -> bit action
            next_obs, reward, terminated, truncated, info = self.env.step(
                action_wrapper(action, self.env.action_space.n))
            if self.log_event:
                self.event_logger.logger.info("Event at S[{}] is '{}'".format(
                    step + 1, info["Event_Log"]))
                self.event_logger.logger.info(
                    "Reward R[{}] is {}. Results @S[{}] = {}.".format(
                        step + 1, reward, step + 1, info['Penalties']))
            next_obs = obs_wrapper(next_obs)
            done = terminated or truncated
            total_reward += reward
            self.agent.Q_approximation(obs, action, reward, next_obs, done)
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

        with tqdm(total=self.episode_num, desc='Training DQN') as pbar:
            for idx_episode in range(self.episode_num):
                if self.log_event:
                    self.event_logger.logger.info(
                        f"=============== Episode {idx_episode+1:d} of {self.episode_num:d} ================="
                    )
                episode_reward = self.train_episode(scenarios_id_list)
                pbar.set_postfix({
                    'reward': episode_reward,
                })
                reward_list.append(episode_reward)
                pbar.update(1)
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
        #Evaluate the reward in all environments specified in reval_scenarios_id_list
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
                        a = self.agent.get_target_action(s0)
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
            self.round) + '_DQN_Qnetwork_' + get_time() + '.pth'
        torch.save(self.agent.target_Q_func.state_dict(),
                   saving_folder / file_name)
        return saving_folder / file_name

    def load_policy(self, loading_path) -> None:
        self.agent.target_Q_func.load_state_dict(
            torch.load(loading_path))  # for DQN, only target_Q_func is needed
        pass
