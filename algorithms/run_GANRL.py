import os
import yaml
import random

from pathlib import Path

import numpy as np
import torch

from utils.misc import event_logger, ensure_directory, get_time
from utils.data import split_scenario_indices, DataBase, save_to_netcdf
from modules.GAN import TT_GAN
import importlib

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #TODO: Chose the GPU to use. !DO NOT SET TO 1 if you have only 1 GPU!

class Runner():

    def __init__(self,
                 yaml_path,
                 rl_algorithm='DQN',
                 rl_device='cpu'):

        # read config from yaml
        with open(yaml_path, 'rb') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.rl_algorithm = rl_algorithm
        self.rl_device = rl_device

        # set seed
        if self.config['seed'] is None:
            self.seed = random.randint(0, 99999)
        else:
            self.seed = self.config['seed']
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

        if rl_algorithm == 'DQN':
            self.rl_args = self.config.get('dqn_args', {})
        elif rl_algorithm == 'PPO':
            self.rl_args = self.config.get('ppo_args', {})
        elif rl_algorithm == 'SAC':
            self.rl_args = self.config.get('sac_args', {})
        elif rl_algorithm == 'SB3_DQN':
            self.rl_args = self.config.get('sb3_dqn_args', {})
        else:
            raise Exception('Invalid RL Algorithm')

        # Here we use importlib to import the RL_Manager class from the corresponding module
        # You can also define them in the above if-else structure
        RL_Manager = getattr(importlib.import_module(f"modules.{rl_algorithm}"),
                             "RL_Manager")

        # set_paths for saving results & data
        self.set_paths()

        self.ganrl_logger = event_logger(log_output_path=self.parent_saving_dir,
                                        user_label='GANRL_Runner',
                                        create_file=True,
                                        verbose=1)

        # create RL Agent
        self.rl_manager = RL_Manager(
            env_args=self.config['env_args'],
            rl_args=self.rl_args,
            seed=self.seed,
            logger_output_path=self.rl_log_saving_dir,  # for event and data
            saving_folder=self.rl_model_saving_dir,
            device=self.rl_device,
            log_event=False)

        self.ood_scenario_begin = self.config['ood_scenario_begin']

        self.ganrl_logger.logger.info(
            f"=============== Task Info =================")
        self.ganrl_logger.logger.info(
            f"ENV Settings == {self.config['env_args']}")
        self.ganrl_logger.logger.info(
            f"|S| == {self.rl_manager.env.S}, In-sample:1~{self.ood_scenario_begin-1}, OOD:{self.ood_scenario_begin}~{self.rl_manager.env.S}"
        )
        self.ganrl_logger.logger.info(f"RL Algorithm == {self.rl_algorithm}")
        self.ganrl_logger.logger.info(f"RL Device == {self.rl_device}")
        self.ganrl_logger.logger.info(f"Global seed == {self.seed}")

        # split scenario indices: INIT(train_indices) & TEST(test_indices)
        self.train_indices, self.test_indices = split_scenario_indices(
            num_S=self.ood_scenario_begin - 1,
            train_ratio=self.config['training_data_rate'],
            seed=self.seed,
            shuffle=True,
            save_path=self.parent_saving_dir)
        #NOTE: You can also split the data manually

        # set OOD-Test data
        self.ood_test_indices = [
            i for i in range(self.ood_scenario_begin - 1, self.rl_manager.env.S)
        ]

        self.ganrl_logger.logger.info(
            f"=============== Data Splitting Results =================")
        self.ganrl_logger.logger.info(
            f"|Training Set| == {len(self.train_indices)}; |Test Set| == {len(self.test_indices)}"
        )
        # create DataBase objective for managing [env_id, type, latent, label]
        self.laten_shape = (self.rl_manager.env.N, self.rl_manager.env.T,
                            self.config['latent_dim'])
        self.scenario_data_base = DataBase(self.train_indices,
                                           self.test_indices,
                                           self.ood_test_indices,
                                           self.laten_shape)

        self.gan = TT_GAN(
            nodes=self.rl_manager.env.N,
            features=self.rl_manager.env.T,
            time_num=self.rl_manager.env.T,
            logger_output_path=self.gan_log_saving_dir,  # for data 
            model_saving_path=self.gan_model_saving_dir,
            **self.config['aegan_args'])

        # NOTE: The normalizer is not used due to the pu. raw data.
        # self.normalizer = Normalizer(normalization_method=self.config['normalization_method'])

        pass

    def set_paths(self):
        # Set the parent path for saving the results
        paths = self.config.get('folder_args', {})
        result_saving_dir = paths['result_saving_dir']
        # Set the run-time name for logging the results of the test, including the RL algorithm used
        test_name = paths[
            'test_name'] + '_rl=' + self.rl_algorithm + '_' + get_time()
        # Set model-saving dir for:
        # AE-GAN: [D, G, PE, RP];  RL: [Actor, Critic \ Q]
        model_saving_dir = paths['model_saving_dir']
        #
        log_saving_dir = paths['log_saving_dir']
        gen_data_saving_dir = paths['gen_data_saving_dir']

        self.parent_saving_dir = Path(
            __file__).parent.parent / result_saving_dir / test_name
        self.rl_model_saving_dir = self.parent_saving_dir / 'RL' / model_saving_dir
        self.gan_model_saving_dir = self.parent_saving_dir / 'GAN' / model_saving_dir
        self.rl_log_saving_dir = self.parent_saving_dir / 'RL' / log_saving_dir
        self.gan_log_saving_dir = self.parent_saving_dir / 'GAN' / log_saving_dir
        self.gen_data_saving_dir = self.parent_saving_dir / gen_data_saving_dir

        ensure_directory(self.parent_saving_dir)
        ensure_directory(self.rl_model_saving_dir)
        ensure_directory(self.gan_model_saving_dir)
        ensure_directory(self.rl_log_saving_dir)
        ensure_directory(self.gan_log_saving_dir)
        ensure_directory(self.gen_data_saving_dir)

        pass

    def train(self):
        training_set = self.train_indices.copy()
        self.ganrl_logger.logger.info(
                f"------ #1. Generate New ENVs using GAN ------"
            )
        x_env = self.make_dataset(
                env_id=self.train_indices,
                shuffle=True,  #NOTE: Need to shuffle the dataset
                normalize=False)
        last_d_loss, last_g_loss, last_rc_loss = self.gan.train(x_env, True)
        self.ganrl_logger.logger.info(
                f"Training GAN Done! Latest Loss == D:{last_d_loss:.4f}, G:{last_g_loss:.4f}, RC:{last_rc_loss:.4f}"
            )
        
        self.ganrl_logger.logger.info(
                f"------ #2. Generate New ENVs using GAN ------"
            )
        new_env = self.gan.generate(self.config['maximal_gen_num_per_epoch'])
        
        # NOTE: The inverse-normalizer is not used here
        # new_env = self.normalizer.inverse_transform(new_env)
        gen_list = self.add_gen_env(new_env)
        self.ganrl_logger.logger.info(
            f"Obtained New Generated ENV_ID == {gen_list}")
        training_set.extend(
            gen_list)  #NOTE: Add generated env to training_set

        # Save the generated data
        self.save_gen_env()
        
        self.ganrl_logger.logger.info(
            f"----- #3. RL Training using GEN ENVs -----")
        self.rl_manager.reset()
        update_round(self.rl_manager, 0)
        avg_train_reward = self.rl_manager.train(
            scenarios_id_list=training_set, #TODO: Add gen -> training_set\ or directly use gendata -> gen_list
            save_policy=True)
        self.ganrl_logger.logger.info(f"RL Training Done!")
        self.ganrl_logger.logger.info(
            f"Avg.R of {self.rl_args['episode_num']} training episodes == {avg_train_reward:.4f}"
        )
        
        _, avg_in_sample_test_reward = self.rl_manager.eval(
            scenarios_id_list=self.test_indices,
            multi_eval=True,
            policy_path=None,
            user_label='InSample_Test')
        self.ganrl_logger.logger.info(f"In-Sample Test Done!")
        self.ganrl_logger.logger.info(
            f"Avg.R of {len(self.test_indices)} test episodes == {avg_in_sample_test_reward:.4f}"
        )
        _, avg_ood_test_reward = self.rl_manager.eval(
            scenarios_id_list=self.ood_test_indices,
            multi_eval=True,
            policy_path=None,
            user_label='OOD_Test')
        self.ganrl_logger.logger.info(f"OOD Test Done!")
        self.ganrl_logger.logger.info(
            f"Avg.R of {len(self.ood_test_indices)} test episodes == {avg_ood_test_reward:.4f}"
        )

    def add_gen_env(self, gen_data):
        '''
        Add generated data to Pandapower core & 'Register' the generated data to DataBase.
        Return: the env_id list of the generated data.
        '''
        self.rl_manager.env.PP_Agent.dataset_to_scenario(gen_data)
        self.scenario_data_base.add_gen_data(gen_data.shape[0])
        _S = self.rl_manager.env.PP_Agent.S
        #NOTE: Don't forget to update the |S| in Pandapower core
        self.rl_manager.env.PP_Agent.S += gen_data.shape[0]

        return [i for i in range(_S, _S + gen_data.shape[0])]
    
    
    def make_dataset(self, env_id, shuffle=True, normalize=True):
        
        '''
        Make dataset of [
                        X := features of ENV,     Time series data --> from the source data, i.e., the Pandapower core.
                        Y := labels of Reward     Scalar  --> from the DataBase
                        ]
        
        '''
        
        x = self.rl_manager.env.PP_Agent.scenario_to_dataset(env_id)
        if normalize:
            self.normalizer.fit(x)
            x = self.normalizer.transform(x)            
        x = torch.tensor(x, dtype=torch.float) #NOTE: No need to move to cuda yet

        
        if shuffle:
            shuffle_index = [i for i in range(len(env_id))]
            random.shuffle(shuffle_index)
            x = x[shuffle_index]
        
        return x

    def save_gen_env(self):
        '''
        Save the generated data to the gen_data_saving_dir
        '''
        table = self.scenario_data_base.table
        gen_latent = np.array(
            table[table['type'] == 3]['latent'].values.tolist())  
        gen_id = table[table['type'] == 3]['id'].values.tolist()  
        gen_pv = self.rl_manager.env.PP_Agent.pv_scenario[gen_id].transpose(
            0, 2, 1)
        gen_load = self.rl_manager.env.PP_Agent.load_scenario[gen_id]
        gen_wp = self.rl_manager.env.PP_Agent.wp_scenario[gen_id].transpose(
            0, 2, 1)
        gen_disruption = self.rl_manager.env.PP_Agent.disruption_scenario[
            gen_id].transpose(0, 2, 1)
        path = self.gen_data_saving_dir / ('gen_data_' + get_time() + '.nc')
        save_to_netcdf(gen_latent, gen_pv, gen_wp, gen_disruption, gen_load,
                       path)


def update_round(obj, round_number):
    if hasattr(obj, 'round'):
        obj.round = round_number
    else:
        setattr(obj, 'round', round_number)
        

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'GANRL_config_7bus.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    runner = Runner(yaml_path=path,
                    rl_algorithm='DQN',
                    rl_device=device)
    runner.train()