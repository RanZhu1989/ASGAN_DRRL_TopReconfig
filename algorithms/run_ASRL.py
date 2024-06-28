import os
import yaml
import random

from pathlib import Path

import numpy as np
import torch

from utils.misc import event_logger, ensure_directory, get_time
from utils.data import split_scenario_indices, DataBase, save_to_netcdf,random_sample

import importlib

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #TODO: Chose the GPU to use. !DO NOT SET TO 1 if you have only 1 GPU!

class Runner():

    def __init__(self,
                 yaml_path,
                 rl_algorithm='DQN',
                 rl_device='cpu',
                 adv_gen_mode='AEGAN_TTN'): # DPE, AEGAN_TTN, AEGAN_Naive

        # read config from yaml
        with open(yaml_path, 'rb') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.rl_algorithm = rl_algorithm
        self.rl_device = rl_device
        
        self.adv_gen_mode = adv_gen_mode
        if self.adv_gen_mode == 'EPD':
            from modules.EPD import EPD
            
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

        self.asrl_logger = event_logger(log_output_path=self.parent_saving_dir,
                                        user_label='ASRL_Runner',
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

        self.asrl_logger.logger.info(
            f"=============== Task Info =================")
        self.asrl_logger.logger.info(
            f"ENV Settings == {self.config['env_args']}")
        self.asrl_logger.logger.info(
            f"|S| == {self.rl_manager.env.S}, In-sample:1~{self.ood_scenario_begin-1}, OOD:{self.ood_scenario_begin}~{self.rl_manager.env.S}"
        )
        self.asrl_logger.logger.info(f"RL Algorithm == {self.rl_algorithm}")
        self.asrl_logger.logger.info(f"RL Device == {self.rl_device}")
        self.asrl_logger.logger.info(f"Adversarial Environment Generation Model == {self.adv_gen_mode}")
        self.asrl_logger.logger.info(f"Global seed == {self.seed}")

        # split scenario indices: INIT(train_indices) & TEST(test_indices)
        self.train_indices, self.test_indices = split_scenario_indices(
            num_S=self.ood_scenario_begin - 1,
            train_ratio=self.config['training_data_rate'],
            seed=self.seed, # Fix the seed for data splitting
            shuffle=True,
            save_path=self.parent_saving_dir)
        #NOTE: You can also split the data manually

        # set OOD-Test data
        self.ood_test_indices = [
            i for i in range(self.ood_scenario_begin - 1, self.rl_manager.env.S)
        ]

        self.asrl_logger.logger.info(
            f"=============== Data Splitting Results =================")
        self.asrl_logger.logger.info(
            f"|Training Set| == {len(self.train_indices)}; |Test Set| == {len(self.test_indices)}"
        )
        # create DataBase objective for managing [env_id, type, latent, label]
        if self.adv_gen_mode == 'AEGAN_TTN':
            self.laten_shape = (self.rl_manager.env.N, self.rl_manager.env.T,
                                self.config['latent_dim'])
        elif self.adv_gen_mode == 'EPD' or self.adv_gen_mode == 'AEGAN_Naive':
            self.laten_shape = [self.config['latent_dim']]
        else: 
            raise Exception('Invalid Adversarial Generation Mode')
        
        self.scenario_data_base = DataBase(self.train_indices,
                                           self.test_indices,
                                           self.ood_test_indices,
                                           self.laten_shape)
        if self.adv_gen_mode == 'AEGAN_TTN' or self.adv_gen_mode == 'AEGAN_Naive':
            AE_GAN = getattr(importlib.import_module(f"modules.{self.adv_gen_mode}"),
                             "AE_GAN")
            self.ae_gan = AE_GAN(
                nodes=self.rl_manager.env.N,
                features=self.rl_manager.env.T,
                time_num=self.rl_manager.env.T,
                logger_output_path=self.adv_gen_log_saving_dir,  # for data 
                model_saving_path=self.adv_gen_model_saving_dir,
                **self.config['aegan_args'])
        elif self.adv_gen_mode == 'EPD':
            self.epd = EPD(
                nodes=self.rl_manager.env.N,
                features=self.rl_manager.env.T,
                logger_output_path=self.adv_gen_log_saving_dir,  # for data 
                model_saving_path=self.adv_gen_model_saving_dir,
                **self.config['epd_args'])

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
        self.adv_gen_model_saving_dir = self.parent_saving_dir / self.adv_gen_mode / model_saving_dir
        self.rl_log_saving_dir = self.parent_saving_dir / 'RL' / log_saving_dir
        self.adv_gen_log_saving_dir = self.parent_saving_dir / self.adv_gen_mode / log_saving_dir
        self.gen_data_saving_dir = self.parent_saving_dir / gen_data_saving_dir

        ensure_directory(self.parent_saving_dir)
        ensure_directory(self.rl_model_saving_dir)
        ensure_directory(self.adv_gen_model_saving_dir)
        ensure_directory(self.rl_log_saving_dir)
        ensure_directory(self.adv_gen_log_saving_dir)
        ensure_directory(self.gen_data_saving_dir)

        pass

    def train(self):
        '''
        Algorithm 2 in the paper
        '''
        epoch = 0
        max_epoch = self.config['max_epoch']
        training_set = self.train_indices.copy(
        )  #NOTE: training_set = INIT + GEN
        for epoch in range(max_epoch):
            self.current_epoch = epoch
            self.asrl_logger.logger.info(
                f"=============== Epoch {epoch+1:d} of {max_epoch:d} ================="
            )
            # 1. Re-train & eval -> update pi & get cost label y
            self.rl_manager.reset()
            update_round(self.rl_manager, epoch)
            self.asrl_logger.logger.info(
                f"----- #1. RL Retraining using INIT & GEN ENVs -----")
            avg_train_reward = self.rl_manager.train(
                scenarios_id_list=training_set, save_policy=True)
            self.asrl_logger.logger.info(f"RL Retrain Done!")
            self.asrl_logger.logger.info(
                f"Avg.R of {self.rl_args['episode_num']} training episodes == {avg_train_reward:.4f}"
            )
            # Eval the policy on training_set
            _, _ = self.rl_manager.eval(scenarios_id_list=self.train_indices,
                                        multi_eval=True,
                                        policy_path=None,
                                        user_label='Training_Set_Evaluation')
            self.asrl_logger.logger.info(
                f"In-Sample Training Set Evaluation Done!")
            _, avg_in_sample_test_reward = self.rl_manager.eval(
                scenarios_id_list=self.test_indices,
                multi_eval=True,
                policy_path=None,
                user_label='InSample_Test')
            self.asrl_logger.logger.info(f"In-Sample Test Done!")
            self.asrl_logger.logger.info(
                f"Avg.R of {len(self.test_indices)} test episodes == {avg_in_sample_test_reward:.4f}"
            )
            _, avg_ood_test_reward = self.rl_manager.eval(
                scenarios_id_list=self.ood_test_indices,
                multi_eval=True,
                policy_path=None,
                user_label='OOD_Test')
            self.asrl_logger.logger.info(f"OOD Test Done!")
            self.asrl_logger.logger.info(
                f"Avg.R of {len(self.ood_test_indices)} test episodes == {avg_ood_test_reward:.4f}"
            )

            self.asrl_logger.logger.info(
                f"----- #2. RL Evaluation using INIT & GEN ENVs -----")
            label, _ = self.rl_manager.eval(
                scenarios_id_list=training_set,
                multi_eval=True,
                # policy_path=self.rl_manager.last_saving_policy_path
                policy_path=None,  #NOTE: Use the latest policy in memory
                user_label='Eval_Cost')
            self.asrl_logger.logger.info(
                f"Evaluating {len(label)} ENVs ... Done!")

            # 2. setup_dataloader()
            self.scenario_data_base.update_label(env_id=training_set,
                                                 label=label)  # update y

            # 3. Embedding (Train AE-GAN or EPD)
            self.asrl_logger.logger.info(
                f"------ #3. Embedding Training using INIT & GEN ENVs with Latest Policy ------"
            )
            x_env, y_label = self.make_dataset(
                env_id=training_set,
                shuffle=True,  #NOTE: Need to shuffle the dataset
                normalize=False)
            
            if self.adv_gen_mode == 'AEGAN_TTN' or self.adv_gen_mode == 'AEGAN_Naive':
                update_round(self.ae_gan, epoch)
                latent, last_d_loss, last_g_loss, last_rc_loss, last_lip_loss, \
                last_norm_loss, last_rp_loss = self.ae_gan.train(x_env, y_label,save_model=True)
            elif self.adv_gen_mode == 'EPD':
                update_round(self.epd, epoch)
                latent, last_rc_loss, last_rp_loss,\
                    last_lip_loss, last_norm_loss = self.epd.train(x_env, y_label,save_model=True)
                  
            latent_list = self.scenario_data_base.update_latent(
                env_id=training_set,  #NOTE 同上
                latent=latent)  # update z
            
            if self.adv_gen_mode == 'AEGAN_TTN' or self.adv_gen_mode == 'AEGAN_Naive':
                self.asrl_logger.logger.info(
                    f"Embedding Done! Latest Loss == D:{last_d_loss:.4f}, G:{last_g_loss:.4f}, RC:{last_rc_loss:.4f}, LIP:{last_lip_loss:.4f}, NORM:{last_norm_loss:.4f}, RP:{last_rp_loss:.4f}"
                )
            elif self.adv_gen_mode == 'EPD':
                self.asrl_logger.logger.info(
                    f"Embedding Done! Latest Loss == RC:{last_rc_loss:.4f}, RP:{last_rp_loss:.4f}, LIP:{last_lip_loss:.4f}, NORM:{last_norm_loss:.4f}"
                )
            # 4. Gen()
            self.asrl_logger.logger.info(
                f"------ #4. Generating New ENVs using Random Selected ENVs -----"
            )
            gen_index = random_sample(
                data_list=[i for i in range(len(training_set))],
                sample_ratio=self.config['latent_used_rate_per_gen'],
                fixed=False,
                threshold=self.config['maximal_gen_num_per_epoch'])

            self.asrl_logger.logger.info(
                f"Selected ENV_ID == {[idx for idx in gen_index]}")
            target_drop = self.calculate_target_drop(
                [label[i] for i in gen_index])
            if self.adv_gen_mode == 'AEGAN_TTN' or self.adv_gen_mode == 'AEGAN_Naive':
                new_env = self.ae_gan.adversarial_gen(
                    target_drop=target_drop,
                    latent_list=[latent_list[i] for i in gen_index],
                    num_gen_per_latent=self.config['num_gen_per_latent'])
            elif self.adv_gen_mode == 'EPD':
                new_env = self.epd.adversarial_gen(
                    target_drop=target_drop,
                    latent_list=[latent_list[i] for i in gen_index],
                    num_gen_per_latent=self.config['num_gen_per_latent'])
            
            # NOTE: The inverse-normalizer is not used here
            # new_env = self.normalizer.inverse_transform(new_env)
            gen_list = self.add_gen_env(new_env)
            self.asrl_logger.logger.info(
                f"Obtained New Generated Adversarial ENV_ID == {gen_list}")
            training_set.extend(
                gen_list)  #NOTE: Add generated env to training_set

            # Save the generated data
            self.save_gen_env()

            # Eval the policy on generated_set
            self.asrl_logger.logger.info(
                f"Evaluating the total {len(self.train_indices)} in-sample ENVs"
            )
            _, _ = self.rl_manager.eval(scenarios_id_list=gen_list,
                                        multi_eval=True,
                                        policy_path=None,
                                        user_label='GEN_Set_Evaluation')
            self.asrl_logger.logger.info(f"New Generated ENV Evaluation Done!")

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
        x = torch.tensor(x,
                         dtype=torch.float)  #NOTE: No need to move to cuda yet
        y_arrays = self.scenario_data_base.table.loc[
            self.scenario_data_base.table['id'].isin(env_id), 'label'].values
        y = np.array(
            [float(arr) if not np.isnan(arr) else np.nan for arr in y_arrays])
        y = torch.tensor(y, dtype=torch.float)

        if shuffle:
            shuffle_index = [i for i in range(len(env_id))]
            random.shuffle(shuffle_index)
            x = x[shuffle_index]
            y = y[shuffle_index]

        return x, y

    def calculate_target_drop(self, label):
        label_range = np.max(label) - np.min(label)
        target_drop = label_range * self.config['target_drop_rate']

        return target_drop

    def save_gen_env(self):
        '''
        Save the generated data to the gen_data_saving_dir

        '''
        table = self.scenario_data_base.table
        # Find the values in the 'latent' column for all rows in the table where the 'type' column is 'GEN'
        gen_latent = np.array(
            table[table['type'] == 3]['latent'].values.tolist())  #
        gen_id = table[table['type'] == 3]['id'].values.tolist()  # gen场景的id
        gen_pv = self.rl_manager.env.PP_Agent.pv_scenario[gen_id].transpose(
            0, 2, 1)
        gen_load = self.rl_manager.env.PP_Agent.load_scenario[gen_id]
        gen_wp = self.rl_manager.env.PP_Agent.wp_scenario[gen_id].transpose(
            0, 2, 1)
        gen_disruption = self.rl_manager.env.PP_Agent.disruption_scenario[
            gen_id].transpose(0, 2, 1)
        path = self.gen_data_saving_dir / ('Round_' + str(self.current_epoch) +
                                           '_gen_data_' + get_time() + '.nc')
        save_to_netcdf(gen_latent, gen_pv, gen_wp, gen_disruption, gen_load,
                       path)


def update_round(obj, round_number):
    if hasattr(obj, 'round'):
        obj.round = round_number
    else:
        setattr(obj, 'round', round_number)


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'ASRL_config_7bus.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    runner = Runner(yaml_path=path,
                    rl_algorithm='DQN',
                    rl_device=device,
                    adv_gen_mode='AEGAN_TTN')
    runner.train()
