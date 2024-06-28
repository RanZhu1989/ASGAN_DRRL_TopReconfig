"""Gym entry point for Simple MGF_Env environment."""
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from .PP_Agent import Pandapower_Agent
import numpy as np


class SimpleMGF_Env(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self,
                 system_data_file,
                 scenario_data_file,
                 v_min=0.95,
                 v_max=1.05,
                 static_penalty=1,
                 penalty_v_violation=1,
                 penalty_p_loss=1,
                 penalty_q_loss=1,
                 penalty_p_unbalance=1,
                 penalty_q_unbalance=1,
                 mp_evl=False):
        # Create the Pandapower Agent
        system_file_name = Path(
            __file__).parent / 'case_data' / 'system' / system_data_file
        scenario_file_name = Path(
            __file__).parent / 'case_data' / 'scenario' / scenario_data_file
        self.PP_Agent = Pandapower_Agent(system_file_name, scenario_file_name,
                                         v_min, v_max)
        '''
        Load results from MP-based method
        '''
        self.mp_evl = mp_evl
        #----------------------------------------------------------------
        # Code for loading the results from MP-based method
        if self.mp_evl:
            b_state = np.load(Path(__file__).parent / 'ba.npy')
            b_state = np.round(b_state)
            self.ba = b_state.copy()
            self.ba[13], self.ba[52] = b_state[52], b_state[13] # Note: the 13th and 52th line are reversed
        #---------------------------------------------------------------
        
        # Obtain the case data
        self.S = self.PP_Agent.S  # number of scenarios
        self.T = self.PP_Agent.T  # number of time steps
        self.N = self.PP_Agent.num_sgen + self.PP_Agent.num_vulne_line + 1  # RDG + Vulne_Line + Load
        self.num_switch = self.PP_Agent.num_switch
        self.num_bus = self.PP_Agent.num_bus
        self.num_line = self.PP_Agent.num_line

        # For penalty calculation
        self.static_penalty = static_penalty
        self.penalty_v_violation = penalty_v_violation
        self.penalty_p_loss = penalty_p_loss
        self.penalty_q_loss = penalty_q_loss
        self.penalty_p_unbalance = penalty_p_unbalance
        self.penalty_q_unbalance = penalty_q_unbalance

        # Set the action and observation space
        self.action_space = spaces.MultiBinary(self.num_switch)
        #TODO Can be revised in future
        self.observation_space = spaces.Dict({
            "P":
                spaces.Box(low=np.float32(-np.inf * np.ones(self.num_bus)),
                           high=np.float32(np.inf * np.ones(self.num_bus))),
            "Q":
                spaces.Box(low=np.float32(-np.inf * np.ones(self.num_bus)),
                           high=np.float32(np.inf * np.ones(self.num_bus))),
            "Top":
                spaces.MultiBinary(self.num_line)
        })

    def reset(
            self,
            options={
                "Candidate_Scenario_ID_List": None,
                "Specific_Scenario_ID": None,
                "External_RNG": None
            },
            seed=None):
        super().reset(seed=seed)
        # If the range of candidate scenarios is given, random scenarios are selected within this range, 
        # while specified scenarios need to be checked if they exceed this range.
        # If the range of candidate scenarios is not given, random scenarios are selected from all possible scenarios.

        Candidate_Scenario_ID_List = options["Candidate_Scenario_ID_List"]
        Specific_Scenario_ID = options["Specific_Scenario_ID"]
        external_rng = options["External_RNG"]

        if Candidate_Scenario_ID_List is not None:
            if Specific_Scenario_ID == None:
                random_scenario = True
                if external_rng is not None:
                    chosen_scenario_id = external_rng.choice(
                        Candidate_Scenario_ID_List)
                else:
                    chosen_scenario_id = self.np_random.choice(
                        Candidate_Scenario_ID_List)

            else:
                # check if Specific_Scenario_ID is in Candidate_Scenario_ID_List
                if Specific_Scenario_ID not in Candidate_Scenario_ID_List:
                    raise ValueError(
                        "Specific_Scenario_ID must be in Candidate_Scenario_ID_List"
                    )

                random_scenario = False
                chosen_scenario_id = Specific_Scenario_ID

        else:
            if Specific_Scenario_ID == None:
                random_scenario = True
                if external_rng is not None:
                    chosen_scenario_id = external_rng.randint(
                        0, self.S - 1)
                else:
                    chosen_scenario_id = self.np_random.integers(low=0,
                                                                 high=self.S -
                                                                 1)

            else:
                if Specific_Scenario_ID < 0 or Specific_Scenario_ID >= self.S:
                    raise ValueError(
                        "Specific_Scenario_ID must be in the range [0, S-1]")

                random_scenario = False
                chosen_scenario_id = Specific_Scenario_ID

        # If Specific_Scenario_ID is None, randomly choose a scenario_id

        # Reset := Time step 0
        self.current_scenario_id = chosen_scenario_id
        load_profile, pv_profile, wp_profile, disruption_profile = self.PP_Agent.ts_assign(
            scenario_id=chosen_scenario_id, time_step=0)
        if self.mp_evl:
            self.PP_Agent.force_line_status(self.ba,scenario_id=self.current_scenario_id,
                    time_step=0)

        # Set step counter
        self.current_step = 0
        self.step_seq_idx = [i for i in range(self.T)]

        # Obtain the initial observation
        p_bus, q_bus = self.PP_Agent.run_pf()
        p_bus = p_bus.to_numpy()
        q_bus = q_bus.to_numpy()
        line_table, _, _ = self.PP_Agent.get_line_states()
        sys_top = line_table['line_connectivity'].to_numpy(
        )  # Obtain the topology from the line table
        self.obs = {
            'P': p_bus.astype(np.float32),
            'Q': q_bus.astype(np.float32),
            'Top': sys_top.astype(np.int8)
        }

        info = {
            'Random_Scenario': random_scenario,
            'Chosen_Scenario_ID': chosen_scenario_id,
            'ENV_Profile': {
                'Load': load_profile,
                'PV': pv_profile,
                'WP': wp_profile,
                'Disruption': disruption_profile
            }
        }

        return self.obs, info

    def step(self, action):
        assert self.action_space.contains(
            action
        ), "Invalid action. See docstring of step() for more information."
        action_switch = action

        # Update the last switch states for checking illegal switch
        self.PP_Agent.update_last_switch_states('push')

        # Initialize serval flags
        event_log = None
        action_accepted = False  # Whether the action is accepted by the environment
        events = {
            'Episode_Closed': 0,
            'OK': 1,
            'Illegal_Switch': 2.1,
            'Load_Energized_Failed': 2.2,
            'Illegal_Switch_and_Load_Energized_Failed': 2.3
        }

        # If current step is out of range, set the termination flag to True
        if self.current_step <= self.T - 2:  # T-1-1
            # Proceed to the next step
            if self.current_step == self.T - 2:  # For the last step, and the episode is terminated
                done = True
            else:
                done = False
            # Try switch action
            self.PP_Agent.switch_action(action_switch)
            # Assign scenario data of the next time step
            load_profile, pv_profile, wp_profile, disruption_profile = self.PP_Agent.ts_assign(
                scenario_id=self.current_scenario_id,
                time_step=self.current_step + 1)
            if self.mp_evl:
                self.PP_Agent.force_line_status(self.ba,scenario_id=self.current_scenario_id,
                    time_step=self.current_step + 1)
            # Check if the switch action is illegal or load energized failed
            _, flag_illegal_switch, _ = self.PP_Agent.get_line_states()
            _, flag_load_energized_failed = self.PP_Agent.get_load_energized_states(
            )
            if flag_illegal_switch or flag_load_energized_failed:
                # Infeasible action leads to large penalty
                reward = -self.static_penalty
                self.PP_Agent.update_last_switch_states(
                    'pull')  # Must pull back the switch states
                # log the events
                if flag_illegal_switch and not flag_load_energized_failed:
                    event_log = 'Illegal_Switch'
                elif not flag_illegal_switch and flag_load_energized_failed:
                    event_log = 'Load_Energized_Failed'
                else:
                    event_log = 'Illegal_Switch_and_Load_Energized_Failed'

                # fake penalties
                pen_v_violation = np.nan
                pen_p_loss = np.nan
                pen_q_loss = np.nan
                pen_p_unbalance = np.nan
                pen_q_unbalance = np.nan

            else:
                # Only if action is feasible, then run power flow
                res_p, res_q = self.PP_Agent.run_pf()
                # Calculate the reward, get observation
                res_line, _, _ = self.PP_Agent.get_line_states()
                v_violation = self.PP_Agent.get_v_violation()
                p_loss, q_loss = self.PP_Agent.get_pq_loss()
                p_unbalance, q_unbalance = self.PP_Agent.get_pq_unbalance()
                reward = self.static_penalty - self.penalty_v_violation * sum(v_violation) \
                        - self.penalty_p_loss * sum(p_loss) - self.penalty_q_loss * sum(q_loss)  \
                        - self.penalty_p_unbalance * p_unbalance - self.penalty_q_unbalance * q_unbalance
                action_accepted = True
                self.obs = {
                    'P':
                        res_p.to_numpy().astype(np.float32),
                    'Q':
                        res_q.to_numpy().astype(np.float32),
                    'Top':
                        res_line['line_connectivity'].to_numpy().astype(np.int8)
                }
                event_log = 'OK'

                pen_v_violation = sum(v_violation)
                pen_p_loss = sum(p_loss)
                pen_q_loss = sum(q_loss)
                pen_p_unbalance = p_unbalance
                pen_q_unbalance = q_unbalance

                pass

        else:
            # Step out of range, episode is terminated
            done = True
            reward = 0
            event_log = "Episode_Closed"
            pen_v_violation = np.nan
            pen_p_loss = np.nan
            pen_q_loss = np.nan
            pen_p_unbalance = np.nan
            pen_q_unbalance = np.nan

        self.current_step += 1  # Move to the current step

        info = {
            "Attempted_Switch_Action": action_switch,
            "Action_Accepted": action_accepted,  # bool
            "Current_Step": self.current_step,  # int
            "Event_Log": event_log,
            "Event_Code": events[event_log],  # float
            'ENV_Profile': {
                'Load': load_profile,
                'PV': pv_profile,
                'WP': wp_profile,
                'Disruption': disruption_profile
            },
            "Penalties": {
                "V_Violation": pen_v_violation,
                "P_Loss": pen_p_loss,
                "Q_Loss": pen_q_loss,
                "P_Unbalance": pen_p_unbalance,
                "Q_Unbalance": pen_q_unbalance
            }
        }

        # Use self.obs to keep the observation in case of infeasible action
        return self.obs, reward, done, False, info
