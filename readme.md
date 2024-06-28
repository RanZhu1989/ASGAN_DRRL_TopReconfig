# Updates

- (July, 2023) Upload codes for review. Many thanks for suggestions and supports!


# Adversarial Scenario Generation Integrated Distributionally Robust Reinforcement Learning for Survival of Critical Loads
by [Ran Zhu](https://scholar.google.com/citations?hl=zh-CN&user=ETHzl_0AAAAJ), [Hongzhe Liu](https://scholar.google.com/citations?hl=zh-CN&user=M6fT730AAAAJ), [Mingyu Kang](https://scholar.google.com/citations?hl=zh-CN&user=2CGxd5sAAAAJ), [Wenwu Yu*](https://scholar.google.com/citations?hl=zh-CN&user=I7XxngUAAAAJ), and [Jin Zhao*](https://scholar.google.com/citations?hl=zh-CN&user=vV68wh8AAAAJ)

## Abstract
We proposes a *distributionally robust reinforcement learning (**DRRL**)*-based method for topology reconfiguration against extreme events, addressing the current limitation where scenario generation is treated as a separate task and lacks of feedback, leading to less robust policies against uncertainty events. The approach integrates an temporal-embed transformer networks (**TTNs**)-enhanced *adversarial scenario generative adversarial network (**AS-GAN**)* to dynamically generate adversarial scenarios, enhancing policy robustness. Validated on a 7-bus test system and the IEEE 123-bus system, the method shows superior performance compared to traditional reinforcement learning (RL) approaches.

## Framework
![Implementation Approach](/pics/AEGAN.png)

## Comparison with other methods
### &bull; Comaprison with conventional RL methods:
**M1**: Directly train; **M2**: Train with raw-generated-mixed scenarios; **M3**: Train with only generated scenarios; **M4**: Proposed DRRL, train with adversarial scenarios
![Comparison](/pics/Compare_OtherRL.png)
### &bull; Comaprison with mathematical optimization methods:
![Comparison](/pics/Compare_MP.png)
### &bull; Comaprison with other adversarial scenario generation models:
![Comparison](/pics/Compare_SGs.png)

## Getting started
- To install, `cd` into the gym environment directory `./gym_SimpleMGF/` and type `pip install -e .`

- To run the DRRL simluations, set hyperparameters in `*.yaml` files and run `run_ASRL.py`

- To use other test systems, add system file in `./gym_SimpleMGF/SimpleMGF_env/envs/case_data/system` and scenario file in `./gym_SimpleMGF/SimpleMGF_env/envs/case_data/scenario`

- You can edit the environment in `./gym_SimpleMGF/SimpleMGF_env/envs/SimpleMGF_env.py` (also welcome to refer to our previous [repository](https://github.com/RanZhu1989/IL_Self_Healing))

- Known dependencies: 
  - **For DRRL:** Python (3.8.19), PyTorch (2.3.0), Gymnasium (0.28.1), Pandapower (2.14.6), numba (0.58.1), CUDA (12.1)
  - **For Mathematical Optimization:** Julia (1.10.3), JuMP (1.22.1)
  - **Solvers:** Gurobi (v10 and v11), CPLEX (12.10)

## Code structure
### Source codes of algorithms:
- `./algorithms/run_ASRL.py`: Main script for running the DRRL
- `./algorithms/run_GANRL.py`: Main script for running the GAN-enhanced RL, i.e., scenario generation and RL were sequentially executed without feedback
- `./algorithms/*_config_*.yaml`: Configuration files for the DRRL and GANRL
- `./algorithms/modules/`: Folder where various deep learning-based modules are stored, including:
  1) `AEGAN_TTN.py`: Class of the **TTN-enhanced AS-GAN**: Definitions, Training approach (Algorithm 1 in the paper), and adversarial generating method
  2) `AEGAN_Naive.py`: Class of the **Naive AS-GAN** (use MLP): Definitions, Training approach, and adversarial generating method
  3) `EPD.py`: Class of the **Autoencoder**-based adversarial scenario generation [model](https://github.com/irom-lab/DRAGEN)
  4) `DQN.py`: Agent class of the **Deep Q-Network** (DQN) for the RL task. Source code is from our pervious [repository](https://github.com/RanZhu1989/RL_PlayGround)
  5) `PPO.py`: Agent class of the **Proximal Policy Optimization** (PPO) for the RL task. Source code is from our pervious [repository](https://github.com/RanZhu1989/RL_PlayGround)

- `./algorithms/MOPs/`: Folder where the mathematical optimization implementation (coded by Julia) are stored, including:
  1) `mop_utils.jl`: Utility functions (mainly for data processing) for the optimization task
  2) `MOP_Deterministic.jl`: The deterministic optimization model used in the Case Study
  3) `MOP_TSSO.jl`: The two-stage stochastic optimization model used in the Case Study
  4) `123Bus_Data.xlsx`: Data file for the 123-bus system (recast version for convenience)
  5) `123Bus_*Scenario.xlsx`: Scenario files for TSSO model
- `./algorithms/utils/`: Utility functions for tasks, including:
  1) `data.py`: Environment database, data processing, and statistical metrics
  2) `misc.py`: Logging, time measurement, and other miscellaneous functions
  3) `wrappers.py`: Environment wrappers for the RL task
### Source codes of environments:
- `./gym_SimpleMGF/SimpleMGF_env/envs/PP_Agent.py`: Power flow agent made by Pandapower
- `./gym_SimpleMGF/SimpleMGF_env/envs/SimpleMGF_env.py`: Environment for the RL task
- `./gym_SimpleMGF/SimpleMGF_env/envs/case_data/scenario/*_scenario.xlsx`: Scenario files, including load, generation, and line outages
- `./gym_SimpleMGF/SimpleMGF_env/envs/case_data/system/*_Data.xlsx`: Test system files, including bus data, line data, and generator data, etc.
- The environment strictly adheres to the OpenAI Gymnasium API and has passed validation using Stable Baselines3