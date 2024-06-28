from gymnasium.envs.registration import register

register(
    id="SimpleMGF-v0",
    entry_point="SimpleMGF_env.envs:SimpleMGF_Env",
)