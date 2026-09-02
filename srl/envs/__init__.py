"""Environment factory.

Returns ``(env, state_dim, goal_start_idx, goal_end_idx)`` where ``state_dim`` is
the width of the state prefix of an observation (the goal is appended after it)
and ``[goal_start_idx, goal_end_idx)`` slices the goal coordinates out of a state.
"""


def make_env(env_id):
    print(f"making env with env_id: {env_id}", flush=True)

    if env_id == "ant":
        from srl.envs.ant import Ant

        env = Ant(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )

        state_dim = 29
        goal_start_idx = 0
        goal_end_idx = 2

    # The "ant" check differentiates these from the humanoid mazes below.
    elif "ant" in env_id and "maze" in env_id:
        from srl.envs.ant_maze import AntMaze

        env = AntMaze(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
            maze_layout_name=env_id[4:],
        )

        state_dim = 29
        goal_start_idx = 0
        goal_end_idx = 2

    elif env_id == "humanoid":
        from srl.envs.humanoid import Humanoid

        env = Humanoid(
            backend="spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )

        state_dim = 268
        goal_start_idx = 0
        goal_end_idx = 3

    elif "humanoid" in env_id and "maze" in env_id:
        from srl.envs.humanoid_maze import HumanoidMaze

        env = HumanoidMaze(backend="spring", maze_layout_name=env_id[9:])

        state_dim = 268
        goal_start_idx = 0
        goal_end_idx = 3

    elif env_id == "arm_push_easy":
        from srl.envs.manipulation.arm_push_easy import ArmPushEasy

        env = ArmPushEasy(
            backend="mjx",
        )

        state_dim = 17
        goal_start_idx = 0
        goal_end_idx = 3

    elif env_id == "arm_push_hard":
        from srl.envs.manipulation.arm_push_hard import ArmPushHard

        env = ArmPushHard(
            backend="mjx",
        )

        state_dim = 17
        goal_start_idx = 0
        goal_end_idx = 3

    elif env_id == "arm_binpick_hard":
        from srl.envs.manipulation.arm_binpick_hard import ArmBinpickHard

        env = ArmBinpickHard(
            backend="mjx",
        )

        state_dim = 17
        goal_start_idx = 0
        goal_end_idx = 3

    else:
        raise NotImplementedError(f"Unknown env_id: {env_id}")

    return env, state_dim, goal_start_idx, goal_end_idx
