from gym.wrappers import ClipAction

from rlkit.envs.contextual.goal_conditioned import (
    PresampledPathDistribution,
)
from rlkit.envs.contextual.latent_distributions import (  # NOQA
    AddDecodedImageDistribution,
    AddLatentDistribution,
    PresamplePriorDistribution,
)
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv
from rlkit.envs.images import EnvRenderer
from rlkit.envs.images import InsertImageEnv
from rlkit.launchers.contextual.rig.rig_launcher import StateImageGoalDiagnosticsFn  # NOQA
from rlkit.launchers.contextual.util import get_gym_env

from rlkit.experimental.kuanfang.envs.reward_fns import GoalReachingRewardFn
from rlkit.experimental.kuanfang.envs.contextual_env import ContextualEnv

############################
# arguments normally passed in
############################
init_camera = None

renderer_kwargs=dict(
    create_image_format='HWC',
    output_image_format='CWH',
    flatten_image=True,
    width=48,
    height=48,
),

use_vqvae = False
model = {}
reset_keys_map = {}
obs_key = 'image_observation'
image_goal_key = 'image_desired_goal'
latent_goal_key = 'latent_desired_goal'
use_image = True

############################
# arguments normally passed in
############################

renderer = EnvRenderer(init_camera=init_camera, **renderer_kwargs)

def contextual_env_distrib_and_reward(
    env_id,
    env_class,
    env_kwargs,
    goal_sampling_mode,
    presampled_goals_path,
    num_presample,
    reward_kwargs,
    presampled_goals_kwargs,
):
    state_env = get_gym_env(
        env_id,
        env_class=env_class,
        env_kwargs=env_kwargs,
    )
    state_env = ClipAction(state_env)
    renderer = EnvRenderer(
        init_camera=init_camera,
        **renderer_kwargs)

    env = InsertImageEnv(
        state_env,
        renderer=renderer)

    if use_vqvae:
        vqvae = model['vqvae']
        env = EncoderWrappedEnv(
            env,
            vqvae,
            step_keys_map=dict(image_observation='latent_observation'),
            reset_keys_map=reset_keys_map,
        )

    if goal_sampling_mode == 'presampled_images':
        print(presampled_goals_path)
        diagnostics = state_env.get_contextual_diagnostics
        context_distribution = PresampledPathDistribution(
            presampled_goals_path,
            vqvae.representation_size if use_vqvae else None,
            initialize_encodings=use_vqvae)
        if use_vqvae:
            context_distribution = AddLatentDistribution(
                context_distribution,
                input_key=image_goal_key,
                output_key=latent_goal_key,
                model=vqvae)
    elif goal_sampling_mode == 'presample_latents':
        assert use_vqvae
        diagnostics = StateImageGoalDiagnosticsFn({}, )
        context_distribution = PresamplePriorDistribution(
            model,
            latent_goal_key,
            state_env,
            num_presample=num_presample,
            affordance_type='cc_vae')
        if use_image:
            context_distribution = AddDecodedImageDistribution(
                context_distribution,
                latent_goal_key,
                image_goal_key,
                vqvae)
    else:
        raise NotImplementedError

    reward_fn = GoalReachingRewardFn(
        state_env,
        **reward_kwargs
    )

    contextual_env = ContextualEnv(
        env,
        context_distribution=context_distribution,
        reward_fn=reward_fn,
        observation_key=obs_key,
        contextual_diagnostics_fns=[diagnostics] if not isinstance(
            diagnostics, list) else diagnostics,
    )

    return contextual_env, context_distribution, reward_fn
