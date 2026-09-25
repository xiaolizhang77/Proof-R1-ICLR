import importlib
import logging
import os
from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import load_class_from_fqn

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def run_ppo(config, task_runner_class) -> None:


    rollout_cfg = config.actor_rollout_ref.rollout
    rm_rollout_cfg = config.reward.reward_model.rollout
    if rollout_cfg.full_determinism or (config.reward.reward_model.enable and rm_rollout_cfg.full_determinism):
        os.environ["VERL_FULL_DETERMINISM"] = "1"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["PYTHONHASHSEED"] = str(rollout_cfg.seed)


    if not ray.is_initialized():


        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:

            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))


    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote
class TaskRunnerV1:


    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):


        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = AgentLoopManagerTQ

        self.agent_loop_manager = agent_loop_manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):

        import transfer_queue as tq

        from verl.trainer.ppo.v1 import get_trainer_cls

        custom_trainer_module = config.trainer.v1.get("custom_trainer_module")
        if custom_trainer_module:
            importlib.import_module(custom_trainer_module)
        trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config


        tq.init(config.transfer_queue)
        try:
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
        finally:
            tq.close()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):


    auto_set_device(config)


    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    if config.trainer.use_v1:
        run_ppo(config, task_runner_class=TaskRunnerV1)
    else:
        from verl.trainer.main_ppo_v0 import TaskRunner

        logger.warning(
            "Legacy trainer `main_ppo_v0.py` is deprecated, and wil be removed in v0.9.0."
            "Please set `trainer.use_v1=True` in config to use V1 trainer."
        )
        run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
