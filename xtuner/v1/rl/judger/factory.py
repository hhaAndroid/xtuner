from xtuner.v1.rl.utils import register_cpu_resources

from .native import Judger, JudgerConfig, JudgerPool


def build_judger(config: JudgerConfig) -> Judger:
    if config.cpu_resources is None:
        return config.build_local()

    register_cpu_resources(
        name=f"judger:{config.judger_name}",
        cpu_resources=config.cpu_resources,
    )

    if config.cpu_resources.num_workers == 1:
        return config._build_remote_judger(cpu_resources=config.cpu_resources)
    return JudgerPool(
        replicas=config._build_remote_judgers(cpu_resources=config.cpu_resources),
        judger_name=config.judger_name,
        is_batch_judger=config.is_batch_judger,
    )
