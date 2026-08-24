import os
import time
import ray
from transformers import AutoTokenizer
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers
import os

os.environ['XTUNER_USE_LMDEPLOY'] = '1'
os.environ["LMD_SKIP_WARMUP"] = "1"
os.environ["XTUNER_USE_FA3"] = "1"


if __name__ == "__main__":
    ray.init(num_cpus=80, ignore_reinit_error=True)

    MODEL_PATH = '/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B'

    resources_cfg = AcceleratorResourcesConfig(
        accelerator='GPU',
        num_workers=8,
        num_cpus_per_worker=8,
        cpu_memory_per_worker=16 * 1024**3,
    )
    max_prompt_length = 8196
    max_response_length = 256*1024
    context_length = max_prompt_length + max_response_length
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    pg = AutoAcceleratorWorkers.build_placement_group(resources_cfg, name="moe_api_pg")
    rollout_config = RolloutConfig(
            env="test_rollout_api_server_dense",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            tensor_parallel_size=2,
            expert_parallel_size=1,
            context_length=context_length,
            worker_log_dir='./worker_logs',
            dist_port_base=38000,
            api_host="127.0.0.1",
            api_port=28000,
    )
    rollout_controller = ray.remote(RolloutController).remote(rollout_config, pg)
    metadata = ray.get(rollout_controller.get_rollout_metadata.remote(), timeout=1800)
    base_url = metadata["api_server_url"]
    print(f"base_url: {base_url}")

    import time
    time.sleep(1000000000)
