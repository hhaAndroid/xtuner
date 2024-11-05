import argparse
import subprocess
from rich.syntax import Syntax
from rich.panel import Panel
import rich
import os


def parse_args():
    parser = argparse.ArgumentParser(description='aliyun')
    parser.add_argument('job_name', default='job_name')
    parser.add_argument('num_gpu', help='num of gpus')
    parser.add_argument('command')
    parser.add_argument('--image',
                        default='pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/huanghaian:hha')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    job_name = args.job_name
    num_gpus = int(args.num_gpu)
    command = args.command

    workspace_id = 'ws1ypm687hjzj222'  # llm_razor
    if num_gpus >= 8:
        assert num_gpus % 8 == 0
        worker_count = num_gpus // 8
        _num_gpus = 8
    else:
        worker_count = 1
        _num_gpus = num_gpus
    dlc_config_path = os.environ.get('DLC_CONFIG_PATH', '/root/.dlc/config')
    worker_gpu = min(_num_gpus, 8)
    worker_cpu = worker_gpu * 12
    worker_memory = min(num_gpus * 80, 800)
    worker_shared_memory = min(num_gpus * 20, 200)
    worker_image = args.image

    dlc_command = [
        '~/dlc create job --kind PyTorchJob',
        f'--command "{command}"',
        f'--config {dlc_config_path}',
        f'--name {job_name}',
        f'--workspace_id {workspace_id}',
        f'--worker_count {worker_count}',
        f'--worker_gpu {worker_gpu}',
        f'--worker_cpu {worker_cpu}',
        f'--worker_memory {worker_memory}',
        f'--worker_shared_memory {worker_shared_memory}',
        f'--worker_image {worker_image}'
    ]
    command = ' \\\n  '.join(dlc_command)
    rich.print(Panel(Syntax(command, 'bash', word_wrap=True, background_color="default"), expand=False))
    subprocess.run(command, shell=True, text=True)
