import argparse
from transformers import AutoTokenizer
import time
from xtuner.v1.utils import Config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test dataloader')
    parser.add_argument('--cfg', type=str, default='/mnt/shared-storage-user/huanghaian/code/temp/xtuner/examples/v1/cpt_qwen3vl_8b_config.py')
    args = parser.parse_args()

    trainer_cfg = Config.fromfile(args.cfg)['trainer']

    tokenizer = AutoTokenizer.from_pretrained(trainer_cfg.tokenizer_path, trust_remote_code=True)

    dataloader = trainer_cfg.dataloader_cfg.build(
        tokenizer=tokenizer,
        dp_mesh=None,
        global_batch_size=1,
        micro_batch_size=1,
        seed=42
    )
    
    time_before_get_data = time.time()
    for data in dataloader:
        time_before_train_step = time.time()
        data_time = time_before_train_step - time_before_get_data
        if int(data_time)>10:
            print(f"data_time: {data_time:.4f}")
            print(data['types'])
        time_before_get_data=time_before_train_step
    
    
    # for i, data in enumerate(dataloader):
    #     time_before_train_step = time.time()
    #     data_time = time_before_train_step - time_before_get_data
    #     # if data_time>6:
    #     #     print(f"data_time: {data_time:.4f},{data[0]['types']}")
    #     time_before_get_data = time_before_train_step
        
    #     seq_ctx = data[0]['seq_ctx']
    #     step_consumed_img_tokens = sum(seq_ctx.num_img_tokens)
    #     step_consumed_tokens = seq_ctx.input_ids.shape[-1]
    #     print(f'step: {i}, {step_consumed_img_tokens} {step_consumed_tokens}')
    
    

