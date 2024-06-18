import logging
import os
import sys
import warnings
from xtuner.dataset.collate_fns import default_collate_fn
from functools import partial
from xtuner.registry import BUILDER
import transformers
from dist_utils import init_dist
from PIL import Image, ImageFile, PngImagePlugin
from transformers import (Trainer, TrainingArguments, set_seed)
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
import argparse
from mmengine.config import Config
from train_sampler_patch import replace_train_sampler

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

replace_train_sampler()


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    args = parser.parse_args()
    return args


def main():
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')

    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.training_args['output_dir'] = args.work_dir
    training_args = TrainingArguments(**cfg.training_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training parameters {training_args}')

    # Set seed before initializing model.
    set_seed(training_args.seed)
    logger.info('==== start model ====')
    model = BUILDER.build(cfg.model)
    logger.info('==== end model ====')

    # set seed for torch dataloaders
    set_seed(training_args.seed)
    logger.info('==== start train_dataset ====')
    train_dataset = BUILDER.build(cfg.train_dataset)
    logger.info('==== end train_dataset ====')
    # set seed for torch dataloaders
    set_seed(training_args.seed)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=default_collate_fn
    )
    logger.info('==== start trainer ====')
    if training_args.do_train:
        trainer.train()
        trainer.save_model(output_dir=training_args.output_dir)
        trainer.save_state()


if __name__ == '__main__':
    main()
