from mmengine.hooks import Hook


class EvaluateMMHook(Hook):
    def before_run(self, runner) -> None:
        runner.logger.info('before_run in EvaluateMMHook.')

        # build val dataset
