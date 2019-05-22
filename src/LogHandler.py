from tensorboardX import SummaryWriter

class LogHandler:
    def __init__(self, logdir, module_id):
        self.logdir = logdir
        self.log = SummaryWriter(self.logdir)
        self.module_id = module_id

    def scalar(self, key, val, step):
        self.log.add_scalar('{}_{}'.format(self.module_id, key), val, step)

    def text(self, key, val, step):
        self.log.add_text('{}_{}'.format(self.module_id, key), val, step)

    def image(self, key, val, step):
        self.log.add_image('{}_{}'.format(self.module_id, key), val, step)
    
    def figure(self, key, val, step):
        self.log.add_figure('{}_{}'.format(self.module_id, key), val, step)

