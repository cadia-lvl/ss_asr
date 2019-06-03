from tensorboardX import SummaryWriter

class LogHandler:
    def __init__(self, logdir, module_id):
        self.logdir = logdir
        self.log = SummaryWriter(self.logdir)
        self.module_id = module_id

    def scalar(self, key, val, step):
        '''
        val can either be a scalar or a dictionary e.g.
        {'a': 3, 'b': 2} to plot e.g. the values of a and b
        onto the same graph
        '''
        if isinstance(val, dict):
            self.log.add_scalars('{}_{}'.format(self.module_id, key), val, step)
        else:
            self.log.add_scalar('{}_{}'.format(self.module_id, key), val, step)

    def text(self, key, val, step):
        self.log.add_text('{}_{}'.format(self.module_id, key), val, step)

    def image(self, key, val, step):
        self.log.add_image('{}_{}'.format(self.module_id, key), val, step)
    
    def figure(self, key, val, step):
        self.log.add_figure('{}_{}'.format(self.module_id, key), val, step)
    
    def embedding(self, key, val, meta, step):
        self.log.add_embedding(val, tag=key, metadata=meta, global_step=step)

