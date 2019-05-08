'''
SuperSolver combines every training solver in src.solver.py and runs each
in turns. The asr core is only saved when it's evaluation loss is decreased
'''
from solver import ASRTrainer, TAETrainer, SAETrainer, AdvTrainer, Solver

class SuperSolver:
    def __init__(self, config, paras):
        
        self.config = config
        self.paras = paras

        self.asr_solver = ASRTrainer(self.config, self.paras)
        self.asr_solver.load_data()
        self.asr_solver.set_model()
        self.asr_model = self.asr_solver.get_asr_model()

        self.tae_solver = TAETrainer(self.config, self.paras)
        self.tae_solver.load_data()
        self.tae_solver.set_model(asr_model=self.asr_model)
        self.tae_model = self.tae_solver.get_tae_model()
        
        self.sae_solver = SAETrainer(self.config, self.paras)
        self.sae_solver.load_data()
        self.sae_solver.set_model(asr_model=self.asr_model)

        self.adv_solver = ADvTrainer(self.config, self.paras)
        self.adv_solver.load_data()
        self.adv_solver.set_model(asr_model=self.asr_model, tae_model=self.tae_model)

        self.superloops = 5
    def train(self):
        # each solver takes care of logging, evaluating, checkpointing
        # and so on
        for i in range(self.superloops):
            self.asr_solver()
            self.tae_solver()
            self.sae_solver()
            self.adv_solver()