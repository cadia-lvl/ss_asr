import os
import json

class TrackerHandler:
    def __init__(self, path, module_id):
        self.path = path
        self.module_id = module_id
        if not os.path.exists(self.path):
            with open(self.path, 'w') as tf:
                tf.write("{}")
        self.data = json.load(open(self.path, 'r'))
        # if module id is not in tracker file, add it.
        if self.module_id not in self.data:
            self.data[self.module_id] = {'best': 10000, 'step': 0}
        
        self.step = self.data[self.module_id]['step']

    def do_step(self):
        self.data[self.module_id]['step'] += 1
        self.step += 1
        self._save()

    def get_best(self):
        '''
        Gets the 'best' value of the metric for the module
        being trained 
        '''
        return self.data[self.module_id]['best']
    
    def set_best(self, val):
        '''
        Sets the best value of the module metric in the
        tracker
        '''
        self.data[self.module_id]['best'] = val
        self._save()

    def _save(self):
        '''
        Saves the current version of the tracker data to
        the assigned tracker path
        '''
        json.dump(self.data, open(self.path, 'w'))