import datetime
import logging
import os


class Config:

    def __init__(self, model_name, is_test):
        self.model_name = model_name
        self.is_test = is_test

        # directories
        self.dirs = {
            'logging' : os.path.join(os.getcwd(), 'log'),
            'models' : os.path.join(os.getcwd(), 'data', 'models'),
            'datasets' : os.path.join(os.getcwd(), '..', 'dataset')
            }

        # create directories
        for path in self.dirs.values():
            if not os.path.exists(path):
                print(f'Create directory {path}')
                os.makedirs(path)

        # datasets
        # see self.get_dataset_path()

        # logging
        log_suffix = 'test' if self.is_test else 'train'
        self.log_level = logging.INFO
        self.log_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_filename = f'{self.log_timestamp}-{self.model_name}-{log_suffix}'
        self.log_path = os.path.join(self.dirs['logging'], f'{self.log_filename}.log')
        self.log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"


        self.is_relevant = [.0, 1.]
        self.is_not_relevant = [1., 0.]


    def get_dataset(self, train, part=None):
        if not part or part == 'full':
            return os.path.join(self.dirs['datasets'], f'wikidata-disambig-{train}.json')
        else:
            return os.path.join(self.dirs['datasets'], f'wikidata-disambig-{train}.{part}.json')
