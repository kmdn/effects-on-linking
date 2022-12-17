import logging

from wikidata_query.gcn_qa_model import GCN_QA
from wikidata_query.read_data import load_train_datasets
from wikidata_query.utils import log_experiment_settings
from wikidata_query.config import Config


_settings = {
        'model_name' : 'model-20210603-1',
        'epochs' : 3,
        'dataset_size' : 'sample',
        'batch_size' : 1,
        'dropout' : 1.0
    }

_config = Config(_settings['model_name'], is_test=False)

logging.basicConfig(level=_config.log_level, format=_config.log_format,
        handlers=[logging.FileHandler(_config.log_path), logging.StreamHandler()])
_logger = logging.getLogger()


def train(data, model, saving_dir, name_prefix, epochs=20, batch_size=32):
    datasets = []
    for dataset in data:
        dataset_reshaped = {
                'text': [item['text'] for item in dataset],
                'node_vectors': [item['graph']['vectors'] for item in dataset],
                'item_vector': [item['item_vector'] for item in dataset],
                'question_vectors': [item['question_vectors'] for item in dataset],
                'question_mask': [item['question_mask'] for item in dataset],
                'y': [item['answer'] for item in dataset]
        }
        datasets.append(dataset_reshaped)

    saving_path = f'{saving_dir}/{name_prefix}'
    model.train(datasets, saving_path, epochs, batch_size)


if __name__ == '__main__':
    log_experiment_settings(settings=_settings, is_test=False)
    data = load_train_datasets(_config, _settings['dataset_size'], use_bert=False, use_pbg=False)
    model = GCN_QA(_config, _settings['dropout'])
    train(data, model, _config.dirs['models'],
            name_prefix=_settings['model_name'],
            epochs=_settings['epochs'],
            batch_size=_settings['batch_size']
    )
