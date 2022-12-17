import logging
import os

from wikidata_query.gcn_qa_model import GCN_QA
from wikidata_query.read_data import load_test_dataset
from wikidata_query.utils import log_experiment_settings
from wikidata_query.config import Config


_settings = {
        'model_name' : 'model-20210701-1',
        'epochs' : 60,
        'dataset_size' : 'full'
    }

_config = Config(_settings['model_name'], is_test=True)

logging.basicConfig(level=_config.log_level, format=_config.log_format,
        handlers=[logging.FileHandler(_config.log_path), logging.StreamHandler()])
_logger = logging.getLogger()


def test(data, model, epoch):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    count = 0
    count_all = len(data)
    for item in data:
        count += 1
        expected = item['answer']
        text = item['text'].strip()
        mention = item['item']
        wikidata_id = item['wikidata_id']
        node_vectors = item['graph']['vectors']
        item_vector = item['item_vector']
        item_pbg = item['item_pbg']
        question_vectors = item['question_vectors']
        question_mask = item['question_mask']
#        try:
        prediction = model.predict(text, node_vectors, item_vector, item_pbg, question_vectors, question_mask)
        if prediction == expected and expected == _config.is_relevant:
            true_positives += 1
            _logger.info(f'Item {str(count)}/{str(count_all)}: [TP] Predicted {wikidata_id} consistent for "{mention}" in "{text}"')
        if prediction == expected and expected == _config.is_not_relevant:
            true_negatives += 1
            _logger.info(f'Item {str(count)}/{str(count_all)}: [TN] Predicted {wikidata_id} not consistent for "{mention}" in "{text}"')
        if prediction != expected and expected == _config.is_relevant:
            false_negatives += 1
            _logger.info(f'Item {str(count)}/{str(count_all)}: [FN] Predicted {wikidata_id} not consistent for "{mention}" in "{text}"')
        if prediction != expected and expected == _config.is_not_relevant:
            false_positives += 1
            _logger.info(f'Item {str(count)}/{str(count_all)}: [FP] Predicted {wikidata_id} consistent for "{mention}" in "{text}"')
#        except Exception as e:
#            print('Exception caught during training: ' + str(e))
    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * 1 / (1 / precision + 1 / recall)
        _logger.info(f'Epoch {epoch}: Precision {precision}')
        _logger.info(f'Epoch {epoch}: Recall {recall}')
        _logger.info(f'Epoch {epoch}: F1 {f1}')
    except:
        _logger.warning('Cannot compute precision and recall.')
        _logger.warning(str(e))


if __name__ == '__main__':
    log_experiment_settings(settings=_settings, is_test=True)
    data = load_test_dataset(_config, _settings['dataset_size'], use_bert=False, use_pbg=True)
    for epoch in range(1, _settings['epochs'] + 1):
        _logger.info('')
        _logger.info(f'--- Epoch {str(epoch)}/{str(_settings["epochs"])} ---')
        model_path = os.path.join(_config.dirs["models"], _settings['model_name'], f'cp-{epoch:04d}.ckpt')
        model = GCN_QA.load(model_path, _config)
        test(data, model, epoch)
