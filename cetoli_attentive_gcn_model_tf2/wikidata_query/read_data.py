import logging
import numpy as np
import os
import random
import requests

from pathlib import Path
from gensim.models import KeyedVectors

from wikidata_query.utils import infer_vector_from_word, infer_vector_from_doc
from wikidata_query.utils import get_words
from wikidata_query.utils import _is_relevant
from wikidata_query.utils import _is_not_relevant
#from wikidata_query.wikidata_items import wikidata_items
from wikidata_query.sentence_processor import get_adjacency_matrices_and_vectors_given_triplets
from wikidata_query.glove import GloveModel
from wikidata_query.wikidata_items import WikidataItems

_path = os.path.dirname(__file__)

cache_dir = os.path.join(_path, '../../data/triple_cache/')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

_logger = logging.getLogger(__name__)
_logging_level = logging.INFO
logging.basicConfig(level=_logging_level, format="%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s")

_fast_mode = 0
#_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../../data/glove_2.2M.txt'))
_model = GloveModel(_path, _fast_mode, _logger)
_wikidata_items = WikidataItems(_path, _fast_mode, _logger)
#_wikidata_items = wikidata_items
_query = '''
SELECT ?rel ?item ?rel2 ?to_item {
  wd:%s ?rel ?item
  OPTIONAL { ?item ?rel2 ?to_item }
  FILTER regex (str(?item), '^((?!statement).)*$') .
  FILTER regex (str(?item), '^((?!https).)*$') .
} LIMIT 1500
'''


def get_wikidata_id_from_wikipedia_id(wikipedia_id):
    url = 'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&pageids=%s&format=json' % str(
        wikipedia_id)
    try:
        return requests.get(url).json()['query']['pages'][str(wikipedia_id)]['pageprops']['wikibase_item']
    except:
        return ''


def get_graph_from_wikidata_id(wikidata_id, central_item):
    triplets = []
    # NEW Check cache first
    cache_file = cache_dir + wikidata_id + ".nt"
    if os.path.exists(cache_file):
        _logger.debug(f'Read {wikidata_id} from cache')
        with open(cache_file) as cache:
            for line in cache:
                from_item, relation, to_item = line.strip("\n").split("\t")
                triplets.append((from_item, relation, to_item))
    elif _fast_mode >= 2:
        _logger.debug(f'{wikidata_id} not in cache, skipping (fast mode)')
    else:
        url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
        response = requests.get(url, params={'query': _query % wikidata_id, 'format': 'json'})

        if not response.status_code == 200:
            raise Exception(f'Request for {wikidata_id} failed (status code {response.status_code} ({response.reason}))')

        try:
            data = response.json()
        except Exception as e:
            _logger.debug(f'Failed to read JSON response:')
            _logger.debug(f'{response.text}')
            raise e

        triplets = []
        _logger.debug(f'Adding neighbor triples for {wikidata_id}:')
        for item in data['results']['bindings']:
            try:
                from_item = _wikidata_items.translate_from_url(wikidata_id)
                relation = _wikidata_items.translate_from_url(item['rel']['value'])
                to_item = _wikidata_items.translate_from_url(item['item']['value'])
                triplets.append((from_item, relation, to_item))
                _logger.debug(f'Added ({from_item}, {relation}, {to_item}) for {wikidata_id}')
            except Exception as e:
                _logger.debug(f'Skipped ({wikidata_id}, {item["rel"]["value"]}, {item["item"]["value"]}) because of:')
                _logger.debug(f'{e.__class__.__name__}: {str(e)}')
            try:
                from_item = _wikidata_items.translate_from_url(item['item']['value'])
                relation = _wikidata_items.translate_from_url(item['rel2']['value'])
                to_item = _wikidata_items.translate_from_url(item['to_item']['value'])
                triplets.append((from_item, relation, to_item))
                _logger.debug(f'Added ({from_item}, {relation}, {to_item}) for {wikidata_id}')
            except Exception as e:
                _logger.debug(f'Skipped ({item["item"]["value"]}, {item["rel2"]["value"]}, {item["to_item"]["value"]}) because of:')
                _logger.debug(f'{e.__class__.__name__}: {str(e)}')
        triplets = sorted(list(set(triplets)))

        # NEW Caching
        with open(f'{cache_dir}{wikidata_id}.nt', "w+") as cache_file:
            for triple in triplets:
                line = '\t'.join(triple) + '\n'
                cache_file.write(line)

    if not triplets:
        raise RuntimeError(f'The graph of {wikidata_id} contains no suitable triplets')

    _logger.info(f'Added {len(triplets)} triplets from the graph of {wikidata_id}')
    return get_adjacency_matrices_and_vectors_given_triplets(triplets, central_item, _model)


def convert_text_into_vector_sequence(model, text):
    words = get_words(text)
    vectors = []
    for word in words:
        vectors.append(infer_vector_from_word(model, word))
    return vectors


def get_item_mask_for_words(text, item):
    words = get_words(text)
    types = []
    words_in_item = get_words(item.lower())
    for word in words:
        types.append([1. if word.lower() in words_in_item else 0.] * 200)
    return types


def get_data(filename, offset, limit):
    with open(filename) as file:
        lines = file.readlines()[2 * offset:2 * limit]
        data = []
        for i in range(int(len(lines) / 2)):
            text_item_graph_dict = {}
            text = lines[2 * i].replace('\n', '')
            item, wikipedia_id = lines[2 * i + 1].replace('\n', '').split('\t')
            wikidata_id = get_wikidata_id_from_wikipedia_id(wikipedia_id)
            if wikidata_id:
                try:
                    text_item_graph_dict['text'] = text
                    text_item_graph_dict['item'] = item
                    text_item_graph_dict['wikidata_id'] = wikidata_id
                    text_item_graph_dict['graph'] = get_graph_from_wikidata_id(wikidata_id, item)
                    text_item_graph_dict['item_vector'] = infer_vector_from_doc(_model, item)
                    text_item_graph_dict['question_vectors'] = convert_text_into_vector_sequence(_model, text)
                    text_item_graph_dict['question_mask'] = get_item_mask_for_words(text, item)
                    data.append(text_item_graph_dict)
                except Exception as e:
                    _logger.warning(f'{e.__class__.__name__}: {str(e)}')
    return data


def get_data_and_write_json(filename, offset, limit, json_file):
    import json

    with open(filename) as file:
        lines = file.readlines()[2 * offset:2 * limit]
        for i in range(int(len(lines) / 2)):
            text_item_graph_dict = {}
            text = lines[2 * i].replace('\n', '')
            item, wikipedia_id = lines[2 * i + 1].replace('\n', '').split('\t')
            wikidata_id = get_wikidata_id_from_wikipedia_id(wikipedia_id)
            if wikidata_id:
                try:
                    text_item_graph_dict['text'] = text
                    text_item_graph_dict['item'] = item
                    text_item_graph_dict['wikidata_id'] = wikidata_id
                    text_item_graph_dict['graph'] = get_graph_from_wikidata_id(wikidata_id, item)
                    text_item_graph_dict['item_vector'] = infer_vector_from_doc(_model, item)
                    text_item_graph_dict['question_vectors'] = convert_text_into_vector_sequence(_model, text)
                    text_item_graph_dict['question_mask'] = get_item_mask_for_words(text, item)
                    negative_wiki_id = get_wikidata_id_of_item_different_from_given_one(text_item_graph_dict['item'],
                                                                                        text_item_graph_dict[
                                                                                            'wikidata_id'])
                    get_graph_from_wikidata_id(negative_wiki_id, text_item_graph_dict['item'])
                    item = {}
                    item['text'] = text_item_graph_dict['text']
                    item['string'] = text_item_graph_dict['item']
                    item['correct_id'] = text_item_graph_dict['wikidata_id']
                    item['wrong_id'] = negative_wiki_id
                    json.dump(item, json_file, indent=2, sort_keys=True)
                    json_file.write(',\n')

                except Exception as e:
                    _logger.warning(f'{e.__class__.__name__}: {str(e)}')


def infer_vector_from_vector_nodes(vector_list):
    vector = np.zeros(300)
    for v in vector_list:
        vector += v
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def create_text_item_graph_dict(text, item, wikidata_id):
    text_item_graph_dict = {}
    text_item_graph_dict['text'] = text
    text_item_graph_dict['item'] = item
    text_item_graph_dict['wikidata_id'] = wikidata_id
    text_item_graph_dict['graph'] = get_graph_from_wikidata_id(wikidata_id, item)
    # text_item_graph_dict['item_vector'] = infer_vector_from_doc(_model, item)
    text_item_graph_dict['item_vector'] = infer_vector_from_vector_nodes(text_item_graph_dict['graph']['vectors'])
    text_item_graph_dict['question_vectors'] = convert_text_into_vector_sequence(_model, text)
    text_item_graph_dict['question_mask'] = get_item_mask_for_words(text, item)
    return text_item_graph_dict


def get_json_data_many_wrong_ids(json_data):
    data = []
    for json_item in json_data:
        try:
            text = json_item['text']
            item = json_item['string']
            wikidata_id = json_item['positive_id']
            text_item_graph_dict = create_text_item_graph_dict(text, item, wikidata_id)
            text_item_graph_dict['answer'] = _is_relevant
            data.append(text_item_graph_dict)
            for wikidata_id in json_item['negative_ids']:
                text_item_graph_dict = create_text_item_graph_dict(text, item, wikidata_id)
                text_item_graph_dict['answer'] = _is_not_relevant
                data.append(text_item_graph_dict)
        except Exception as e:
            _logger.warning(f'{e.__class__.__name__}: {str(e)}')
    return data


def get_json_data(json_data):
    data = []
    count_all = len(json_data)
    item_count = 0
    for json_item in json_data:
        item_count += 1
        _logger.info(f'')
        _logger.info(f'Item {item_count}/{count_all}')
        try:
            text = json_item['text']
            item = json_item['string']

            wikidata_id = json_item['correct_id']
            text_item_graph_dict = create_text_item_graph_dict(text, item, wikidata_id)
            text_item_graph_dict['answer'] = _is_relevant
            data.append(text_item_graph_dict)

            wikidata_id = json_item['wrong_id']
            text_item_graph_dict = create_text_item_graph_dict(text, item, wikidata_id)
            text_item_graph_dict['answer'] = _is_not_relevant
            data.append(text_item_graph_dict)
        except Exception as e:
            _logger.warning(f'{e.__class__.__name__}: {str(e)}')
    return data


def get_wikidata_id_of_item_different_from_given_one_with_boundaries(item_str,
                                                                     wikidata_id,
                                                                     min_number_of_negative_items=1,
                                                                     max_number_of_negative_items=1):
    items = _wikidata_items.reverse_lookup(item_str)
    items = list(set(items))
    del items[items.index(wikidata_id)]
    if not items:
        raise RuntimeWarning('No negative items!')
    if len(items) < min_number_of_negative_items:
        raise RuntimeWarning('Not enough negative items!')
    if len(items) > max_number_of_negative_items:
        raise RuntimeWarning('Too many negative items!')
    return items


def get_wikidata_id_of_item_different_from_given_one(item_str,
                                                     wikidata_id):
    items = _wikidata_items.reverse_lookup(item_str)
    items = list(set(items))
    del items[items.index(wikidata_id)]
    if not items:
        raise RuntimeWarning('No negative items!')
    return random.choice(items)
