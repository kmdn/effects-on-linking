import json
import logging
import os
import numpy as np
from tqdm import tqdm

class PBG:
    _url_prefix = 'http://www.wikidata.org/entity/'
    _names = None # names (Q123)
    _vectors = None # mmap of vectors, ordered but without index
    _name_index_dict = {} # index (Q123 -> PBG index)
    _vectors_dict = {} # memory (Q123 -> vector)


    def __init__(self, path, sample_mode=False, use_cache=False):
        self._dir = os.path.join(path, '..', '..', 'data', 'pbg')
        self._cache_dir = os.path.join(self._dir, 'cache')
        self._logger = logging.getLogger(__name__)
        self._sample_mode = sample_mode
        self._use_cache = use_cache

        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)


    def __getitem__(self, item):
        return self._vectors_dict[item]


    def __setitem__(self, index, value):
        self._vectors_dict[index] = value


    def __load_pbg(self):
        self.__read_names()
        self.__read_vectors()
        self.__create_names_index()


    def __read_names(self):
        if self._sample_mode:
            self._logger.info('Reading PyTorch Big Graph Wikidata item names (sample)...')
            f_names = os.path.join(self._dir, 'wikidata_translation_v1_names.sample.json')
        else:
            self._logger.info('Reading PyTorch Big Graph Wikidata item names...')
            f_names = os.path.join(self._dir, 'wikidata_translation_v1_names.json')

        with open(f_names, 'r') as myfile:
            data = myfile.read()
        self._names = json.loads(data)
        self._logger.info(f'Read {len(self._names)} names')


    def __read_vectors(self):
        self._logger.info('Reading PyTorch Big Graph vectors...')
        f_vectors = 'data/pbg/'
        f_vectors = os.path.join(self._dir, 'wikidata_translation_v1_vectors.npy')
        self._vectors = np.load(f_vectors, mmap_mode='r') # mmap_mode for not loading the file into memory
        self._logger.info('Read vectors (mmap)')


    def __create_names_index(self):
        self._logger.info('Creating PyTorch Big Graph names index...')
        i = 0
        for name in tqdm(self._names):
            self._name_index_dict[name] = i
            i += 1
        self._logger.info('Created index')


    def __get_uri_of_item_id(self, item_id):
        return f'<{self._url_prefix}{item_id}>'


    def __get_index_by_item_id(self, item_id):
        item_uri = self.__get_uri_of_item_id(item_id)
        try:
            return self._name_index_dict[item_uri]
        except KeyError:
            return None


    def __get_vector_from_pbg(self, item_id):
        if self._vectors is None or self._names is None:
            self.__load_pbg()

        item_index = self.__get_index_by_item_id(item_id)
        if item_index is None:
            # There is definetly no embedding available
            # Store zero vector
            item_vector = np.zeros(200)
            self.__store_vector_in_memory(item_id, item_vector)
            self.__save_vector_to_cache(item_id, item_vector)
            # Exception leads to skipping in the preprocessor/entity disambiguator
            raise ValueError(f'"{item_id}" has no embedding')
        return self._vectors[item_index]


    def __get_vector_from_cache(self, item_id):
        cache_file = f'{self._cache_dir}{item_id}.txt'
        if not os.path.exists(cache_file):
            self._logger.debug(f'Embedding of "{item_id}" not in cache')
            return None
        item_vector = np.loadtxt(cache_file)
        if not item_vector.any(): # zero vector
            self._logger.debug(f'Read zero vector (200,) from cache for "{item_id}"')
            # Exception leads to skipping in the preprocessor/entity disambiguator
            raise ValueError(f'"{item_id}" has no embedding')
        self._logger.debug(f'Read embedding of "{item_id}" from cache')
        return item_vector


    def __save_vector_to_cache(self, item_id, item_vector):
        cache_file = f'{self._cache_dir}{item_id}.txt'
        try:
            np.savetxt(cache_file, item_vector)
            self._logger.debug(f'Wrote embedding of "{item_id}" to cache')
        except FileNotFoundError:
            self._logger.info(f'Could not cache "{item_id}". Tried to cache word with slash? ({str(e)})')


    def __get_vector_from_memory(self, item_id):
        try:
            item_vector = self[item_id]
            self._logger.debug(f'Read embedding of "{item_id}" from memory')
        except KeyError:
            self._logger.debug(f'Embedding of "{item_id}" not in memory')
            return None
        if not item_vector.any(): # zero vector
            self._logger.debug(f'Read zero vector (200,) from memory for "{item_id}"')
            # Exception leads to skipping in the preprocessor/entity disambiguator
            raise ValueError(f'"{item_id}" has no embedding')
        return item_vector


    def __store_vector_in_memory(self, item_id, item_vector):
        self[item_id] = item_vector
        self._logger.debug(f'Stored embedding of "{item_id}" in memory')


    def get_item_embedding(self, item_id):
        # Try memory (dict) of PBG object
        item_vector = self.__get_vector_from_memory(item_id)

        if item_vector is not None:
            return item_vector

        # Try cache (files in cache directory, one for each item_id)
        if self._use_cache:
            item_vector = self.__get_vector_from_cache(item_id)

            if item_vector is not None:
                self.__store_vector_in_memory(item_id, item_vector)
                return item_vector

        # Try PBG file (requires loading the whole PGB embeddings file)
        item_vector = self.__get_vector_from_pbg(item_id)

        if item_vector is not None:
            self.__store_vector_in_memory(item_id, item_vector)
            if self._use_cache:
                self.__save_vector_to_cache(item_id, item_vector)
            return item_vector

        return item_vector

