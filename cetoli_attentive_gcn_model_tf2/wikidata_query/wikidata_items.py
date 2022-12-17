import logging
import os


class WikidataItems:

    def __init__(self, dir_path, fast_mode, logger):
        self._logger = logging.getLogger(__name__)
        self._fast_mode = fast_mode
        self._dir = dir_path
        self._cache_dir = os.path.join(dir_path, '../../data/items_cache/')
        self._items_dict = {}
        self._model = None
        #self._reverse_dict = {}

        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)


    def __getitem__(self, item):
        return self._items_dict[item]


    def __setitem__(self, index, value):
        self._items_dict[index] = value


    def __load_items(self):
        if self._fast_mode == 0:
            self._logger.info("Loading Wikidata items")
            self._file_dir = '../../data/wikidata_items.csv'
        else:
            self._logger.info("Loading Wikidata items (sample)")
            self._file_dir = '../../data/wikidata_items.sample.csv'

        self._file_path = os.path.join(self._dir, self._file_dir)

        self._model = {}
        with open(self._file_path, encoding='utf8') as f:
            for item in f.readlines():
                item = item[:-1]
                item_key, item_value = item.split('\t')[:2]
                if ':' in item_value or len(item_value) < 2:
                    continue
                if item_key not in self._model:
                    self._model[item_key] = item_value
                #try:
                #    self._reverse_dict[item_value.lower()].append(item_key)
                #except:
                #    self._reverse_dict[item_value.lower()] = [item_key]
                ## add also string without '.'
                #try:
                #    self._reverse_dict[item_value.lower().replace('.', '')].append(item_key)
                #except:
                #    self._reverse_dict[item_value.lower().replace('.', '')] = [item_key]

        self._logger.info("Wikidata items loaded")


    def __get_item_from_file(self, item_id):
        if self._model is None:
            self.__load_items()

        try:
            return self._model[item_id]
        except:
            self._logger.warning(f'Could not find name for "{item_id}"')
            return None


    def __get_item_from_cache(self, item_id):
        cache_file = f'{self._cache_dir}{item_id}.txt'
        if not os.path.exists(cache_file):
            self._logger.debug(f'No name found in cache for "{item_id}"')
            return None
        with open(cache_file) as cache:
            item_name = cache.readline().strip()
        if not item_name: # empty string
            self._logger.debug(f'Read empty string from cache for "{item_id}"')
            # TODO Add first the empty string to memory cache, then raise exception
            raise Exception(f'"{item_id}" is not a named Wikidata item') # exception leads to passing this item_id/word
        else:
            self._logger.debug(f'Read name of "{item_id}" from cache')
        return item_name


    def __save_item_to_cache(self, item_id, item_name):
        cache_file = f'{self._cache_dir}{item_id}.txt'
        with open(cache_file, "w+") as cache:
            cache.write(item_name)
        self._logger.debug(f'Wrote name of "{item_id}" to cache')


    def __get_item_from_memory(self, item):
        try:
            item_name = self[item]
            self._logger.debug(f'Read name of "{item}" from memory')
            if not item_name: # empty string
                self._logger.debug(f'Read empty string from memory for "{item_id}"')
                raise Exception(f'"{item_id}" is not a named Wikidata item') # exception leads to passing this item_id/word
            return item_name
        except:
            return None


    def __store_item_in_memory(self, item, item_name):
        self[item] = item_name
        self._logger.debug(f'Stored name of "{item}" in memory')


    def get_item(self, item_id):
        item_name = self.__get_item_from_memory(item_id)

        if item_name is not None:
            return item_name

        item_name = self.__get_item_from_cache(item_id)

        if item_name is not None:
            self.__store_item_in_memory(item_id, item_name)
            return item_name

        item_name = self.__get_item_from_file(item_id)

        if item_name is not None:
            self.__store_item_in_memory(item_id, item_name)
            self.__save_item_to_cache(item_id, item_name)
            return item_name

        item_name = ""
        self.__store_item_in_memory(item_id, item_name)
        self.__save_item_to_cache(item_id, item_name)
        raise Exception(f'"{item_id}" is not a named Wikidata item') # exception leads to passing this item_id/word


    def translate_from_url(self, url):
        if '/' in url and '-' not in url:
            item = url.split('/')[-1]
        elif '/' in url and '-' in url:
            item = url.split('/')[-1].split('-')[0]
        else:
            item = url
        return self.get_item(item)

    def reverse_lookup(self, word):
        raise Exception("Not implemented")
        #return self._reverse_dict[word.lower()]

