# Entity Disambiguation with Knowledge Graph Structure

This are four entity disambiguation models incorporating different levels of topological graph structure from Wikidata into the entity embeddings.
For each model there is a version with BERT and a version with a Bi-GRU as mention encoder.
There is one folder for each model named with <tt>&lt;MENTION ENCODER&gt;-&lt;ENTITY ENCODER&gt;</tt>.

These models extend the work of Cetoli et al. (2019) ([GitHub](https://github.com/ContextScout/ned-graphs), [paper](https://arxiv.org/pdf/1810.09164.pdf)).
The folders of the original models have the prefix <tt>cetoli\_</tt>.

## Installation

1. Create virtual environment

```bash
python -m venv ve
source ./ve/bin/activate
pip install -r requirements.txt
```

2. Download mapping of Wikidata nodes from [GitHub](https://github.com/ContextScout/ned-graphs/tree/master/data) and extract it to <tt>./data</tt>:

```bash
bunzip2 x*
cat x* > wikidata_items.csv
```

3. Download Wikidata-Disamb dataset from [GitHub](https://github.com/ContextScout/ned-graphs/tree/master/dataset) and copy it to <tt>./data/wikidata_disamb</tt>

4. Download GloVe to <tt>./data/glove</tt>

```bash
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
echo "2196017 300" | cat - glove.840B.300d.txt > glove_2.2M.txt
```

5. Download PyTorch-BigGraph embeddings to <tt>./data/pbg</tt>

```bash
wget https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1_names.json.gz
gunzip wikidata_translation_v1_names.json.gz
wget https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1_vectors.npy.gz
gunzip wikidata_translation_v1_vectors.json.gz
```

## Train and test models

To train a model change to the directory of the model and execute <tt>train.py</tt>

```bash
cd ./MODEL_NAME
python -m wikidata_query.train
```

To test a model, execute <tt>test.py</tt>

```bash
cd ./MODEL_NAME
python -m wikidata_query.test
```
