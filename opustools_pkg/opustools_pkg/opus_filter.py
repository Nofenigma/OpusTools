import os
import json
import subprocess
import logging
import argparse

from yaml import load, Loader

from . import OpusRead
from .filter.pipeline import FilterPipeline
from .filter import lm
from .filter import word_alignment
from .filter import tokenization
from .util import file_open


logger = logging.getLogger(__name__)


class OpusFilter:
    """Apply filters to language data"""

    def __init__(self, configuration):
        self.configuration = configuration
        self.output_dir = configuration.get('common', {}).get('output_directory')
        if not self.output_dir:
            logger.warning(
                'Output directory not specified. Writing files to current '
                'directory.')
            self.output_dir = '.'
        elif not os.path.isdir(self.output_dir):
            logger.warning(
                'Directory "{}" does not exist. It will be '
                'created.'.format(self.output_dir))
            os.mkdir(self.output_dir)

        self.step_functions = {
                'opus_read': self.read_from_opus,
                'filter': self.clean_data,
                'train_ngram': self.train_ngram,
                'train_alignment': self.train_alignment,
                'score': self.score_data
            }

    def execute_steps(self, overwrite=False):
        """Execute steps in the same order as they are in the configuration"""
        for num, step in enumerate(self.configuration['steps']):
            logger.info('Running step {}: {}'.format(num + 1, step))
            self.step_functions[step['type']](step['parameters'], overwrite=overwrite)

    def read_from_opus(self, parameters, overwrite=False):
        """Download and read a corpus from OPUS"""
        src_out = os.path.join(self.output_dir, parameters['src_output'])
        tgt_out = os.path.join(self.output_dir, parameters['tgt_output'])
        if not overwrite and os.path.isfile(src_out) and os.path.isfile(tgt_out):
            logger.info("Output files exists, skipping step")
            return
        opus_reader = OpusRead('-d {corpus_name} -s {src} -t {tgt} '
            '-r {version} -p {preprocessing} -wm moses '
            '-w {result_dir}/{src_filename} '
            '{result_dir}/{tgt_filename} -ln'.format(
                corpus_name=parameters['corpus_name'],
                src=parameters['source_language'],
                tgt=parameters['target_language'],
                version=parameters['release'],
                preprocessing=parameters['preprocessing'],
                result_dir=self.output_dir,
                src_filename=parameters['src_output'],
                tgt_filename=parameters['tgt_output']).split())

        opus_reader.printPairs()

    def pair_generator(self, source_file_name, target_file_name, src_tokenizer=None, tgt_tokenizer=None):
        """Yield and optionally tokenize sentence pairs from given files"""
        src_tokenize = tokenization.get_tokenize(src_tokenizer)
        tgt_tokenize = tokenization.get_tokenize(tgt_tokenizer)
        with file_open(source_file_name) as source_file, \
                file_open(target_file_name) as target_file:
            for src_line in source_file:
                tgt_line = target_file.readline()
                yield (src_tokenize(src_line.rstrip()), tgt_tokenize(tgt_line.rstrip()))

    def get_pairs(self, src_filename, tgt_filename):
        """Return a generator for given sentence files"""
        source_file_name = '{result_dir}/{src_filename}'.format(
            result_dir=self.output_dir, src_filename=src_filename)
        target_file_name = '{result_dir}/{tgt_filename}'.format(
            result_dir=self.output_dir, tgt_filename=tgt_filename)
        return self.pair_generator(source_file_name, target_file_name)

    def clean_data(self, parameters, overwrite=False):
        """Write sentences to file if they pass given filters"""
        src_out = os.path.join(self.output_dir, parameters['src_output'])
        tgt_out = os.path.join(self.output_dir, parameters['tgt_output'])
        if not overwrite and os.path.isfile(src_out) and os.path.isfile(tgt_out):
            logger.info("Output files exists, skipping step")
            return
        filter_pipe = FilterPipeline.from_config(parameters['filters'])
        pairs_gen = self.get_pairs(parameters['src_input'],
                parameters['tgt_input'])
        pairs = filter_pipe.filter(pairs_gen)
        with file_open(src_out, 'w') as source_file, \
                file_open(tgt_out, 'w') as target_file:
            for pair in pairs:
                source_file.write(pair[0]+'\n')
                target_file.write(pair[1]+'\n')

    def train_ngram(self, parameters, overwrite=False):
        """Train an n-gram language model"""
        model_out = os.path.join(self.output_dir, parameters['model'])
        if not overwrite and os.path.isfile(model_out):
            logger.info("Output file exists, skipping step")
            return
        data_name = parameters['data']
        seg_name = data_name + '.seg.gz'
        tokenizer = lm.LMTokenizer(**parameters['parameters'])
        with file_open(os.path.join(self.output_dir, data_name), 'r') as \
                infile, \
                file_open(os.path.join(self.output_dir, seg_name), 'w') as \
                outfile:
            for line in infile:
                tokens = tokenizer.tokenize(line.strip())
                outfile.write(' '.join(tokens) + '\n')
        lm.train(os.path.join(self.output_dir, seg_name), model_out,
                 **parameters['parameters'])

    def train_alignment(self, parameters, overwrite=False):
        """Train eflomal alignment priors"""
        model_out = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(model_out):
            logger.info("Output file exists, skipping step")
            return
        pair_gen = self.pair_generator(
                os.path.join(self.output_dir, parameters['src_data']),
                os.path.join(self.output_dir, parameters['tgt_data']),
                src_tokenizer=parameters['parameters'].get('src_tokenizer', None),
                tgt_tokenizer=parameters['parameters'].get('tgt_tokenizer', None))
        word_alignment.make_priors(
                pair_gen, model_out, model=parameters['parameters']['model'])

    def score_data(self, parameters, overwrite=False):
        """Score language data based on given filters"""
        score_out = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(score_out):
            logger.info("Output file exists, skipping step")
            return
        for f in parameters['filters']:
            filter_name = next(iter(f.items()))[0]
            if filter_name == 'WordAlignFilter' and 'priors' in f[filter_name]:
                f[filter_name]['priors'] = os.path.join(self.output_dir,
                        f[filter_name]['priors'])
            if filter_name == 'CrossEntropyFilter':
                src_lm_params = f[filter_name]['src_lm_params']
                src_lm_params['filename'] = os.path.join(self.output_dir,
                        src_lm_params['filename'])
                tgt_lm_params = f[filter_name]['tgt_lm_params']
                tgt_lm_params['filename'] = os.path.join(self.output_dir,
                        tgt_lm_params['filename'])

        pairs_gen = self.get_pairs(parameters['src_input'],
                parameters['tgt_input'])
        filter_pipe = FilterPipeline.from_config(parameters['filters'])
        scores_gen = filter_pipe.score(pairs_gen)

        with file_open(score_out, 'w') as score_file:
            for score in scores_gen:
                score_file.write(json.dumps(score, sort_keys=True)+'\n')
