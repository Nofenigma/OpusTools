import argparse
import logging
import os
import tempfile
import unittest

from opustools_pkg.filter import tokenization


class TestTokenization(unittest.TestCase):

    def test_dummy(self):
        tokenize = tokenization.get_tokenize(None)
        self.assertEqual(tokenize("Hello, world!"), "Hello, world!")

    def test_moses(self):
        tokenize = tokenization.get_tokenize(('moses', 'en'))
        self.assertEqual(tokenize("Hello, world!"), "Hello , world !")
