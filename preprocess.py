# -*- coding: utf-8 -*-

import utils

import nlppreprocess.lowercase
import nlppreprocess.tokenizer
import nlppreprocess.replace_digits
import nlppreprocess.append_eos
import nlppreprocess.split_corpus
import nlppreprocess.create_vocabulary
import nlppreprocess.replace_rare_words
import nlppreprocess.flatten

def main():
    config = utils.Config()

    raw = config.getpath("raw_corpus")
    prep = config.getpath("prep_corpus")
    
    nlppreprocess.lowercase.run(
            raw,
            "tmp.txt.lowercase")
    nlppreprocess.tokenizer.run(
            "tmp.txt.lowercase",
            "tmp.txt.lowercase.tokenize")
    nlppreprocess.replace_digits.run(
            "tmp.txt.lowercase.tokenize",
            "tmp.txt.lowercase.tokenize.replace_digits")
    nlppreprocess.append_eos.run(
            "tmp.txt.lowercase.tokenize.replace_digits",
            "tmp.txt.lowercase.tokenize.replace_digits.append_eos")
    nlppreprocess.split_corpus.run(
            "tmp.txt.lowercase.tokenize.replace_digits.append_eos",
            "tmp.txt.lowercase.tokenize.replace_digits.append_eos.train",
            "tmp.txt.lowercase.tokenize.replace_digits.append_eos.val",
            size=10000)
    nlppreprocess.create_vocabulary.run(
            "tmp.txt.lowercase.tokenize.replace_digits.append_eos.train",
            prep + ".train.vocab",
            prune_at=300000,
            min_count=5,
            special_words=["<EOS>"])
    nlppreprocess.replace_rare_words.run(
            "tmp.txt.lowercase.tokenize.replace_digits.append_eos.train",
            prep + ".train",
            path_vocab=prep + ".train.vocab")
    nlppreprocess.replace_rare_words.run(
            "tmp.txt.lowercase.tokenize.replace_digits.append_eos.val",
            prep + ".val",
            path_vocab=prep + ".train.vocab")
    nlppreprocess.flatten.run(
            prep + ".train",
            prep + ".train.flatten")

    utils.logger.debug("[info] Done.") 

if __name__ == "__main__":
    main()

