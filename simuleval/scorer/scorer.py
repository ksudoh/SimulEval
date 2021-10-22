# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sacrebleu
from . instance import (
    TextToTextInstance,
    SpeechToTextInstance,
    SpeechToSpeechInstance
)
import os
import sys
import logging
from statistics import mean
import soundfile as sf

logger = logging.getLogger('simuleval.scorer')

def build_scorer(args):
    return Scorer(args)

def load_text_file(file_name):
    with open(file_name) as f:
        return [r.strip()for r in f]


class Scorer(object):
    def __init__(self, args):
        self.args = args
        self.no_score = args.no_score
        self.data = {
            "src": load_text_file(args.source),
            "tgt": load_text_file(args.target) if args.target is not None else None
        }

        if type(args.data_type) is list:
            if len(args.data_type) == 2:
                self.data_type = {"source": args.data_type[0], "target": args.data_type[1]}
            elif len(args.data_type) == 1:
                self.data_type = {"source": args.data_type[0], "target": args.data_type[0]}
            else:
                logger.error(
                    "Number of arguments for --data-type is wrong, should be 1 or 2."
                    f"{len(self.data_type)} is given."
                )
                sys.exit(1)
        else:
            self.data_type = args.data_type

        logger.info(f"Evaluating on {self.data_type}")
        logger.info(f"Source: {os.path.abspath(args.source)}, type:{self.data_type['source']}")
        if self.data["tgt"] is not None:
            logger.info(f"Target: {os.path.abspath(args.target)}, type:{self.data_type['target']}")
        else:
            logger.error(
                "Target: no reference file is given. Will not score the output."
            )
            self.no_score = True
            self.data["tgt"] = [None for _ in range(len(self))]
        logger.info(f"Number of sentences: {len(self)}")

        self.instances = {}


        #self.eval_latency_unit = args.eval_latency_unit
        self.sacrebleu_tokenizer = args.sacrebleu_tokenizer
        self.no_space = args.no_space

        self.reset()

    def instance_class(self):
        # Instance class for scorer
        if self.data_type['source'] == "text" and self.data_type['target'] == "text":
            return TextToTextInstance
        elif self.data_type['source'] == "speech" and self.data_type['target'] == "text":
            return  SpeechToTextInstance
        elif self.data_type['source'] == "speech" and self.data_type['target'] == "speech":
            return  SpeechToSpeechInstance
        else:
            logger.error(
                f"{self.data_type} is not supported."
            )
            sys.exit(1)

    def get_info(self):
        return {
            "num_sentences": len(self),
            "data_type": self.data_type
        }

    def send_src(self, instance_id, segment_size):
        dict_to_return = (
            self.instances[instance_id]
            .send_src(segment_size=segment_size)
        )
        dict_to_return["instance_id"] = instance_id
        return dict_to_return

    def recv_hyp(self, instance_id, list_of_hypos):
        self.instances[instance_id].recv_hypo(list_of_hypos)

    def reset(self):
        if len(self.instances) > 0:
            logger.warning("Resetting scorer")

        for i, (src, tgt) in enumerate(zip(self.data["src"], self.data["tgt"])):
            self.instances[i] = self.instance_class()(
                i, src, tgt, self.args
            )

    def gather_translation(self):
        not_finish_write_id = [i for i in range(
            len(self)) if not self.instances[i].finish_target]
        empty_hypo_id = [str(i) for i in range(len(self)) if len(
            self.instances[i].prediction()) == 0]

        if len(not_finish_write_id) > 0:
            print(
                "Warning: these hypothesis don't have EOS in predictions",
                file=sys.stderr)
            print(
                ", ".join((str(x) for x in not_finish_write_id)),
                file=sys.stderr
            )
            for idx in not_finish_write_id:
                self.instances[idx].sentence_level_eval()

        if len(empty_hypo_id) > 0:
            print("Warning: these hypothesis are empty", file=sys.stderr)
            print(", ".join(empty_hypo_id), file=sys.stderr)

        translations = [self.instances[i].prediction(
            eos=False, no_space=self.no_space) for i in range(len(self))]

        return translations

    def get_quality_score(self):

        translations = self.gather_translation()

        try:
            bleu_score = sacrebleu.corpus_bleu(
                translations,
                [self.data["tgt"]],
                tokenize=self.sacrebleu_tokenizer
            ).score
        except Exception as e:
            print(e, file=sys.stderr)
            bleu_score = 0

        return {"BLEU": bleu_score}

    def get_latency_score(self):
        results = {}
        for metric in ["AL", "AP", "DAL"]:
            results[metric] = mean(
                [seg.metrics["latency"][metric]
                    for seg in self.instances.values()]
            )
            if "latency_ca" in self.instances[0].metrics:
                results[metric + "_CA"] = mean(
                    [seg.metrics["latency_ca"][metric]
                        for seg in self.instances.values()]
                )

        return results

    def score(self):
        if not self.no_score:
            return {
                'Quality': self.get_quality_score(),
                'Latency': self.get_latency_score(),
            }
        else:
            return None

    def __len__(self):
        return len(self.data["src"])
