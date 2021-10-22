# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import os
import math
import soundfile
import sacrebleu
import json

from simuleval import DEFAULT_EOS
from simuleval.metrics.latency import (
    AverageLagging,
    AverageProportion,
    DifferentiableAverageLagging
)


def eval_all_latency(delays, src_len, ref_len=None):
    if ref_len is None:
        ref_len = len(delays)
    results = {}
    for name, func in {
        "AL": AverageLagging,
        "AP": AverageProportion,
        "DAL": DifferentiableAverageLagging
    }.items():
        results[name] = func(delays, src_len, ref_len).item()

    return results


class Instance(object):
    def __init__(
        self,
        instance_id,
        source,
        target,
        args
    ):
        self.finish_source = False
        self.finish_target = False
        self.target = self.preprocess_target(target)
        self.source = self.preprocess_source(source)
        self.step = 0
        self.elapsed = []
        self.hypos = []
        self.delays = []
        self.start_time = None
        self.metrics = {}
        self.instance_id = instance_id
        self.eval_latency_unit = args.eval_latency_unit

    @property
    def finish(self):
        return self.finish_target

    @finish.setter
    def finish(self, status: bool):
        if status:
            self.sentence_level_eval()
        self.finish_target = status

    def preprocess_source(self, source: str):
        """
        Preprocess the source,
        for example tokenization for text
        feature extraction for speech
        """
        raise NotImplementedError
    def preprocess_target(self, target: str):
        """
        Preprocess the target, for example tokenization.
        """
        raise NotImplementedError

    def recv_hypo(self, list_hypo: str):
        """
        Handler for sending new segments
        """
        raise NotImplementedError


    def send_src(self, **kwargs):
        """
        Handler for receiving new predictions
        """
        raise NotImplementedError

    def reference(self):
        """
        Reference
        """
        raise NotImplementedError

    def source_length(self):
        raise NotImplementedError

    def target_length(self):
        raise NotImplementedError

    def source_info(self):
        raise NotImplementedError

    def sentence_level_eval(self):
        raise NotImplementedError

    def step_to_delay(self, step):
        return step

    def step_to_elapsed(self, step, current_time):
        return (current_time - self.start_time) * 1000

    def reference_length(self):
        """
        Length of the reference
        """
        raise NotImplementedError

    def summarize(self):
        return {
            "index": self.instance_id,
            "prediction": self.prediction(),
            "delays": self.delays,
            "elapsed": self.elapsed,
            "prediction_length": self.target_length(),
            "reference": self.reference(),
            "source": self.source_info(),
            "source_length": self.source_length(),
            "reference_length": self.reference_length(),
            "metric": self.metrics,
        }


def text_input_instance(klass):
    """
    Decorator for text input instances
    """
    class TextInputInstance(klass):
        def preprocess_source(self, source):
            return source.strip().split()

        def source_length(self):
            return len(self.source)

        def source_info(self):
            return " ".join(self.source)

        def send_src(self, **kwargs):
            if self.step == 0:
                self.start_time = time.time()

            if self.step >= self.source_length():
                dict_to_return = {"segment_id": self.step, "segment": DEFAULT_EOS}
                # Consider EOS
                self.step = self.source_length() + 1
            else:
                dict_to_return = {"segment_id": self.step,
                                "segment": self.source[self.step]}
                self.step += 1

            return dict_to_return
    return TextInputInstance


def speech_input_instance(klass):
    """
    Decorator for speech input instances
    """
    class SpeechInputInstance(klass):
        def preprocess_source(self, source):
            # Only get info (sample rate),
            # read audio file when first read request happens
            self.audio_info = soundfile.info(source)
            self.sample_rate = self.audio_info.samplerate
            self.samples = []
            return source

        def send_src(self, segment_size=10):
            if self.step == 0:
                self.start_time = time.time()
                self.load_audio_from_path(self.source)
            assert segment_size >= 1, "instance size has to be larger than 1 ms"

            num_samples = math.ceil(segment_size / 1000 * self.sample_rate)

            if self.step < len(self.samples):
                if self.step + num_samples > len(self.samples):
                    # Pad zeros if the requested number of samples
                    # are more than available samples.
                    instance = (
                        self.samples[self.step:]
                    )
                    is_finished = True
                else:
                    instance = self.samples[self.step: self.step + num_samples]
                    is_finished = False

                self.step = min(self.step + num_samples, len(self.samples))

                dict_to_return = {
                    "segment_id": self.len_sample_to_ms(self.step),
                    "segment": instance,
                    "sample_rate": self.audio_info.samplerate,
                    "dtype": "int16",
                    "finished": is_finished,
                }

            else:
                # Finish reading this audio
                dict_to_return = {
                    "segment_id": self.source_length(),
                    "segment": DEFAULT_EOS,
                    "sample_rate": self.audio_info.samplerate,
                    "dtype": "int16",
                    "finished": True,
                }

            return dict_to_return

        def load_audio_from_path(self, wav_path):
            assert os.path.isfile(wav_path) and wav_path.endswith('.wav')
            samples, _ = soundfile.read(wav_path, dtype="int16")
            self.samples = samples.tolist()

        def source_length(self):
            # In milliseconds
            return self.len_sample_to_ms(len(self.samples))

        def source_info(self):
            return str(self.audio_info).split("\n")

        def len_sample_to_ms(self, length):
            assert getattr(self, "sample_rate", None), "Read a audio file first"
            return length * 1000 / self.sample_rate

        def len_ms_to_samples(self, length):
            assert getattr(self, "sample_rate", None), "Read a audio file first"
            return math.ceil(length / 1000 * self.sample_rate)

        def step_to_delay(self, step):
            return self.len_sample_to_ms(self.step)

        def step_to_elapsed(self, step, current_time):
            return self.len_sample_to_ms(step) + (current_time - self.start_time) * 1000

    return SpeechInputInstance


def text_output_instance(klass):
    class TextOutputInstance(klass):
        def recv_hypo(
            self,
            list_hypo: str,
            latency_unit: str = "word"
        ):
            """
            Handler for receiving new predictions
            """
            if self.finish:
                return

            if self.start_time is None:
                self.start_time = time.time()

            current_time = time.time()

            for hypo in list_hypo:
                self.hypos.append(hypo)
                if latency_unit == "word" or hypo in [DEFAULT_EOS]:
                    self.elapsed.append(self.step_to_elapsed(self.step, current_time))
                    self.delays.append(self.step_to_delay(self.step))
                elif latency_unit == "char":
                    self.elapsed += [self.step_to_elapsed(self.step, current_time)] * len(hypo)
                    self.delays += [self.step_to_delay(self.step)] * len(hypo)
                else:
                    raise NotImplementedError
                if hypo in [DEFAULT_EOS]:
                    self.finish = True
                    return

        def preprocess_target(self, target: str):
            return target.strip().split()

        def prediction(self, eos=True, no_space=False):
            join_char = "" if no_space else " "
            if eos:
                return join_char.join(self.hypos)
            else:
                return join_char.join(x for x in self.hypos if x != DEFAULT_EOS)

        def reference_length(self):
            if self.eval_latency_unit == "word":
                return len(self.reference().split(" "))
            elif self.eval_latency_unit == "char":
                return len(self.reference().replace(" ", ""))
            else:
                raise NotImplementedError

        def sentence_level_eval(self, src_eos=True):
            self.metrics["sentence_bleu"] = sacrebleu.sentence_bleu(
                self.prediction(), [self.reference()]
            ).score
            self.metrics["latency"] = eval_all_latency(
                self.delays,
                self.source_length() + int(src_eos),
                self.reference_length() + 1
            )

        def reference(self):
            return " ".join(self.target)

    return TextOutputInstance


def speech_output_instance(klass):
    class SpeechOutputInstance(klass):
        def __init__(
            self,
            instance_id,
            source,
            target,
            args,
        ):
            super().__init__(
                instance_id,
                source,
                target,
                args,
            )
            self.durations = []
            self.output = args.output
            self.output_sample_rate = 16000 #TODO: make it configurable
            self.output_path = os.path.join(self.output, "generated_wavs",f"{self.instance_id}_pred.wav")
            os.makedirs(os.path.join(self.output, "generated_wavs"), exist_ok=True)

        def recv_hypo(
            self,
            list_hypo: str,
        ):
            """
            Handler for receiving new predictions
            """
            if self.finish:
                return

            if self.start_time is None:
                self.start_time = time.time()

            current_time = time.time()

            for hypo in list_hypo:
                if hypo in [DEFAULT_EOS]:
                    self.finish = True
                    return
                self.hypos.append(json.loads(hypo))
                self.durations.append(
                    1000 * len(self.hypos[-1]) / self.sample_rate
                )
                self.elapsed.append((current_time - self.start_time) * 1000)
                self.delays.append(self.step_to_delay(self.step))

        def preprocess_target(self, target: str):
            #TODO let's leave it for now
            if target is not None:
                return target.strip()
            else:
                return ""

        def sentence_level_eval(self):

            def insert_silence(sample, duration_in_ms):
                sample += [0.0] * int(duration_in_ms / 1000 * 16000)
            # The duration of source speech used for each target segment
            source_durations = self.delays
            target_durations = self.durations
            current_time = 0
            samples = []
            for i in range(len(source_durations)):
                computation_time = (
                    self.elapsed[i] if i == 0
                    else self.elapsed[i] - self.elapsed[i-1]
                )
                if current_time < source_durations[i]:
                    insert_silence(samples, computation_time + source_durations[i] - current_time)
                    current_time += (computation_time + source_durations[i])

                samples += self.hypos[i]
                current_time += target_durations[i]

            soundfile.write(
                os.path.join(self.output, "generated_wavs",f"{self.instance_id}_pred.wav"),
                samples,
                self.output_sample_rate,
            )

        def summarize(self):
            return {
                "index": self.instance_id,
                "delays": self.delays,
                "elapsed": self.elapsed,
                "generated_wav": os.path.abspath(self.output_path),
                "duration": self.durations,
            }

        def prediction(self):
            return []

    return SpeechOutputInstance

@text_input_instance
@text_output_instance
class TextToTextInstance(Instance):
    pass

@speech_input_instance
@text_output_instance
class SpeechToTextInstance(Instance):
    pass

@speech_input_instance
@speech_output_instance
class SpeechToSpeechInstance(Instance):
    pass

@text_input_instance
@speech_output_instance
class TextToSpeechInstance(Instance):
    pass
