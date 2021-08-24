# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . agent import Agent


class SpeechToTextAgent(Agent):
    data_type = "speech"
    speech_segment_size = 10

class SpeechToSpeechAgent(SpeechToTextAgent):
    data_type = "speech_to_speech"