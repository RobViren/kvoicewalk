import re
import warnings

import torch
from kokoro import KPipeline

from utilities.kvw_informer import KVW_Informer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class SpeechGenerator:
    def __init__(self, kvw_informer: KVW_Informer, target_text: str, other_text: str):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.kvw_informer = kvw_informer
        self.log_view = kvw_informer.settings['speech_gen_logs']
        self.process_times = self.kvw_informer.settings['tps_reports']

        if self.log_view is True:
            kvw_informer.log_gpu_memory(f"DEBUG: Initializing pipeline with device: {device}", self.log_view)

        self.pipeline = KPipeline(lang_code="a", repo_id='hexgrad/Kokoro-82M', device=device)
        self.device = device

        # Verify model is actually on GPU
        if self.log_view is True and self.pipeline.model:
            print(f"DEBUG: Pipeline model device: {self.pipeline.model.device}")
            print(f"DEBUG: Model parameters device: {next(self.pipeline.model.parameters()).device}")

        # Preprocess BOTH texts at initialization
        self.target_text = target_text
        self.other_text = other_text
        self.target_segments = self._preprocess_text(target_text)
        self.other_segments = self._preprocess_text(other_text)

        if self.log_view:
            print(f"DEBUG: Preprocessed target text: {len(self.target_segments)} segments")
            print(f"DEBUG: Preprocessed other text: {len(self.other_segments)} segments")

    def _preprocess_text(self, text: str):
        """Private helper to preprocess text once"""
        if self.log_view:
            print(f"DEBUG: Preprocessing text: {text[:50]}...")

        # Same logic as pipeline.__call__ but done once
        split_pattern = r'\n+'
        text_segments = re.split(split_pattern, text.strip()) if split_pattern else [text]

        segments = []
        for graphemes_index, graphemes in enumerate(text_segments):
            if not graphemes.strip():
                continue

            # Do expensive g2p and tokenization once
            if self.pipeline.lang_code in 'ab':  # English processing
                _, tokens = self.pipeline.g2p(graphemes)
                for gs, ps, tks in self.pipeline.en_tokenize(tokens):
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        ps = ps[:510]
                    segments.append((gs, ps, tks, graphemes_index))

        return segments

    def generate_audio(self, text: str, voice: torch.Tensor, speed: float = 1.0) -> torch.Tensor:
        """Returns GPU tensor, optimized for both target and other text"""

        # Check which preprocessed segments to use
        if text == self.target_text:
            segments = self.target_segments
        elif text == self.other_text:
            segments = self.other_segments
        else:
            # Fallback for unexpected different text (shouldn't happen in practice)
            segments = self._preprocess_text(text)

        # use preprocessed segments with direct KPipeline.infer()
        audio_chunks = []
        for gs, ps, tks, text_index in segments:
            output = KPipeline.infer(self.pipeline.model, ps, voice, speed)
            if output is not None and output.audio is not None:
                audio_chunks.append(output.audio.to(self.device))

        # Concatenate and return GPU tensor
        if audio_chunks:
            return torch.cat(audio_chunks, dim=0)
        else:
            return torch.tensor([], device=self.device)
