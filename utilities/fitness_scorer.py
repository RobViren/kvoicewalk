import datetime
import warnings
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torchaudio
from numpy._typing import NDArray
from speechbrain.inference.speaker import SpeakerRecognition
from torch import FloatTensor
from torchaudio.prototype.transforms import ChromaSpectrogram

from utilities.kvw_informer import KVW_Informer
from utilities.speech_generator import SpeechGenerator

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# TODO: Review SpeechBrain Feature Extraction & Analysis
# TODO: Revisit Scoring Calculations

class FitnessScorer:
    def __init__(self, target_wav: str, kvw_informer: KVW_Informer, speech_generator: SpeechGenerator,
                 device: str = 'cuda'):
        """
        Initialize FitnessScorer with GPU optimization.

        Args:
            target_wav: Path to target audio file
            device: Device to use ('cuda' or 'cpu')
        """
        self.kvw_informer = kvw_informer
        self.speech_generator = speech_generator
        self.log_view = self.kvw_informer.settings['fitness_logs']
        self.process_times = self.kvw_informer.settings['tps_reports']
        self.feature_times = self.kvw_informer.settings['feature_times']
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.target_wav = target_wav

        # Constants
        self.sr = 24000
        self.n_fft = 2048
        self.n_mels = 128
        self.hop_length = 512
        self.fmin = 200
        self.n_bands = 6
        # Initialize Audio Analysis Classes and Objects on GPU
        self.verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

        # Preload Frequency bins
        self.freqs = torch.fft.fftfreq(self.n_fft, 1 / self.sr)[:self.n_fft // 2 + 1].to(device)

        # Preload Mel spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels).to(device)
        # Preload MFCCs
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr, n_mfcc=13,
            melkwargs={'n_fft': self.n_fft, 'hop_length': self.hop_length, 'n_mels': self.n_mels}).to(device)

        # Preload Chroma transform
        self.chroma_transform = ChromaSpectrogram(sample_rate=self.sr, n_fft=self.n_fft, hop_length=self.hop_length).to(
            device)

        # Preload Tonnetz transformation matrix (6x12) - standard musicology matrix
        self.tonnetz_matrix = torch.tensor([
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Circle of fifths
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # Circle of fifths offset
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # Minor thirds
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # Minor thirds offset
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],  # Minor thirds offset2
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]  # Major thirds
        ], dtype=torch.float32, device=device)

        self.bands = torch.logspace(
            torch.log10(torch.tensor(self.fmin)),
            torch.log10(torch.tensor(self.sr / 2)),
            self.n_bands + 1).to(device)

        # Precompute band masks
        self.band_masks = []
        for i in range(self.n_bands):
            band_mask = (self.freqs >= self.bands[i]) & (self.freqs < self.bands[i + 1])
            self.band_masks.append(band_mask)

        # Preload contrast values
        self.contrast_values = torch.zeros(self.n_bands, device=device)

        # Preload target audio tensor to GPU (do this once!)
        self._target_tensor = self.verification.load_audio(target_wav).to(self.device)
        self.batch_target = self._target_tensor.unsqueeze(0)
        self.emb1 = self.verification.encode_batch(self.batch_target, None, normalize=False)
        if self.log_view is True: self.kvw_informer.log_gpu_memory(
            f"Target audio loaded to {self._target_tensor.device}", self.log_view)

        # Pre-compute target features for feature penalty calculation
        target_audio_numpy = self._target_tensor.cpu().numpy()
        self.target_features = self.extract_features(target_audio_numpy)
        if self.log_view is True: self.kvw_informer.log_gpu_memory(
            f"Extracted {len(self.target_features)} target features", self.log_view)

    def hybrid_similarity(self, best_results: dict[str, Any], audio_array: NDArray[np.float32] | torch.Tensor,
                          audio_embed1: torch.Tensor, other_text: str,
                          voice_tensor: NDArray[np.float32] | torch.Tensor,
                          target_similarity: float, results: dict[str, Any]) -> tuple[dict[
        str, Any], timedelta, timedelta, timedelta] | tuple[dict[str, Any], timedelta, float, float]:
        """
        Calculate hybrid similarity score combining target similarity, self similarity, and feature similarity.
        GPU-compatible version that accepts both numpy arrays and tensors.

        Args:
            :param best_results: best voice scores
            :param audio_array: First audio signal (numpy array or torch tensor)
            :param audio_embed1: First audio embedding (passed from target_similarity
            :param other_text: Comparison text for Audio 2 Self Sim check
            :param voice_tensor: voice tensor to be used in Audio 2 Self Sim check
            :param target_similarity: Pre-calculated target similarity score
            :param results: results dict to return after scoring

        Returns:
            Dictionary containing all similarity scores and final score

        """
        audio_tensor1 = audio_array
        # Extract features using GPU-optimized method
        # Target feature extraction
        if self.log_view is True: self.kvw_informer.log_gpu_memory("Evaluating Feature Sim", self.log_view)
        feature_start = datetime.datetime.now()
        features = self.extract_features(audio_tensor1)

        # Calculate target feature penalty
        if self.log_view is True: self.kvw_informer.log_gpu_memory("Evaluating Target Feature Penalty", self.log_view)
        # target_penalty_start = datetime.datetime.now()
        target_features_penalty = self.target_feature_penalty(features)
        # target_penalty_time = datetime.datetime.now() - target_penalty_start

        # Normalize and make higher = better
        feature_similarity = (100.0 - target_features_penalty) / 100.0
        if feature_similarity < 0.0:
            feature_similarity = 0.01
        feature_time = datetime.datetime.now() - feature_start

        # Added a check for feature similarity within a certain bounds, to avoid audio2 gen if possible
        if feature_similarity >= best_results["feature_similarity"] - 0.1:
            audio2_start = datetime.datetime.now()
            audio2_array = self.speech_generator.generate_audio(other_text, voice_tensor)
            audio2_time = datetime.datetime.now() - audio2_start

            # Calculate self similarity with tensors
            if self.log_view is True: self.kvw_informer.log_gpu_memory("Evaluating Self Sim", self.log_view)
            self_sim_start = datetime.datetime.now()
            self_similarity = self.self_similarity(audio_embed1, audio2_array)
            self_sim_time = datetime.datetime.now() - self_sim_start

            # Prepare values for scoring
            values = np.array([target_similarity, self_similarity, feature_similarity])

            # Weights for potential future use (currently using unweighted harmonic mean)
            # weights = np.array([0.48, 0.5, 0.02])

            # Harmonic mean calculation (unweighted as per current implementation)
            # Harmonic mean heavily penalizes low scores, encouraging balanced improvement
            score = len(values) / np.sum(1.0 / values)
            results.update({
                "score": float(score),
                "target_similarity": float(target_similarity),
                "self_similarity": float(self_similarity),
                "feature_similarity": float(feature_similarity),
                # "weights": weights.tolist()  # Include weights for potential future use
            })
            return results, feature_time, audio2_time, self_sim_time
        else:
            results.update({
                "score": 0.0,
                "target_similarity": float(target_similarity),
                "self_similarity": 0.0,
                "feature_similarity": float(feature_similarity),
                # "weights": weights.tolist()  # Include weights for potential future use
            })
            audio2_time = 0.0
            self_sim_time = 0.0

            return results, feature_time, audio2_time, self_sim_time

    def target_similarity(self, audio_tensor: torch.Tensor | NDArray[np.float32]) -> tuple[float, FloatTensor, Any]:
        """
        Calculate similarity between generated audio and target audio.voice = voice.to(device)
        GPU-optimized version using direct tensor operations.

        Args:
            audio_tensor: Generated audio (tensor or numpy array)

        Returns:
            Similarity score (float)
        """
        # Ensure input is a tensor on the correct device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if isinstance(audio_tensor, np.ndarray):
            audio_float_tensor = torch.from_numpy(audio_tensor.astype(np.float32)).to(device)
        else:
            audio_float_tensor = audio_tensor.to(device).float()

        # Ensure mono audio
        if len(audio_float_tensor.shape) > 1:
            audio_float_tensor = torch.mean(audio_tensor, dim=-1)

        # Use preloaded target tensor (set in __init__)
        if not hasattr(self, 'emb1'):
            self._target_tensor = self.verification.load_audio(self.target_wav).to(device)
            self.batch_target = self._target_tensor
            self.emb1 = self.verification.encode_batch(self.batch_target, None, normalize=False)

        # Create batch for SpeechBrain
        batch_audio = audio_float_tensor.unsqueeze(0)
        emb2 = self.verification.encode_batch(batch_audio, None, normalize=False)
        score = self.verification.similarity(self.emb1, emb2)

        return float(score[0]), audio_float_tensor, emb2

    def target_feature_penalty(self, features: dict[str, Any]) -> float:
        """
        Calculate penalty for differences in audio features compared to target features.
        Optimized version with improved stability and error handling.

        Args:
            features: Dictionary of extracted audio features

        Returns:
            Penalty score (lower is better, 0 = perfect match)
        """
        if not hasattr(self, 'target_features') or not self.target_features:
            # If no target features are available, return neutral penalty
            return 50.0

        penalty = 0.0
        feature_count = 0

        for key, value in features.items():
            if key not in self.target_features:
                # Skip features that don't exist in target (for forward compatibility)
                continue

            target_val = self.target_features[key]

            # Handle different cases for robust calculation
            try:
                # Convert to float if needed (handles numpy types)
                current_val = float(value)
                target_val = float(target_val)

                # Handle near-zero or zero targets
                if abs(target_val) < 1e-8:
                    # For near-zero targets, use absolute difference
                    diff = abs(current_val)
                    # Scale by a reasonable factor to keep penalties in reasonable range
                    diff = min(diff * 100, 5.0)  # Cap at 500% penalty equivalent
                else:
                    # For non-zero targets, use relative difference
                    diff = abs((current_val - target_val) / target_val)
                    # Cap maximum penalty per feature to prevent extreme outliers
                    diff = min(diff, 5.0)  # Max 500% difference penalty

                penalty += diff
                feature_count += 1

            except (ValueError, TypeError, ZeroDivisionError):
                # Skip problematic features rather than crashing
                continue

        if feature_count == 0:
            # No valid features to compare
            return 50.0

        # Average penalty across features, convert to percentage scale
        average_penalty = penalty / feature_count
        return float(average_penalty * 100.0)

    def self_similarity(self, audio_tensor1: torch.Tensor | NDArray[np.float32],
                        audio_tensor2: torch.Tensor | NDArray[np.float32]) -> float:
        """
        Calculate self-similarity between two audio samples from the same voice.
        GPU-optimized version using direct tensor operations.

        Args:
            audio_tensor1: First audio sample (tensor or numpy array)
            audio_tensor2: Second audio sample (tensor or numpy array)

        Returns:
            Self-similarity score (float)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if isinstance(audio_tensor2, np.ndarray):
            audio_tensor2 = torch.from_numpy(audio_tensor2.astype(np.float32)).to(device)
        else:
            audio_tensor2 = audio_tensor2.to(device).float()

        if len(audio_tensor2.shape) > 1:
            audio_tensor2 = torch.mean(audio_tensor2, dim=-1)

        # Create batches
        batch2 = audio_tensor2.unsqueeze(0)

        # Verify on GPU
        emb1 = audio_tensor1
        emb2 = self.verification.encode_batch(batch2, None, normalize=False)
        score = self.verification.similarity(emb1, emb2)

        return float(score[0])

    def extract_features(self, audio_array: NDArray[np.float32] | NDArray[np.float64] | torch.Tensor) -> dict[str, Any]:
        """
        Extract a comprehensive set of audio features for fingerprinting speech segments.
        GPU-optimized version using torchaudio where possible.

        Args:
            audio_array: Audio signal as numpy array or torch tensor

        Returns:
            Dictionary containing extracted features
        """
        start = datetime.datetime.now()
        # Convert input to tensor and ensure it's on GPU
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.from_numpy(audio_array.astype(np.float32)).to(self.device)
        else:
            audio_tensor = audio_array.to(self.device).float()

        # Ensure mono
        if len(audio_tensor.shape) > 1:
            if audio_tensor.shape[-1] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=-1)
            else:
                audio_tensor = audio_tensor.squeeze()

        # Initialize features dictionary
        features = {}
        if self.feature_times is True: print(
            f"Feature Prep Time: {(datetime.datetime.now() - start).total_seconds():.3f}")
        # ===== GPU-ACCELERATED FEATURES =====
        with torch.no_grad():
            start = datetime.datetime.now()
            # Basic features - GPU
            features["rms_energy"] = float(torch.sqrt(torch.mean(audio_tensor ** 2)).item())

            # Zero crossing rate - GPU implementation
            zero_crossings = torch.diff(torch.sign(audio_tensor), dim=0)
            features["zero_crossing_rate"] = float(torch.mean(torch.abs(zero_crossings)).item() / 2.0)

            # STFT for spectral features - GPU
            stft = torch.stft(audio_tensor, n_fft=self.n_fft, hop_length=self.hop_length,
                              win_length=self.n_fft, return_complex=True, center=True)
            magnitude = torch.abs(stft)
            power = magnitude ** 2

            # Spectral centroid - GPU
            weighted_freqs = self.freqs.unsqueeze(1) * magnitude
            spectral_centroids = torch.sum(weighted_freqs, dim=0) / (torch.sum(magnitude, dim=0) + 1e-8)
            features["spectral_centroid_mean"] = float(torch.mean(spectral_centroids).item())
            features["spectral_centroid_std"] = float(torch.std(spectral_centroids).item())

            # Spectral bandwidth - GPU
            freq_diff = (self.freqs.unsqueeze(1) - spectral_centroids.unsqueeze(0)) ** 2
            spectral_bandwidth = torch.sqrt(
                torch.sum(freq_diff * magnitude, dim=0) / (torch.sum(magnitude, dim=0) + 1e-8))
            features["spectral_bandwidth_mean"] = float(torch.mean(spectral_bandwidth).item())
            features["spectral_bandwidth_std"] = float(torch.std(spectral_bandwidth).item())

            # Spectral rolloff - GPU (85% of spectral energy)
            cumsum_power = torch.cumsum(power, dim=0)
            total_power = torch.sum(power, dim=0)
            rolloff_thresh = 0.85 * total_power
            rolloff_indices = torch.argmax((cumsum_power >= rolloff_thresh.unsqueeze(0)).float(), dim=0)
            rolloff_freqs = self.freqs[rolloff_indices]
            features["spectral_rolloff_mean"] = float(torch.mean(rolloff_freqs).item())
            features["spectral_rolloff_std"] = float(torch.std(rolloff_freqs).item())

            mel_spec = self.mel_transform(audio_tensor.unsqueeze(0)).squeeze(0)
            features["mel_spec_mean"] = float(torch.mean(mel_spec).item())
            features["mel_spec_std"] = float(torch.std(mel_spec).item())

            if self.feature_times is True: print(
                f"Spectral Analysis Time: {(datetime.datetime.now() - start).total_seconds():.3f}")
            start = datetime.datetime.now()

            mfccs = self.mfcc_transform(audio_tensor.unsqueeze(0)).squeeze(0)

            # Store each MFCC coefficient mean and std
            mfcc_means = torch.mean(mfccs, dim=1)
            mfcc_stds = torch.std(mfccs, dim=1)

            # MFCC delta features (first derivative) - GPU
            mfcc_delta = torch.diff(mfccs, dim=1, prepend=mfccs[:, :1])
            mfcc_delta_means = torch.mean(mfcc_delta, dim=1)
            mfcc_delta_stds = torch.std(mfcc_delta, dim=1)

            # Then just index into the results
            for i in range(13):
                features[f"mfcc{i + 1}_mean"] = float(mfcc_means[i].item())
                features[f"mfcc{i + 1}_std"] = float(mfcc_stds[i].item())

            for i in range(13):
                features[f"mfcc{i + 1}delta_mean"] = float(mfcc_delta_means[i].item())
                features[f"mfcc{i + 1}delta_std"] = float(mfcc_delta_stds[i].item())

            if self.feature_times is True: print(
                f"MFCC Analysis Time: {(datetime.datetime.now() - start).total_seconds():.3f}")
            start = datetime.datetime.now()

            # Spectral flatness - GPU
            geometric_mean = torch.exp(torch.mean(torch.log(magnitude + 1e-8), dim=0))
            arithmetic_mean = torch.mean(magnitude, dim=0)
            flatness = geometric_mean / (arithmetic_mean + 1e-8)
            features["spectral_flatness_mean"] = float(torch.mean(flatness).item())
            features["spectral_flatness_std"] = float(torch.std(flatness).item())

            # Spectral contrast - GPU implementation
            for i, band_mask in enumerate(self.band_masks):  # Add the missing loop
                if torch.sum(band_mask) == 0:
                    self.contrast_values[i] = 0.0
                    continue

                # Get magnitude in this band
                band_mag = magnitude[band_mask, :]  # Use the current band_mask from loop

                # Calculate contrast (peak vs valley)
                if band_mag.numel() > 0:
                    # Flatten for easier processing
                    band_flat = band_mag.flatten()

                    # Peak: mean of top 20% values
                    top_k = max(1, int(0.2 * band_flat.shape[0]))
                    peaks, _ = torch.topk(band_flat, top_k, dim=0)  # Fix: band_flat not band*flat
                    peak_val = torch.mean(peaks)

                    # Valley: mean of bottom 20% values
                    bottom_k = max(1, int(0.2 * band_flat.shape[0]))
                    valleys, _ = torch.topk(band_flat, bottom_k, dim=0, largest=False)  # Fix: band_flat
                    valley_val = torch.mean(valleys)

                    # Contrast ratio
                    self.contrast_values[i] = peak_val / (valley_val + 1e-8)
                else:
                    self.contrast_values[i] = 0.0

            # After the loop, compute final features
            features["spectral_contrast_mean"] = float(torch.mean(self.contrast_values).item())
            features["spectral_contrast_std"] = float(torch.std(self.contrast_values).item())

            if self.feature_times is True: print(
                f"Spectral Flatness and Contrast Analysis Time: {(datetime.datetime.now() - start).total_seconds():.3f}")
            start = datetime.datetime.now()

            # Energy features - GPU
            frame_length = self.hop_length
            frames = audio_tensor.unfold(0, frame_length, self.hop_length)
            energy = torch.sum(torch.abs(frames), dim=1)
            features["energy_mean"] = float(torch.mean(energy).item())
            features["energy_std"] = float(torch.std(energy).item())

            # Harmonics-to-noise ratio - GPU
            S_squared = magnitude ** 2
            S_mean = torch.mean(S_squared, dim=1)
            S_std = torch.std(S_squared, dim=1)
            S_ratio = S_mean / (S_std + 1e-8)
            features["harmonic_ratio"] = float(torch.mean(S_ratio).item())

            # Statistical features from raw waveform - GPU
            features["audio_mean"] = float(torch.mean(audio_tensor).item())
            features["audio_std"] = float(torch.std(audio_tensor).item())

            if self.feature_times is True: print(
                f"Energy Analysis Time: {(datetime.datetime.now() - start).total_seconds():.3f}")
            start = datetime.datetime.now()

            # Chroma features - GPU using torchaudio.prototype
            chroma = self.chroma_transform(audio_tensor.unsqueeze(0)).squeeze(0)

            # Overall chroma statistics
            chroma_mean = torch.mean(chroma)
            chroma_std = torch.std(chroma)
            features["chroma_mean"] = float(chroma_mean.item())
            features["chroma_std"] = float(chroma_std.item())

            # Per-chroma-bin statistics (same pattern as MFCC)
            chroma_means = torch.mean(chroma, dim=1)
            chroma_stds = torch.std(chroma, dim=1)

            # Store individual chroma features
            for i in range(chroma.shape[0]):
                features[f"chroma_{i + 1}_mean"] = float(chroma_means[i].item())
                features[f"chroma_{i + 1}_std"] = float(chroma_stds[i].item())

            # Normalize chroma (avoid division by zero)
            chroma_norm = chroma / (torch.sum(chroma, dim=0, keepdim=True) + 1e-8)

            if self.feature_times is True: print(
                f"Chroma Analysis Time: {(datetime.datetime.now() - start).total_seconds():.3f}")
            start = datetime.datetime.now()

            # Apply Tonnetz transformation
            tonnetz = torch.matmul(self.tonnetz_matrix, chroma_norm)  # [6, time]

            # Extract features
            features["tonnetz_mean"] = float(torch.mean(tonnetz).item())
            features["tonnetz_std"] = float(torch.std(tonnetz).item())

            if self.feature_times is True: print(
                f"Tonnetz Analysis Time: {(datetime.datetime.now() - start).total_seconds():.3f}")
            start = datetime.datetime.now()

            # Statistical features - GPU implementation
            audio_mean = torch.mean(audio_tensor)
            audio_std = torch.std(audio_tensor)
            normalized = (audio_tensor - audio_mean) / (audio_std + 1e-8)

            # Skewness (third moment)
            skewness = torch.mean(normalized ** 3)
            features["audio_skew"] = float(skewness.item())

            # Kurtosis (fourth moment minus 3)
            kurtosis = torch.mean(normalized ** 4) - 3.0
            features["audio_kurtosis"] = float(kurtosis.item())

            if self.feature_times is True: print(
                f"Statistics Analysis Time: {(datetime.datetime.now() - start).total_seconds():.3f}")
            start = datetime.datetime.now()

            # Extract Kaldi pitch for better quality filtering
            try:
                # compute_kaldi_pitch expects [channels, samples] format
                if audio_tensor.dim() == 1:
                    audio_for_pitch = audio_tensor.unsqueeze(0)
                else:
                    audio_for_pitch = audio_tensor

                # Kaldi pitch computation
                pitch = torchaudio.functional.detect_pitch_frequency(
                    audio_for_pitch,
                    sample_rate=self.sr
                )

                # pitch shape: [channels, frames] - pitch values in Hz
                pitch_hz = pitch[0, :]  # Get pitch values from first channel

                # Filter valid pitches (detect_pitch_frequency returns 0 for unvoiced)
                valid_mask = pitch_hz > 0

                if torch.sum(valid_mask) > 0:
                    valid_pitches = pitch_hz[valid_mask]
                    features["pitch_mean"] = float(torch.mean(valid_pitches).item())
                    features["pitch_std"] = float(torch.std(valid_pitches).item())
                    # No confidence score available with detect_pitch_frequency
                    features["pitch_confidence_mean"] = 1.0  # Placeholder since all detected pitches are "confident"
                else:
                    features["pitch_mean"] = 0.0
                    features["pitch_std"] = 0.0
                    features["pitch_confidence_mean"] = 0.0

            except Exception as e:
                print(f"Pitch detection failed: {e}")
                features["pitch_mean"] = 0.0
                features["pitch_std"] = 0.0
                features["pitch_confidence_mean"] = 0.0

            if self.feature_times is True: print(
                f"Pitch Analysis Time: {(datetime.datetime.now() - start).total_seconds():.3f}")

        # TODO: add tempo, rhythm support... BeatNet won't work for this!
        # # BeatNet expects numpy format
        # audio_numpy = audio_tensor.cpu().numpy()
        #
        # # Process audio - BeatNet returns beat times directly
        # output = self.beat_tracker.process(audio_numpy, self.sr)
        #
        # if len(output) > 0:
        #     # BeatNet output format: [beat_times] or [(beat_times, tempo)]
        #     if isinstance(output[0], tuple):
        #         beat_times, tempo = output[0]
        #     else:
        #         beat_times = output
        #         # Estimate tempo from beat intervals
        #         if len(beat_times) > 1:
        #             beat_intervals = np.diff(beat_times)
        #             tempo = 60.0 / np.mean(beat_intervals)
        #         else:
        #             tempo = 120.0  # Default
        #
        #     features["tempo"] = float(tempo)
        #
        #     if len(beat_times) > 1:
        #         beat_diffs = np.diff(beat_times)
        #         features["beat_mean"] = float(np.mean(beat_diffs))
        #         features["beat_std"] = float(np.std(beat_diffs))
        #     else:
        #         features["beat_mean"] = 0.0
        #         features["beat_std"] = 0.0
        # else:
        #     features["tempo"] = 120.0  # Default tempo
        #     features["beat_mean"] = 0.0
        #     features["beat_std"] = 0.0

        return features
