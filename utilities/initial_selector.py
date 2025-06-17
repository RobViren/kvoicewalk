import os
from typing import Any

import numpy as np
import torch

from utilities.kvw_informer import KVW_Informer
from utilities.path_router import INTERPOLATED_DIR
from utilities.pytorch_sanitizer import load_voice_safely


class InitialSelector:
    def __init__(self, fitness_scorer: Any, speech_generator: Any, target_path: str, target_text: str, other_text: str,
                 kvw_informer: KVW_Informer, voice_folder: str = "./voices") -> None:
        self.kvw_informer = kvw_informer
        self.log_view = self.kvw_informer.settings['voice_loading_logs']
        self.process_times = self.kvw_informer.settings['tps_reports']
        self.fitness_scorer = fitness_scorer
        self.speech_generator = speech_generator
        voices = []
        for filename in os.listdir(voice_folder):
            if filename.endswith('.pt'):
                file_path = os.path.join(voice_folder, filename)
                voice = load_voice_safely(file_path)
                voices.append({
                    'name': filename,
                    'voice': voice
                })

        self.voices = voices
        self.target_text = target_text
        self.other_text = other_text

    def top_performer_start(self,population_limit: int) -> list[torch.Tensor]:
        """Simple top performer search to find best voices to use in random walk"""

        results = {}
        if self.log_view is True: self.kvw_informer.log_gpu_memory("Before top_performer_start call", self.log_view)
        for voice in self.voices:
            best_results = {
                "score": 0.0,
                "target_similarity": 0.0,
                "self_similarity": 0.0,
                "feature_similarity": 0.0,
            }
            audio = self.speech_generator.generate_audio(self.target_text, voice["voice"])
            target_similarity, audio_float_tensor, audio_embed1 = self.fitness_scorer.target_similarity(audio)
            results, _, _, _ = self.fitness_scorer.hybrid_similarity(best_results, audio_float_tensor, audio_embed1,
                                                                     self.other_text, voice["voice"], target_similarity,
                                                                     results)
            if self.log_view is True: print(
                f'{voice["name"]:<30} Target Sim:{results["target_similarity"]:.3f} Self Sim:{results["self_similarity"]:.3f} Feature Sim:{results["feature_similarity"]:.2f} Score:{results["score"]:.2f}')
            voice["results"] = results

        voices = sorted(self.voices, key=lambda x: x["results"]["score"],reverse=True)
        voices = voices[:population_limit]
        print("Top Performers:")
        for voice in voices:
            print(f'{voice["name"]:<30} Target Sim:{voice["results"]["target_similarity"]:.3f} Self Sim:{voice["results"]["self_similarity"]:.3f} Feature Sim:{voice["results"]["feature_similarity"]:.2f} Score:{voice["results"]["score"]:.2f}')

        tensors = [voice["voice"]for voice in voices]
        if self.log_view is True: self.kvw_informer.log_gpu_memory("After top_performer_start call", self.log_view)
        return tensors

    def interpolate_search(self,population_limit: int) -> list[torch.Tensor]:
        """Finds an initial population of voices more optimal because of interpolated features"""
        results = {}
        for voice in self.voices:
            best_results = {
                "score": 0.0,
                "target_similarity": 0.0,
                "self_similarity": 0.0,
                "feature_similarity": 0.0,
            }
            audio = self.speech_generator.generate_audio(self.target_text, voice["voice"])
            target_similarity, audio_float_tensor, audio_embed1 = self.fitness_scorer.target_similarity(audio)
            results, _, _, _ = self.fitness_scorer.hybrid_similarity(best_results, audio_float_tensor, audio_embed1,
                                                                     self.other_text, voice["voice"], target_similarity,
                                                                     results)
            if self.log_view is True: print(
                f'{voice["name"]:<20} Target Sim:{results["target_similarity"]:.3f}, Self Sim:{results["self_similarity"]:.3f}, Feature Sim:{results["feature_similarity"]:.2f}, Score:{results["score"]:.2f}')
            voice["results"] = results

        voices = sorted(self.voices, key=lambda x: x["results"]["score"],reverse=True)
        voices = voices[:population_limit]
        print("Top Performers:")
        for voice in voices:
            print(f'{voice["name"]:<20} Target Sim:{voice["results"]["target_similarity"]:.3f}, Self Sim:{voice["results"]["self_similarity"]:.3f}, Feature Sim:{voice["results"]["feature_similarity"]:.2f}, Score: {voice["results"]["score"]:.2f}')

        res = {}
        print("Interpolating Best Voices:")
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                for iter in np.arange(-1.5,1.5 + 0.01,0.1):
                    voice = interpolate(voices[i]["voice"], voices[j]["voice"], iter)
                    audio = self.speech_generator.generate_audio(self.target_text, voice)
                    target_similarity, audio_float_tensor, audio_embed1 = self.fitness_scorer.target_similarity(audio)
                    results, _, _, _ = self.fitness_scorer.hybrid_similarity(best_results, audio_float_tensor,
                                                                             audio_embed1,
                                                                             self.other_text, voice, target_similarity,
                                                                             results)

                    print(f'{i:<3} {j:<3} {iter:<4.2f} {voices[i]["name"] or "N/A":<10} {voices[j]["name"] or "N/A":<10} Target Sim:{results.get("target_similarity", 0):.3f}, Self Sim:{results.get("self_similarity", 0):.3f}, Feature Sim:{results.get("feature_similarity", 0):.2f}, Score:{results.get("score", 0):.2f}')

                    if i not in res and iter <= 0.0:
                        res[i] = (voice,iter,voices[i]["name"],voices[j]["name"],results)
                    elif i in res and iter <= 0.0:
                        if res[i][4]["score"] < results["score"]:
                            res[i] = (voice,iter,voices[i]["name"],voices[j]["name"],results)

                    if j not in res and iter > 0.0:
                        res[j] = (voice,iter,voices[j]["name"],voices[i]["name"],results)
                    elif j in res and iter > 0.0:
                        if res[j][4]["score"] < results["score"]:
                            res[j] = (voice,iter,voices[j]["name"],voices[i]["name"],results)

        interpolated_voices: list[torch.Tensor] = []
        if not os.path.exists(INTERPOLATED_DIR):
            os.makedirs(INTERPOLATED_DIR)
        for key, value in res.items():
            print(f'{key} {value[1]:.2f} {value[2]} {value[3]} {value[4]["score"]:.3f}')
            torch.save(value[0],f"{INTERPOLATED_DIR}/{value[2]}_{value[3]}_{value[1]:.2f}.pt")
            interpolated_voices.append(value[0])

        return interpolated_voices

def interpolate(voice1, voice2, alpha):
    diff_vector = voice1 - voice2
    midpoint = (voice1 + voice2) / 2.0
    return midpoint + (diff_vector * alpha / 2.0)
