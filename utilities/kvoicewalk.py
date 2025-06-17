import datetime
import gc
import os
import random
import traceback
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch import Tensor
from tqdm import tqdm

from utilities.fitness_scorer import FitnessScorer
from utilities.initial_selector import InitialSelector
from utilities.kvw_informer import KVW_Informer
from utilities.path_router import OUT_DIR
from utilities.speech_generator import SpeechGenerator
from utilities.voice_generator import VoiceGenerator


class KVoiceWalk:
    def __init__(self, target_audio: Path, target_text: str, other_text: str, voice_folder: str,
                 interpolate_start: bool, population_limit: int, starting_voice: str, output_name: str,
                 kvw_informer: KVW_Informer) -> None:
        try:
            self.kvw_informer = kvw_informer
            self.log_view = self.kvw_informer.settings['scoring_results_logs']
            self.process_times = self.kvw_informer.settings['tps_reports']
            self.memcache_clear_freq = self.kvw_informer.settings['memcache_clear_iteration_freq']
            self.target_audio = target_audio
            self.target_text = target_text
            self.other_text = other_text
            if self.log_view is True: self.kvw_informer.log_gpu_memory("Initializing Speech Generator", self.log_view)
            self.speech_generator = SpeechGenerator(kvw_informer=self.kvw_informer, target_text=target_text,
                                                    other_text=other_text)
            if self.log_view is True: self.kvw_informer.log_gpu_memory("Scoring target audio", self.log_view)
            self.fitness_scorer = FitnessScorer(str(target_audio), kvw_informer=self.kvw_informer,
                                                speech_generator=self.speech_generator)
            try:
                if self.log_view is True: self.kvw_informer.log_gpu_memory("Selecting voices", self.log_view)
                self.initial_selector = InitialSelector(self.fitness_scorer, self.speech_generator, str(target_audio),
                                                        target_text, other_text, kvw_informer,
                                                        voice_folder=voice_folder)
            except Exception as e:
                raise Exception(f"Error: {e}")

            if self.log_view is True: self.kvw_informer.log_gpu_memory("Selecting starting voices", self.log_view)
            if interpolate_start:
                voices = self.initial_selector.interpolate_search(population_limit)
            else:
                voices = self.initial_selector.top_performer_start(population_limit)
            if self.log_view is True: self.kvw_informer.log_gpu_memory("Initializing Voice Generator", self.log_view)
            self.voice_generator = VoiceGenerator(kvw_informer, voices, starting_voice)
            self.starting_voice = self.voice_generator.starting_voice
            self.clear_losers_from_memory()
            if self.log_view is True: self.kvw_informer.log_gpu_memory("After voice_clear during initialization",
                                                                       self.log_view)
            self.output_name = output_name

        except Exception as e:
            print("FULL TRACEBACK:")
            traceback.print_exc()
            print(f"\nERROR: {e}")
            print(f"ERROR TYPE: {type(e)}")
            print(f"Error initializing KVoicewalk: {e}")
            raise SystemExit

    def random_walk(self,step_limit: int):
        if self.log_view is True: self.kvw_informer.log_gpu_memory("Scoring initial voice", self.log_view)
        # Score Initial Voice
        t = tqdm()
        best_voice = self.starting_voice
        best_results = {
            "score": 0.0,
            "target_similarity": 0.0,
            "self_similarity": 0.0,
            "feature_similarity": 0.0,
        }
        best_results, tps_report = self.score_voice(best_results=best_results, voice_tensor=self.starting_voice,
                                                    min_similarity=-100.0)
        t.write(f'Target Sim:{best_results["target_similarity"]:.3f}, Self Sim:{best_results["self_similarity"]:.3f}, Feature Sim:{best_results["feature_similarity"]:.2f}, Score:{best_results["score"]:.2f}')

        # Create Results Directory
        now = datetime.datetime.now()
        results_dir = Path(OUT_DIR / f'{self.output_name}_{self.target_audio.stem}_{now.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(results_dir, exist_ok=True)

        # Random Walk Loop
        if self.log_view is True: self.kvw_informer.log_gpu_memory("Starting random walk run", self.log_view)
        progress_bar = tqdm(range(step_limit), desc="KVoiceWalk Progress")
        for i in progress_bar:
            # TODO: Expose to CLI
            diversity = random.uniform(0.01,0.15)
            if self.log_view is True: self.kvw_informer.log_gpu_memory("Generating best voice comparison",
                                                                       self.log_view)
            voice = self.voice_generator.generate_voice(best_voice, diversity)

            # Early function return saves audio generation compute
            min_similarity = best_results["target_similarity"] * 0.98
            if self.log_view is True: self.kvw_informer.log_gpu_memory("Scoring Voice results", self.log_view)
            voice_results, tps_report = self.score_voice(best_results, voice, min_similarity)

            # Check GPU memory
            info = self.kvw_informer.log_gpu_memory("GPU Stats", view=self.log_view, console=True)
            progress_bar.set_postfix_str(f"{info}]")
            # Per config every # of steps, clear Kpipeline cache
            if i % self.memcache_clear_freq == 0 and i > 0:
                self.clear_losers_from_memory()

            # Set new winner if score is better
            if voice_results["score"] > best_results["score"]:
                best_results = voice_results
                best_voice = voice.cpu()
                try:
                    t.write(
                        f'Step:{i:<4} Target Sim:{best_results["target_similarity"]:.3f} Self Sim:{best_results["self_similarity"]:.3f} Feature Sim:{best_results["feature_similarity"]:.3f} Score:{best_results["score"]:.2f} Diversity:{diversity:.2f}')
                    if self.process_times is True: t.set_postfix_str(f"{info}\n{tps_report}")
                except Exception:
                    print("")
                # Save results so folks can listen
                best_voice_name = f'{results_dir}/{self.output_name}_{i}_{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{self.target_audio.stem}.pt'
                torch.save(best_voice,
                           best_voice_name)
                sf.write(
                    f'{results_dir}/{self.output_name}_{i}_{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{self.target_audio.stem}.wav',
                    best_results["audio"], 24000)
                # TODO: Add config file for easy restarting runs from last save point

        # Print Final Results for Random Walk
        print(f"\n\nRandom Walk Final Results for {self.output_name}")
        print(f"Duration: {(t.format_dict['elapsed'] / 60):.2f} minutes")
        print(f"Best Voice: {best_voice_name}")
        print(f"Best Score: {best_results['score']:.2f}")
        print(f"Best Similarity: {best_results['target_similarity']:.2f}")
        print(f"Random Walk pt and wav files ---> {results_dir}")

        # clear memory at completion
        gc.collect()
        torch.cuda.empty_cache()
        return

    # TODO: Move function to fitness_scorer
    def score_voice(self, best_results: dict[str, Any], voice_tensor: torch.Tensor, min_similarity: float = 0.0) -> \
    tuple[dict[str, Any], str]:
        start_time = datetime.datetime.now()

        if self.log_view is True: self.kvw_informer.log_gpu_memory("Generating iterated voice audio", self.log_view)
        # Time audio generation
        audio_start = datetime.datetime.now()
        audio = self.speech_generator.generate_audio(self.target_text, voice_tensor)
        audio = Tensor.cpu(audio).numpy()
        audio_time = datetime.datetime.now() - audio_start
        results: dict[str, Any] = {
            'audio': audio
        }
        if self.log_view is True: self.kvw_informer.log_gpu_memory("Evaluating Target Sim", self.log_view)
        # Time target similarity calculation
        target_sim_start = datetime.datetime.now()
        # Pass embedding for reuse in self_sim
        target_sim, audio_float_tensor, audio_embed1 = self.fitness_scorer.target_similarity(audio)
        target_sim_time = datetime.datetime.now() - target_sim_start

        if self.log_view is True: self.kvw_informer.log_gpu_memory("Checking target vs min sim", self.log_view)
        if target_sim > min_similarity:
            # Time hybrid sim scoring
            if self.log_view is True: self.kvw_informer.log_gpu_memory("Evaluating Hybrid Sim", self.log_view)
            results, feature_sim_time, audio2_time, self_sim_time = (
                self.fitness_scorer.hybrid_similarity(best_results, audio_float_tensor, audio_embed1,
                                                      self.other_text, voice_tensor, target_sim, results))
            total_time = datetime.datetime.now() - start_time
            if audio2_time != 0.0 and self_sim_time != 0.0:
                if self.log_view is True: self.kvw_informer.log_gpu_memory(
                    "Returning success results (target sim > min sim)", self.log_view)
                if self.process_times is True: tps_report = str(
                    f"Process Times: Audio1 gen: {audio_time.total_seconds():3f}s, Audio2 gen: {audio2_time.total_seconds():3f}s, Target Sim: {target_sim_time.total_seconds():3f}s,  Self Sim: {self_sim_time.total_seconds():3f}s, Feat Sim: {feature_sim_time.total_seconds():3f}s, Total: {total_time.total_seconds():3f}s")
                return results, tps_report
            else:
                if self.log_view is True: self.kvw_informer.log_gpu_memory("target sim < min sim)", self.log_view)
        else:
            if self.log_view is True: self.kvw_informer.log_gpu_memory("target sim < min sim)", self.log_view)
        results["score"] = 0.0
        results["target_similarity"] = target_sim
        tps_report = ''
        return results, tps_report

    def clear_losers_from_memory(self):
        # Get reference to the voices cache
        voices_cache = self.speech_generator.pipeline.voices

        # Create list of voices to keep (winners)
        voices_to_keep = {self.starting_voice}  # Always keep starting voice

        # Add current best voice to exempt list
        if hasattr(self, 'best_voice') and self.best_voice:
            voices_to_keep.add(self.best_voice)

        # Clean up all other voices
        voices_to_delete = []
        for voice_name in voices_cache:
            if voice_name not in voices_to_keep:
                voices_to_delete.append(voice_name)

        # Delete the non-winners
        for voice_name in voices_to_delete:
            del voices_cache[voice_name]
