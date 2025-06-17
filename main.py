import argparse
import os
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from utilities.audio_processor import Transcriber, convert_to_wav_mono_24k
from utilities.kvoicewalk import KVoiceWalk
from utilities.kvw_informer import KVW_Informer
from utilities.pytorch_sanitizer import load_multiple_voices
from utilities.speech_generator import SpeechGenerator


def main():
    # import config settings
    kvw_informer = KVW_Informer()
    kvw_settings = kvw_informer.settings
    log_view = kvw_settings["preprocessing_logs"]
    use_cached = kvw_settings["use_cached"]
    cap_memory = kvw_settings["cap_memory"]
    cap_memory_frac = kvw_settings["cap_memory_frac"]
    # After initial download, recommend use cached copies of models == faster load times
    if use_cached:
        os.environ['HF_HUB_OFFLINE'] = '1'  # Force offline mode
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    # True: Limits excess memory overhead reservation, benchmarked at ~0.70GB throughout operation, no spikes
    # Cap_memory_frac = 0.2, can be set 0-1, but recommend no lower than 0.15
    if cap_memory: torch.cuda.set_per_process_memory_fraction(cap_memory_frac)

    parser = argparse.ArgumentParser(description="A random walk Kokoro voice cloner.")

    # Common required arguments
    parser.add_argument("--target_text", type=str, help="The words contained in the target audio file. Should be around 100-200 tokens (two sentences). Alternatively, can point to a txt file of the transcription.")

    # Optional arguments
    parser.add_argument("--other_text", type=str,
                      help="A segment of text used to compare self similarity. Should be around 100-200 tokens.",
                      default="If you mix vinegar, baking soda, and a bit of dish soap in a tall cylinder, the resulting eruption is both a visual and tactile delight, often used in classrooms to simulate volcanic activity on a miniature scale.")
    parser.add_argument("--voice_folder", type=str,
                      help="Path to the voices you want to use as part of the random walk.",
                      default="./voices")
    parser.add_argument("--transcribe_start",
                        help='Input: filepath to wav file\nOutput: Transcription .txt in ./texts\nTranscribes a target wav or wav folder and replaces --target_text',
                      action='store_true')
    parser.add_argument("--interpolate_start",
                      help="Goes through an interpolation search step before random walking",
                      action='store_true')
    parser.add_argument("--population_limit", type=int,
                      help="Limits the amount of voices used as part of the random walk",
                      default=10)
    parser.add_argument("--step_limit", type=int,
                      help="Limits the amount of steps in the random walk",
                      default=10000)
    parser.add_argument("--output_name", type=str,
                      help="Filename for the generated output audio",
                        default="my_new_voice")

    # Arguments for random walk mode
    group_walk = parser.add_argument_group('Random Walk Mode')
    group_walk.add_argument("--target_audio", type=str,
                          help="Path to the target audio file. Must be 24000 Hz mono wav file.")
    group_walk.add_argument("--starting_voice", type=str,
                          help="Path to the starting voice tensor")

    # Arguments for test mode
    group_test = parser.add_argument_group('Test Mode')
    group_test.add_argument("--test_voice", type=str,
                          help="Path to the voice tensor you want to test")

    # Arguments for util mode
    group_util = parser.add_argument_group('Utility Mode')
    group_util.add_argument("--export_bin",
                      help='Exports target voices in the --voice_folder directory',
                      action='store_true')
    group_util.add_argument("--transcribe_many",
                            help='Input: filepath to wav file or folder\nOutput: Individualized transcriptions in ./texts folder\nTranscribes a target wav or wav folder. Replaces --target_text', )
    args = parser.parse_args()


    # Export Utility
    if args.export_bin:
        if not args.voice_folder:
            parser.error("--voice_folder is required to export a voices bin file")

        # Collect all .pt file paths
        file_paths = [os.path.join(args.voice_folder, f) for f in os.listdir(args.voice_folder) if f.endswith('.pt')]
        voices = load_multiple_voices(file_paths, auto_allow_unsafe=False) # Set True if you prefer to bypass Allow/Repair/Reject voice file menu

        with open("voices.bin", "wb") as f:
            np.savez(f,**voices)

        return

    # Handle target_audio input - convert to mono wav 24K automatically
    if args.target_audio:
        if log_view is True: kvw_informer.log_gpu_memory("Preprocessing target audio file", log_view)
        try:
            target_audio_path = Path(args.target_audio)
            if target_audio_path.is_file():
                args.target_audio = convert_to_wav_mono_24k(target_audio_path)
            else:
                print(f"File not found: {target_audio_path}")
        except Exception as e:
            print(f"Error reading target_audio file: {e}")

    # Transcribe (Start Mode)
    if args.transcribe_start:
        if log_view is True: kvw_informer.log_gpu_memory("Transcribing target audio file", log_view)
        try:
            target_path = Path(args.target_audio)

            if target_path.is_file():
                if target_path.suffix.lower() == '.wav':
                    print(f"Sending {target_path.name} for transcription")
                    transcriber = Transcriber()
                    args.target_text = transcriber.transcribe(audio_path=target_path)
                else:
                    try:
                        args.target_audio = convert_to_wav_mono_24k(target_path)
                        transcriber = Transcriber()
                        args.target_text = transcriber.transcribe(audio_path=target_path)
                    except:
                        parser.error(f"File format error: {target_path.name} is not a .wav file.")
            elif target_path.is_dir():
                parser.error("--transcribe_start requires a .wav file only. Use --transcribe_many for directories.")
            else:
                parser.error(f"File not found: {target_path}. Please check your file path.")

        except Exception as e:
            print(f"Error during transcription: {e}")
            return

    # Transcribe (Utility Mode)
    if args.transcribe_many:
        try:
            input_path = Path(args.transcribe_many)

            if input_path.is_file():
                if input_path.suffix.lower() == '.wav':
                    print(f"Sending {input_path.name} for transcription")
                    transcriber = Transcriber()
                    transcriber.transcribe(audio_path=input_path)
                else:
                    print(f"File Format Error: {input_path.name} is not an audio file!")
                return

            elif input_path.is_dir():
                wav_files = list(input_path.glob('*.wav'))
                if not wav_files:
                    # TODO: Handle batch processing of non-wav audios
                    print(f"No .wav files found in {input_path}")
                    return

                transcriber = Transcriber()
                for audio_file in wav_files:
                    print(f"Sending {audio_file.name} for transcription")
                    transcriber.transcribe(audio_path=audio_file)
                return

            else:
                print(f"Input Format Error: {input_path.name} must be a .wav file or a directory!")
                return

        except Exception as e:
            print(f"Error during transcription: {e}")
            return

    # Handle text input - read from file if it's a .txt file path
    if args.target_text and args.target_text.endswith('.txt'):
        try:
            text_path = Path(args.target_text)
            if text_path.is_file():
                args.target_text = text_path.read_text(encoding='utf-8')
            else:
                print(f"File not found: {text_path}")
        except Exception as e:
            print(f"Error reading text file: {e}")

    # Validate arguments based on mode
    if args.test_voice:
        # Test mode
        if not args.target_text:
            parser.error("--target_text is required when using --test_voice")

        speech_generator = SpeechGenerator(kvw_informer=kvw_informer, target_text=args.target_text,
                                           other_text=args.other_text)
        audio = speech_generator.generate_audio(args.target_text, args.test_voice)
        sf.write(args.output_name, audio, 24000)
    else:
        # Random walk mode
        if not args.target_audio:
            parser.error("--target_audio is required for random walk mode")
        if not args.target_text:
            parser.error("--target_text is required for random walk mode")

        if log_view is True: kvw_informer.log_gpu_memory("Initializing KVoicewalk", log_view)
        ktb = KVoiceWalk(args.target_audio,
                         args.target_text,
                         args.other_text,
                         args.voice_folder,
                         args.interpolate_start,
                         args.population_limit,
                         args.starting_voice,
                         args.output_name, kvw_informer)
        try:
            ktb.random_walk(args.step_limit)
        except Exception as e:
            print("FULL TRACEBACK:")
            traceback.print_exc()
            print(f"\nERROR: {e}")
            print(f"ERROR TYPE: {type(e)}")
if __name__ == "__main__":
    main()
