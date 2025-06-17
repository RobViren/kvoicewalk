import torch

from utilities.kvw_informer import KVW_Informer
from utilities.pytorch_sanitizer import load_voice_safely


class VoiceGenerator:
    def __init__(self, kvw_informer: KVW_Informer, voices: list[torch.Tensor], starting_voice: str | None):
        self.voices = voices
        self.kvw_informer = kvw_informer
        self.log_view = self.kvw_informer.settings['voice_gen_logs']
        self.process_times = self.kvw_informer.settings['tps_reports']
        self.stacked = torch.stack(voices,dim=0)
        self.mean = self.stacked.mean(dim=0)
        self.std = self.stacked.std(dim=0)
        self.min = self.stacked.min(dim=0)[0]
        self.max = self.stacked.max(dim=0)[0]

        if starting_voice:
            self.starting_voice = load_voice_safely(starting_voice)
        else:
            self.starting_voice = self.mean

    def generate_voice(self, base_tensor: torch.Tensor | None, diversity: float = 1.0, device: str = "cuda",
                       clip: bool = False):
        """Generate a new voice tensor based on the base_tensor and diversity.

        Args:
            base_tensor (torch.Tensor | None): The base tensor to generate the new voice from.
            diversity (float, optional): The diversity of the new voice. Defaults to 1.0.
            device (str, optional): The device to generate the new voice on. Defaults to "cuda".
            clip (bool, optional): Whether to clip the new voice to the min and max values. Defaults to False.

        Returns:
            torch.Tensor: The new voice tensor.
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if base_tensor is None:
            base_tensor = self.mean.to(device)
        else:
            base_tensor = base_tensor.clone().to(device)

        noise = torch.randn_like(base_tensor, device=device)
        std_tensor = self.std.to(device)
        scaled_noise = noise * std_tensor * diversity
        new_tensor = base_tensor + scaled_noise

        if clip:
            min_tensor = self.min.to(device)
            max_tensor = self.max.to(device)
            new_tensor = torch.clamp(new_tensor, min_tensor, max_tensor)

        return new_tensor.float()

    # TODO: Make more voice generation functions
