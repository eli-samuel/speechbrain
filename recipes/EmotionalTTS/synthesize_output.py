import torchaudio
from speechbrain.pretrained import MSTacotron2
from speechbrain.pretrained import HIFIGAN

# Intialize TTS (mstacotron2) and Vocoder (HiFIGAN)
ms_tacotron2 = MSTacotron2.from_hparams(source="speechbrain/tts-mstacotron2-libritts", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir="tmpdir_vocoder")

# Required input
INPUT_TEXT = "The quick brown fox jumps over the lazy dog."

# Running the Zero-Shot Multi-Speaker Tacotron2 model to generate mel-spectrogram
mel_outputs, mel_lengths, alignments = ms_tacotron2.generate_random_voice(INPUT_TEXT)

# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_outputs)

# Save the waverform
torchaudio.save("synthesized_sample1.wav", waveforms.squeeze(1).cpu(), 22050)