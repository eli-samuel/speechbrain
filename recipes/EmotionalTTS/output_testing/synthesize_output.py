import torchaudio
from speechbrain.pretrained import MSTacotron2
from speechbrain.pretrained import HIFIGAN

print("Synthesizing output let's do this.")

# # Intialize TTS (mstacotron2) and Vocoder (HiFIGAN)
# # change source to the path with the model.ckpt + hyperparams.yaml
# ms_tacotron2 = MSTacotron2.from_hparams(source="/home/elisam/projects/def-ravanelm/elisam/speechbrain/recipes/EmotionalTTS/output_testing", savedir="tmpdir_tts")
# hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir="tmpdir_vocoder")

# # Required input
# INPUT_TEXT = "The quick brown fox jumps over the lazy dog."

# # Running the Zero-Shot Multi-Speaker Tacotron2 model to generate mel-spectrogram
# mel_outputs, mel_lengths, alignments = ms_tacotron2.generate_random_voice(INPUT_TEXT)

# # Running Vocoder (spectrogram-to-waveform)
# waveforms = hifi_gan.decode_batch(mel_outputs)

# # Save the waverform
# torchaudio.save("synthesized_sample2.wav", waveforms.squeeze(1).cpu(), 22050)

tmpdir_tts = "/home/elisam/projects/def-ravanelm/elisam/speechbrain/recipes/EmotionalTTS/output_testing/tmpdir_tts"
mstacotron2 = MSTacotron2.from_hparams(source="/home/elisam/projects/def-ravanelm/elisam/speechbrain/recipes/EmotionalTTS/results/tacotron2/1004/save/CKPT+2023-11-21+21-21-05+00", savedir=tmpdir_tts) # doctest: +SKIP
# Sample rate of the reference audio must be greater or equal to the sample rate of the speaker embedding model
reference_audio_path = "/home/elisam/projects/def-ravanelm/datasets/IEMOCAP/IEMOCAP_full_release/Session4/sentences/wav/Ses04F_impro08/Ses04F_impro08_M015.wav"
input_text = "Mary had a little lamb."
mel_output, mel_length, alignment = mstacotron2.clone_voice(input_text, reference_audio_path) # doctest: +SKIP
# One can combine the TTS model with a vocoder (that generates the final waveform)
# Intialize the Vocoder (HiFIGAN)
tmpdir_vocoder = "/home/elisam/projects/def-ravanelm/elisam/speechbrain/recipes/EmotionalTTS/output_testing/tmpdir_vocoder"
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir=tmpdir_vocoder) # doctest: +SKIP
# Running the TTS
mel_output, mel_length, alignment = mstacotron2.clone_voice(input_text, reference_audio_path) # doctest: +SKIP
# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output) # doctest: +SKIP
# For generating a random speaker voice, use the following
mel_output, mel_length, alignment = mstacotron2.generate_random_voice(input_text) # doctest: +SKIP

# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output)

# Save the waverform
torchaudio.save("synthesized_sample2.wav", waveforms.squeeze(1).cpu(), 22050)