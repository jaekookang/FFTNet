from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from utils import audio
from nnmnkwii.datasets import cmu_arctic
from nnmnkwii import preprocessing as P
import librosa

from utils import is_mulaw_quantize, is_mulaw, is_raw


def build_from_path(hparams, in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    # speakers = cmu_arctic.available_speakers
    speakers = ['bdl', 'clb', 'rms', 'slt']

    wd = cmu_arctic.WavFileDataSource(in_dir, speakers=speakers)
    wav_paths = wd.collect_files()
    speaker_ids = wd.labels

    for index, (speaker_id, wav_path) in enumerate(
            zip(speaker_ids, wav_paths)):
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, index + 1, speaker_id, wav_path, "N/A", hparams)))
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, speaker_id, wav_path, text, hparams):

    # Load the audio to a numpy array. Resample if needed
    wav = audio.load_wav(wav_path)
    if hparams.use_injected_noise:
        noise = np.random.normal(
            0.0, 1.0 / hparams.quantize_channels, wav.shape)
        wav += noise

    wav, _ = librosa.effects.trim(wav, top_db=20)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # [0, quantize_channels)
        out = P.mulaw_quantize(wav, hparams.quantize_channels)

        # Trim silences
        start, end = audio.start_and_end_indices(
            out, hparams.silence_threshold)
        wav = wav[start:end]
        out = out[start:end]
        constant_values = P.mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16
    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        out = P.mulaw(wav, hparams.quantize_channels)
        constant_values = P.mulaw(0.0, hparams.quantize_channels)
        out_dtype = np.float32
    else:
        # [-1, 1]
        out = wav
        constant_values = 0.0
        out_dtype = np.float32

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    # lws pads zeros internally before performing stft
    # this is needed to adjust time resolution between audio and mel-spectrogram
    l, r = audio.lws_pad_lr(wav, hparams.fft_size, audio.get_hop_size())

    # zero pad for quantized signal
    out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * audio.get_hop_size()

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:N * audio.get_hop_size()]
    assert len(out) % audio.get_hop_size() == 0

    timesteps = len(out)

    # Write the spectrograms to disk:
    audio_filename = 'cmu_arctic-audio-%05d.npy' % index
    mel_filename = 'cmu_arctic-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, audio_filename),
            out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return audio_filename, mel_filename, timesteps, text, speaker_id
