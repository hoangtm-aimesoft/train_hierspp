import torch
import torchaudio
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from commons import commons
from utils import parse_filelist
from processing.Mels_preprocess import MelSpectrogramFixed, SpectrogramFixed
from audiomentations import Compose, PitchShift, Gain
from hierspeech.modules.wav2vec2 import Wav2vec2
from scipy.io.wavfile import read


class TrainTTVDataset(torch.utils.data.Dataset):
    """
    Provides dataset management for given filelist.
    """
    def __init__(self, hps, training=True):
        super(TrainTTVDataset, self).__init__()
        self.hps = hps
        self.data_ratio = hps.data.train_data_ratio
        self.hop_length = hps.data.hop_length
        self.training = training
        self.mel_length = hps.train.segment_size // hps.data.hop_length
        if self.training:
            self.segment_length = hps.train.segment_size
        self.sample_rate = hps.data.sampling_rate
        self.filelist_path = hps.data.train_filelist_path \
            if self.training else hps.data.test_filelist_path
        self.audio_paths = parse_filelist(self.filelist_path)
        self.f0_paths = parse_filelist(self.filelist_path.replace('_wav', '_f0'))
        self.token_paths = parse_filelist(self.filelist_path.replace('_wav', '_token'))
        self.w2v_paths = parse_filelist(self.filelist_path.replace('_wav', '_w2v'))

    def load_audio_to_torch(self, audio_path):
        audio, sample_rate = torchaudio.load(audio_path)

        p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1]
        audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data

        return audio.squeeze(), sample_rate

    def __getitem__(self, index): 
        audio_path = self.audio_paths[index]
        f0_path = self.f0_paths[index]
        text_path = self.token_paths[index]
        w2v_path = self.w2v_paths[index]
        audio, sample_rate = self.load_audio_to_torch(audio_path)
        f0 = torch.load(f0_path)
        w2v = torch.load(w2v_path, map_location='cpu')
        text_for_ctc = torch.load(text_path)
        text = self.add_blank_token(text_for_ctc)   

        assert sample_rate == self.sample_rate, \
            f"""Got path to audio of sampling rate {sample_rate}, \
                but required {self.sample_rate} according hps."""

        if not self.training:
            return audio, f0, text, w2v

        segment = torch.nn.functional.pad(audio, (0, self.segment_length - audio.shape[-1]), 'constant')
        length = torch.LongTensor([audio.shape[-1] // self.hop_length])

        f0_segment = torch.nn.functional.pad(f0, (0, self.segment_length // 80 - f0.shape[-1]), 'constant')
        w2v = torch.nn.functional.pad(w2v.squeeze(0), (0, self.segment_length // 320 - w2v.shape[-1]), 'constant')

        text_length = torch.LongTensor([text.shape[-1]])
        text = torch.nn.functional.pad(text, (0, 403 - text.shape[-1]), 'constant')

        text_ctc_length = torch.LongTensor([text_for_ctc.shape[-1]])
        text_for_ctc = torch.nn.functional.pad(text_for_ctc, (0, 201 - text_for_ctc.shape[-1]), 'constant')

        return segment, f0_segment, length, text, text_length, w2v, text_for_ctc, text_ctc_length 

    def __len__(self):  
        return len(self.audio_paths)

    def add_blank_token(self, text):
        text_norm = commons.intersperse(text, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm


class TrainHSSDataset(torch.utils.data.Dataset):
    def __init__(self, hps, training=True) -> None:
        super().__init__()
        self.hps = hps
        self.hps = hps
        self.hop_length = hps.data.hop_length
        self.training = training
        self.mel_length = hps.train.segment_size // hps.data.hop_length
        if self.training:
            self.segment_length = hps.train.segment_size
        self.sample_rate = hps.data.sampling_rate
        self.mel_fn = MelSpectrogramFixed(sample_rate=hps.data.sampling_rate,
                                          n_fft=hps.data.filter_length,
                                          win_length=hps.data.win_length,
                                          hop_length=hps.data.hop_length,
                                          f_min=hps.data.mel_fmin,
                                          f_max=hps.data.mel_fmax,
                                          n_mels=hps.data.n_mel_channels,
                                          window_fn=torch.hann_window)
        self.spec_fn = SpectrogramFixed(n_fft=hps.data.filter_length,
                                        win_length=hps.data.win_length,
                                        hop_length=hps.data.hop_length,
                                        )
        self.filelist_path = hps.data.train_filelist_path \
            if self.training else hps.data.test_filelist_path
        self.audio_paths = parse_filelist(self.filelist_path)
        self.f0_paths = parse_filelist(self.filelist_path.replace('_wav', '_f0'))
        self.w2v_paths = parse_filelist(self.filelist_path.replace('_wav', '_w2v'))
        self.pertube = Compose([PitchShift(-4, 4), Gain()])
        self.w2v = Wav2vec2().cpu()

    def __getitem__(self, index):
        # Get per-computed data
        audio_path = self.audio_paths[index]
        f0_path = self.f0_paths[index]
        w2v_path = self.w2v_paths[index]
        # load f0
        f0 = torch.load(f0_path)  # f0_shape: (1, 1, time)
        f0 = torch.log(f0 + 1)
        #f0_segment = torch.nn.functional.pad(f0, (0, self.segment_length // 80 - f0.shape[-1]), 'constant')
        # load w2v
        w2v = torch.load(w2v_path, map_location='cpu').squeeze(0)  # w2v_shape: (1, 1024, time)
        # convert audio from np array to torch tensor
        audio_new, sample_rate = torchaudio.load(audio_path)
        p = (audio_new.shape[-1] // 1280 + 1) * 1280 - audio_new.shape[-1]
        audio_new = torch.nn.functional.pad(audio_new, (0, p), mode='constant').data
        # extract mel and spectrogram
        mel = self.mel_fn(audio_new).squeeze(0)  # mel shape: (n_mels, time)
        spec = self.spec_fn(audio_new).squeeze(0)
        assert sample_rate == self.sample_rate, \
            f"""Got path to audio of sampling rate {sample_rate}, \
                but required {self.sample_rate} according hps."""
        
        return audio_new, w2v, f0, mel, spec

    def __len__(self):
        return len(self.audio_paths)

    def collate_fn(self, batch):
        """
        batch: (audio, w2v, f0, mel, spec) * batch size
        """
        (audio, w2v, f0, mel, spec) = zip(*batch)

        audio_lengths = torch.tensor([x.size(-1) for x in audio])
        w2v_lengths = torch.tensor([x.size(-1) for x in w2v])
        f0_lengths = torch.tensor([x.size(-1) for x in f0])
        mel_lengths = torch.tensor([x.size(-1) for x in mel])
        spec_lengths = torch.tensor([x.size(-1) for x in spec])

        max_audio_length = int(torch.max(audio_lengths))
        max_w2v_length = int(torch.max(w2v_lengths))
        max_f0_length = int(torch.max(f0_lengths))
        max_mel_length = int(torch.max(mel_lengths))
        max_spec_length = int(torch.max(spec_lengths))
        
        audio = [F.pad(x, (0, max_audio_length - x.size(-1))) for x in audio]
        w2v = [F.pad(x, (0, max_w2v_length - x.size(-1))) for x in w2v]
        f0 = [F.pad(x, (0, max_f0_length - x.size(-1))) for x in f0]
        mel = [F.pad(x, (0, max_mel_length - x.size(-1))) for x in mel]
        spec = [F.pad(x, (0, max_spec_length - x.size(-1))) for x in spec]

        audio = torch.stack(audio, dim=0)
        w2v = torch.stack(w2v, dim=0)
        f0 = torch.stack(f0, dim=0)
        mel = torch.stack(mel, dim=0)
        spec = torch.stack(spec, dim=0)
        return audio, w2v, f0, mel, spec, audio_lengths, mel_lengths, spec_lengths, w2v_lengths


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
