from commons.commons import *
from .alias_free_torch import *
from .modules.styleencoder import StyleEncoder
from .modules.encoders import PosteriorSFEncoder, PosteriorAudioEncoder
from .modules.flow_modules import ResidualCouplingBlock_Transformer
from .modules.decoder import MelDecoder
from .modules.source_net import SourceNetwork
from .modules.generator import Generator
import torch
import time
import torch.nn as nn


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 gin_channels=256,
                 prosody_size=20,
                 uncond_ratio=0.1,
                 cfg=False,
                 **kwargs):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.mel_size = prosody_size

        # source-filter encoder in paper (speaker agnostic)
        self.enc_p_l = PosteriorSFEncoder(1024, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)

        # BiT Flow
        self.flow_l = ResidualCouplingBlock_Transformer(inter_channels, hidden_channels, 5, 1, 3,
                                                        gin_channels=gin_channels)

        # source-filter encoder in paper (speaker related)
        self.enc_p = PosteriorSFEncoder(1024, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)

        # dual audio acoustic encoder
        self.enc_q = PosteriorAudioEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
                                           gin_channels=gin_channels)

        # BiT Flow
        self.flow = ResidualCouplingBlock_Transformer(inter_channels, hidden_channels, 5, 1, 3,
                                                      gin_channels=gin_channels)

        # Prosody Encoder ?
        self.mel_decoder = MelDecoder(inter_channels,
                                      filter_channels,
                                      n_heads=2,
                                      n_layers=2,
                                      kernel_size=5,
                                      p_dropout=0.1,
                                      mel_size=self.mel_size,
                                      gin_channels=gin_channels)

        # Gw in HAG
        self.dec = Generator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                             upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)

        # Gs in HAG
        self.sn = SourceNetwork(upsample_initial_channel // 2)
        # style encoder
        self.emb_g = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=gin_channels)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if cfg:
            self.emb = torch.nn.Embedding(1, 256)
            torch.nn.init.normal_(self.emb.weight, 0.0, 256 ** -0.5)
            self.null = torch.LongTensor([0]).to(self.device)
            self.uncond_ratio = uncond_ratio
        self.cfg = cfg

            
    def forward(self, audio, spec, w2v, mel, f0, spec_length, mel_length, w2v_length):
        mel_mask = torch.unsqueeze(sequence_mask(mel_length, mel.size(2)), 1).to(mel.dtype)
        w2v_mask = torch.unsqueeze(sequence_mask(w2v_length, w2v.size(2)), 1).to(w2v.dtype)
        spec_mask = torch.unsqueeze(sequence_mask(spec_length, spec.size(2)),1).to(spec.dtype)
        # Style Embedding
        style = self.emb_g(mel, mel_mask)
        print(style.size())
        # dual audio acoustic encoder
        z_a, m_a, logs_a = self.enc_q(spec, audio, spec_mask, g=style)
        # forward flow fa
        fa_za = self.flow(z_a, spec_mask, g=style)
        # Speaker-related source-filter encoder
        z_sr, m_sr, logs_sr = self.enc_p(w2v, f0, w2v_mask, g=style)
        # inverted flow fa
        invert_fa_zsr = self.flow(z_sr, w2v_mask, g=style, reverse=True)
        # forward flow fsf
        fsf_zsr = self.flow_l(z_sr, w2v_mask, g=style)
        # Prosody encoder get first 20 mel bins
        prosody = self.mel_decoder(z_sr, w2v_mask)
        # Speaker-agnostic source-filter encoder
        z_sa, m_sa, logs_sa = self.enc_p_l(w2v, f0, w2v_mask, g=style)
        # inverted flow fsf
        invert_fsf_zsa = self.flow_l(z_sa, w2v_mask, g=style, reverse=True)
        # random unconditional generate (style embedding is zero)
        prob = torch.randint(low=1, high=100, size=(1,))
        if prob <= 10:
            style = torch.zeros_like(style)
        # forward z_a to the hierchical adaptive generator
        # first, select a random feature segment
        z_a_slice, slice_ids = rand_slice_segments(z_a, spec_length, self.segment_size)
        # first, forward through the source generator
        p_h, f0 = self.sn(z_a_slice, style)
        f0 = f0.squeeze(1)

        # and through the waveform generator
        wave_out = self.dec(z_a_slice, p_h, style)
        output = {'wave': wave_out, 'f0': f0, 'slice_ids': slice_ids,
                  'mel_mask': mel_mask, 'w2v_mask': w2v_mask,
                  'z_a': z_a, 'm_a': m_a, 'logs_a': logs_a, 'fa_za': fa_za,
                  'z_sr': z_sr, 'm_sr': m_sr, 'logs_sr': logs_sr,
                  'invert_fa_zsr': invert_fa_zsr, 'fsf_zsr': fsf_zsr,
                  'z_sa': z_sa, 'm_sa': m_sa, 'logs_sa': logs_sa,
                  'invert_fsf_zsa': invert_fsf_zsa, 'prosody': prosody}

        return output

    @torch.no_grad()
    def infer(self, x_mel, w2v, length, f0):

        x_mask = torch.unsqueeze(sequence_mask(length, x_mel.size(2)), 1).to(x_mel.dtype)

        # Speaker embedding from mel (Style Encoder)
        g = self.emb_g(x_mel, x_mask).unsqueeze(-1)

        z, _, _ = self.enc_p_l(w2v, f0, x_mask, g=g)

        z = self.flow_l(z, x_mask, g=g, reverse=True)
        z = self.flow(z, x_mask, g=g, reverse=True)

        e, e_ = self.sn(z, g)
        o = self.dec(z, e, g=g)

        return o, e_

    @torch.no_grad()
    def voice_conversion(self, src, src_length, trg_mel, trg_length, f0, noise_scale=0.333, uncond=False):

        trg_mask = torch.unsqueeze(sequence_mask(trg_length, trg_mel.size(2)), 1).to(trg_mel.dtype)
        g = self.emb_g(trg_mel, trg_mask).unsqueeze(-1)

        y_mask = torch.unsqueeze(sequence_mask(src_length, src.size(2)), 1).to(trg_mel.dtype)
        z, m_p, logs_p = self.enc_p_l(src, f0, y_mask, g=g)

        z = (m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale) * y_mask

        z = self.flow_l(z, y_mask, g=g, reverse=True)
        z = self.flow(z, y_mask, g=g, reverse=True)

        if uncond:
            null_emb = self.emb(self.null) * math.sqrt(256)
            g = null_emb.unsqueeze(-1)

        e, _ = self.sn(z, g)
        o = self.dec(z, e, g=g)

        return o

    @torch.no_grad()
    def voice_conversion_noise_control(self, src, src_length, trg_mel, trg_length, f0, noise_scale=0.333, uncond=False,
                                       denoise_ratio=0):
        t1 = time.perf_counter()
        trg_mask = torch.unsqueeze(sequence_mask(trg_length, trg_mel.size(2)), 1).to(trg_mel.dtype)
        t2 = time.perf_counter()
        print('step1:', t2-t1)
        t3 = time.perf_counter()
        g = self.emb_g(trg_mel, trg_mask).unsqueeze(-1)
        t4 = time.perf_counter()
        print('step2:', t4-t3)
        t5 = time.perf_counter()
        g_org, g_denoise = g[:1, :, :], g[1:, :, :]
        t6 = time.perf_counter()
        print('step3:', t6-t5)
        t7 = time.perf_counter()
        g_interpolation = (1 - denoise_ratio) * g_org + denoise_ratio * g_denoise
        t8 = time.perf_counter()
        print('step4:', t8-t7)
        t9 = time.perf_counter()
        y_mask = torch.unsqueeze(sequence_mask(src_length, src.size(2)), 1).to(trg_mel.dtype)
        t10 = time.perf_counter()
        print('step5:', t10-t9)
        t11 = time.perf_counter()
        z, m_p, logs_p = self.enc_p_l(src, f0, y_mask, g=g_interpolation)
        t12 = time.perf_counter()
        print('step6:', t12-t11)
        t13 = time.perf_counter()
        z = (m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale) * y_mask
        t14 = time.perf_counter()
        print('step7:', t14-t13)
        z = self.flow_l(z, y_mask, g=g_interpolation, reverse=True)
        t15 = time.perf_counter()
        z = self.flow(z, y_mask, g=g_interpolation, reverse=True)
        t16 = time.perf_counter()
        print('step8:', t16-t15)

        if uncond:
            null_emb = self.emb(self.null) * math.sqrt(256)
            g = null_emb.unsqueeze(-1)

        t17 = time.perf_counter()
        e, _ = self.sn(z, g_interpolation)
        t18 = time.perf_counter()
        print('step9:', t18-t17)
        t19 = time.perf_counter()
        o = self.dec(z, e, g=g_interpolation)
        t20 = time.perf_counter()
        print('step10:', t20-t19)
        return o

    @torch.no_grad()
    def f0_extraction(self, x_linear, x_mel, length, x_audio, noise_scale=0.333):

        x_mask = torch.unsqueeze(sequence_mask(length, x_mel.size(2)), 1).to(x_mel.dtype)

        # Speaker embedding from mel (Style Encoder)
        g = self.emb_g(x_mel, x_mask).unsqueeze(-1)

        # posterior encoder from linear spec.
        _, m_q, logs_q = self.enc_q(x_linear, x_audio, x_mask, g=g)
        z = (m_q + torch.randn_like(m_q) * torch.exp(logs_q) * noise_scale)

        # Source Networks
        _, e_ = self.sn(z, g)

        return e_
