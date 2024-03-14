import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import random
from commons.commons import *
import utils
from losses import generator_loss, discriminator_loss, kl_loss, feature_loss
from commons.commons import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.data_utils import TrainHSSDataset
from processing.Mels_preprocess import MelSpectrogramFixed
from hierspeech.synthesizer import SynthesizerTrn
from hierspeech.modules.discriminators import MultiPeriodDiscriminator
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
global_step = 0
current_step = 0

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    rank = 0
    n_gpus = 1
    port = 50000 + random.randint(0, 100)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TrainHSSDataset(hps)
    train_loader = DataLoader(train_dataset, batch_size=hps.train.batch_size,
                              drop_last=True, collate_fn=train_dataset.collate_fn,
                              num_workers=0)

    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)#.eval()
    

    mel_fn = MelSpectrogramFixed(sample_rate=hps.data.sampling_rate,
                                 n_fft=hps.data.filter_length,
                                 win_length=hps.data.win_length,
                                 hop_length=hps.data.hop_length,
                                 f_min=hps.data.mel_fmin,
                                 f_max=hps.data.mel_fmax,
                                 n_mels=hps.data.n_mel_channels,
                                 window_fn=torch.hann_window).cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
    current_step = 0
    print("Load pretrained checkpoint")
   

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scaler = GradScaler(enabled=hps.train.fp16_run)
    max_epoch = epoch_str + hps.train.epochs
    for epoch in range(epoch_str, max_epoch):
        logger.info(f"Epoch: {epoch}/====================================================================/")
        train_and_evaluate(rank, epoch, max_epoch, hps, [net_g, net_d, mel_fn], [optim_g, optim_d], scaler,
                           [scheduler_g, scheduler_d],
                           [train_loader, None], logger, writer, train_dataset)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, max_epoch, hps, nets, optims, scaler, schedulers, loaders, logger, writer, train_dataset):
    net_g, net_d, mel_fn = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    global global_step
    global current_step
    net_g.train()
    net_d.train()
    optim_g.zero_grad()
    optim_d.zero_grad()

    for batch_idx, data in tqdm(enumerate(train_loader), total=int(len(train_dataset) / train_loader.batch_size),
                                desc=f"Epoch {epoch} / {max_epoch}"):
        audio = data[0].cuda(rank)
        w2v = data[1].cuda(rank)
        f0 = data[2].cuda(rank)
        mel = data[3].cuda(rank)
        spec = data[4].cuda(rank)
        audio_lengths = data[5].cuda(rank)
        mel_lengths = data[6].cuda(rank)
        spec_lengths = data[7].cuda(rank)
        w2v_lengths = data[8].cuda(rank)
        
        with autocast(enabled=False):
            g_outs = net_g(audio, spec, w2v, mel, f0, spec_lengths, mel_lengths, w2v_lengths)
            g_mel = mel_fn(g_outs["wave"]).squeeze(1)
            audio = slice_segments(audio, g_outs["slice_ids"] * hps.data.hop_length, hps.train.segment_size)
            y_d_hat_r, y_d_hat_g, _, _ = net_d(audio, g_outs["wave"].detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
            mel_sliced = slice_segments(mel, g_outs["slice_ids"], hps.train.segment_size // hps.data.hop_length)

        
        scaler.scale(loss_disc_all).backward()
        if (batch_idx + 1) % 32 == 0:
            scaler.unscale_(optim_d)     
            grad_norm_d = clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)
            optim_d.zero_grad()

        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(audio, g_outs["wave"])
            with autocast(enabled=False):
                f0 = slice_segments_audio(f0.squeeze(1), g_outs["slice_ids"] * 4, g_outs['f0'].size(-1))
                loss_f0 = F.l1_loss(f0, g_outs["f0"]) * hps.train.c_f0
                loss_mel = F.l1_loss(mel_sliced, g_mel) * hps.train.c_mel
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                # kl loss between fa_za, zsr
                loss_kl_1 = kl_loss(g_outs['fa_za'], g_outs["logs_a"],
                                    g_outs["m_sr"], g_outs["logs_sr"],
                                    g_outs["mel_mask"]) * hps.train.c_kl
                # kl loss between invert_fa_zsr and z_a
                loss_kl_2 = kl_loss(g_outs["invert_fa_zsr"], g_outs["logs_sr"],
                                    g_outs["m_a"], g_outs["logs_a"],
                                    g_outs["w2v_mask"]) * hps.train.c_bi_kl
                # kl loss between fsf_zsr and z_sa
                loss_kl_3 = kl_loss(g_outs["fsf_zsr"], g_outs["logs_sr"],
                                    g_outs["m_sa"], g_outs["logs_sa"],
                                    g_outs["w2v_mask"]) * hps.train.c_bi_kl
                # kl loss between invert_fsf_zsa and zsr
                loss_kl_4 = kl_loss(g_outs["invert_fsf_zsa"], g_outs["logs_sa"],
                                    g_outs["m_sr"], g_outs["logs_sr"],
                                    g_outs["w2v_mask"]) * hps.train.c_kl
                loss_prosody = (torch.sum(
                    torch.abs(mel[:, :hps.model.prosody_size, :] - g_outs["prosody"].float()) * g_outs['w2v_mask'])
                                / (torch.sum(g_outs['w2v_mask']) * hps.model.prosody_size)) * hps.train.c_mel

                loss_gen_all = loss_f0 + loss_mel + loss_fm + loss_gen + loss_kl_1 + loss_kl_2 + loss_kl_3 + loss_kl_4 + loss_prosody

        
        scaler.scale(loss_gen_all).backward()
        if (batch_idx + 1) % 32 == 0:
            scaler.unscale_(optim_g)
            grad_norm_g = clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            optim_g.zero_grad()
            scaler.update()
            torch.cuda.empty_cache()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                all_losses = {"loss_disc": loss_disc, "loss_gen": loss_gen, "loss_f0": loss_f0,
                          "loss_mel": loss_mel, "loss_fm": loss_fm, "loss_kl_1": loss_kl_1,
                          "loss_kl_2": loss_kl_2, "loss_kl_3": loss_kl_3, "loss_kl_4": loss_kl_4,
                          "loss_prosody": loss_prosody}
                logger.info('Train Epoch: {} [{:.0f}%]'.format(epoch, 100. * batch_idx / len(train_loader)))
                logger.info(f"Global_steps:{global_step}. Learning rate:{lr}")

                for key, value in all_losses.items():
                    logger.info(f"{key} : {value}")
                image_dict = {"slice/mel_gen": utils.plot_spectrogram_to_numpy(g_mel[0].data.cpu().numpy()),
                              "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())}
                utils.summarize(writer=writer,
                                global_step=global_step,
                                images=image_dict,
                                scalars=all_losses)
            if global_step % hps.train.save_interval == 0:
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        global_step += 1


if __name__ == "__main__":
    main()