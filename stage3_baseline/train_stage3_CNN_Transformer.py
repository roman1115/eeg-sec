import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from split_ser_encoder import SER_Embedding
import commons
import utils
from data_utils import (
  TextAudioSpeakerEmotionEEGLoader,
  TextAudioSpeakerEmotionEEGCollate,
  #make_weights_for_balanced_classes,
  DistributedBucketSampler
)
from model_stage3_CNN_Transformer import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
  entropy,
  classify_loss,
  gaussian_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
torch.backends.cudnn.benchmark = True
global_step = 0
temp = 1.0
train_loss_history = []
eval_loss_history = []


def save_loss_plot(model_dir):
  if not (train_loss_history or eval_loss_history):
    return

  plt.figure()
  if train_loss_history:
    steps, losses = zip(*train_loss_history)
    plt.plot(steps, losses, label="train_loss", color="tab:blue")
  if eval_loss_history:
    steps, losses = zip(*eval_loss_history)
    plt.plot(steps, losses, label="eval_loss", color="tab:orange")
  plt.xlabel("Step")
  plt.ylabel("Loss")
  plt.title("Training vs Validation Loss")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.3)
  plt.tight_layout()
  output_path = os.path.join(model_dir, "loss_curve.png")
  plt.savefig(output_path)
  plt.close()

# def main():
#   """Assume Single Node Multi GPUs Training Only"""
#   assert torch.cuda.is_available(), "CPU training is not allowed."

#   n_gpus = torch.cuda.device_count()
#   os.environ['MASTER_ADDR'] = '127.0.0.1'
#   os.environ['MASTER_PORT'] = '12355'

#   hps = utils.get_hparams()
#   run(0,n_gpus,hps)
#   mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
def main():
    """Assume Single Node Multi GPUs Training Only"""
    if not torch.cuda.is_available():
        print("Warning: GPU is not available. Falling back to CPU training.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = '0.0.0.0'
        os.environ['MASTER_PORT'] = '12355'

    hps = utils.get_hparams()
    if device.type == "cuda":
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
    else:
        run(0, 1, hps)  # 单进程 CPU 训练


def run(rank, n_gpus, hps):
  global global_step
  global temp

  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)
  train_dataset = TextAudioSpeakerEmotionEEGLoader(hps.data.training_files, hps.data)
  
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
     [0, 9999],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerEmotionEEGCollate()
  train_loader = DataLoader(train_dataset, num_workers=2, shuffle=False, pin_memory=True,collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerEmotionEEGLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, batch_size=hps.train.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True, drop_last=False)
  
  ser = SER_Embedding(mel_channels=80, input_channels=1, hidden=256)

  ser_ckpt = os.path.join('checkpts', 'split_ser_embedding.pt')
  if os.path.isfile(ser_ckpt):
      try:
          print(f'Loading pretrained SER from {ser_ckpt} and freezing it')
          ser.load_state_dict(torch.load(ser_ckpt,map_location=f'cuda:{rank}')) # 读取权重并加载到模型中
          ser = ser.cuda(rank)  # 设置为评估模式并移动到 GPU
          ser.train()
          for p in ser.parameters():
              p.requires_grad = False
            # ensure optimizer does not include SER params (we create optimizer after this)
      except Exception as e:
          print(f'Warning: failed to load/freeze SER checkpoint: {e}')

  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

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
  net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval], ser)
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None, ser)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, ser):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders

  MIN_TEMP = torch.tensor(0.1,dtype=torch.float32)
  ANNEAL_RATE = torch.tensor(0.00003,dtype=torch.float32)
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step
  global temp

  net_g.train()
  net_d.train()


  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, sid, eid, eeg) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    sid = sid.cuda(rank, non_blocking=True)
    eid = eid.cuda(rank, non_blocking=True)
    eeg = eeg.cuda(rank, non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q, e, log_q_e, q_e), logits_eeg = net_g(x, x_lengths, spec, spec_lengths, sid, temp, eeg)
      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )
      with torch.no_grad():  # 添加no_grad上下文
        y_hat_mel_ser = y_hat_mel.to(next(ser.parameters()).device)
        e_eeg, logits_o = ser(y_hat_mel_ser)
        e_au , logits_au = ser(y_mel)


      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        eid_sup = eid[eid!=100000]
        eid_sup = eid_sup.reshape(-1,1,1)
        eid_sup = eid_sup.repeat(1,32,1)
        eid_sup = eid_sup.reshape(-1)

        loss_eegcls = 0.1*F.cross_entropy(logits_eeg, eid)

        log_q_e = log_q_e.reshape([z.shape[0], 32, hps.model.num_class])
        log_q_e_unsup = log_q_e[eid==100000]
        log_q_e_sup = log_q_e[eid!=100000]
        log_q_e_unsup = log_q_e_unsup.reshape([-1,hps.model.num_class])
        log_q_e_sup = log_q_e_sup.reshape([-1,hps.model.num_class])

        q_e = q_e.reshape([z.shape[0],32,hps.model.num_class])
        q_e_unsup = q_e[eid==100000]
        q_e_sup = q_e[eid!=100000]
        q_e_unsup = q_e_unsup.reshape([-1,hps.model.num_class])
        q_e_sup = q_e_sup.reshape([-1,hps.model.num_class])

        if q_e_sup.any():
            # alpha = 0.1
            loss_sup = 0.1*classify_loss(q_e_sup, eid_sup)
        else:
            loss_sup = 0.0
        if log_q_e_unsup.any():
            loss_unsup = entropy(log_q_e_unsup, q_e_unsup, hps.model.num_class)
        else:
            loss_unsup = 0.0
        # gamma =1.0
        if loss_unsup<1.0:
            loss_unsup=1.0
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        T=2.0
        logits_au = F.softmax(logits_au / T, dim=1)
        logits_o = F.log_softmax(logits_o / T, dim=1)
        loss_class = F.kl_div(logits_o, logits_au, reduction='batchmean') * (T * T)
        loss_embed = F.mse_loss(e_eeg, e_au)
        loss_emotion = 0.7 * loss_embed + 0.3 * loss_class
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl  + loss_sup + loss_unsup + loss_eegcls + loss_emotion
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl, loss_sup, loss_unsup, loss_eegcls, loss_emotion]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info(f"loss/d/total:{loss_disc_all}, loss/g/total:{loss_gen_all}, lr:{lr}, grad_norm_d:{grad_norm_d}, grad_norm_g:{grad_norm_g}, loss/g/fm:{loss_fm},loss/g/mel:{loss_mel}, loss/g/dur:{loss_dur}, loss/g/kl:{loss_kl}, loss/g/sup:{loss_sup}, loss/g/unsup:{loss_unsup}, loss/g/eegcls:{loss_eegcls}, loss/g/emotion:{loss_emotion}")
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl, "loss/g/sup":loss_sup, "loss/g/unsup":loss_unsup, "loss/g/eegcls":loss_eegcls, "loss/g/emotion":loss_emotion})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)
        train_loss_history.append((global_step, float(loss_gen_all.detach().cpu())))
        save_loss_plot(hps.model_dir)
      if global_step % 1000==1:
        temp = torch.maximum(torch.tensor(1.0,dtype=torch.float32)*torch.exp(-ANNEAL_RATE*global_step),MIN_TEMP) 

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, net_d, eval_loader, writer_eval, logger, ser)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, net_g, net_d, eval_loader, writer_eval, logger=None, ser=None):
    net_g.eval()
    net_d.eval()
    ser.eval()

    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, sid, eid, eeg) in enumerate(eval_loader):
        device = next(net_g.parameters()).device
        x, x_lengths = x.to(device), x_lengths.to(device)
        spec, spec_lengths = spec.to(device), spec_lengths.to(device)
        y, y_lengths = y.to(device), y_lengths.to(device)
        sid, eid = sid.to(device), eid.to(device)
        eeg = eeg.to(device)

        # 使用一个样本进行验证以节省显存，与原逻辑保持一致
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        sid = sid[:1]
        eid = eid[:1].to(device)
        eeg = eeg[:1].to(device)

        with autocast(enabled=hps.train.fp16_run):
          y_hat, l_length, attn, ids_slice, x_mask, z_mask, (
            z, z_p, m_p, logs_p, m_q, logs_q, e, log_q_e, q_e
          ), logits_eeg = net_g(x, x_lengths, spec, spec_lengths, sid, temp, eeg)

          mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)
          y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
          y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
          )
          with torch.no_grad():
            y_hat_mel_ser = y_hat_mel.to(next(ser.parameters()).device)
            e_eeg, logits_o = ser(y_hat_mel_ser)
            e_au , logits_au = ser(y_mel)


          y_slice = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

          # 判别器损失
          y_d_hat_r, y_d_hat_g, _, _ = net_d(y_slice, y_hat.detach())
          with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

        # 生成器损失
        with autocast(enabled=hps.train.fp16_run):
          y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_slice, y_hat)
          with autocast(enabled=False):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_eegcls = 0.1 * F.cross_entropy(logits_eeg, eid)
            eid_sup = eid[eid != 100000]
            if eid_sup.numel() > 0:
              eid_sup = eid_sup.reshape(-1, 1, 1).repeat(1, 32, 1).reshape(-1)

            log_q_e = log_q_e.reshape([z.shape[0], 32, hps.model.num_class])
            log_q_e_unsup = log_q_e[eid == 100000]
            log_q_e_sup = log_q_e[eid != 100000]
            if log_q_e_unsup.numel() > 0:
              log_q_e_unsup = log_q_e_unsup.reshape([-1, hps.model.num_class])
            if log_q_e_sup.numel() > 0:
              log_q_e_sup = log_q_e_sup.reshape([-1, hps.model.num_class])

            q_e = q_e.reshape([z.shape[0], 32, hps.model.num_class])
            q_e_unsup = q_e[eid == 100000]
            q_e_sup = q_e[eid != 100000]
            if q_e_unsup.numel() > 0:
              q_e_unsup = q_e_unsup.reshape([-1, hps.model.num_class])
            if q_e_sup.numel() > 0:
              q_e_sup = q_e_sup.reshape([-1, hps.model.num_class])

            if q_e_sup.numel() > 0:
              loss_sup = 0.1 * classify_loss(q_e_sup, eid_sup)
            else:
              loss_sup = torch.tensor(0., device=device)
            if log_q_e_unsup.numel() > 0:
              loss_unsup = entropy(log_q_e_unsup, q_e_unsup, hps.model.num_class)
            else:
              loss_unsup = torch.tensor(0., device=device)
            if loss_unsup < 1.0:
              loss_unsup = loss_unsup.new_tensor(1.0)
            T = 2.0
            logits_au = F.softmax(logits_au / T, dim=1)
            logits_o = F.log_softmax(logits_o / T, dim=1)
            loss_class = F.kl_div(logits_o, logits_au, reduction='batchmean') * (T * T)
            loss_embed = F.mse_loss(e_eeg, e_au)
            loss_emotion = 0.7 * loss_embed + 0.3 * loss_class
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_sup + loss_unsup + loss_eegcls + loss_emotion

        # 仅处理一个 batch，与原先逻辑一致
        scalar_dict = {
          "eval/loss/d/total": float(loss_disc.detach().cpu()),
          "eval/loss/g/total": float(loss_gen_all.detach().cpu()),
          "eval/loss/g/mel": float(loss_mel.detach().cpu()),
          "eval/loss/g/dur": float(loss_dur.detach().cpu()),
          "eval/loss/g/kl": float(loss_kl.detach().cpu()),
          "eval/loss/g/fm": float(loss_fm.detach().cpu()),
          "eval/loss/g/sup": float(loss_sup.detach().cpu()),
          "eval/loss/g/unsup": float(loss_unsup.detach().cpu()),
          "eval/loss/g/eegcls": float(loss_eegcls.detach().cpu()),
          "eval/loss/g/emotion": float(loss_emotion.detach().cpu())
        }

        eval_loss_history.append((global_step, scalar_dict["eval/loss/g/total"]))
        save_loss_plot(hps.model_dir)

        if logger is not None:
          logger.info(
            "Eval step {}: loss/d/total={:.6f}, loss/g/total={:.6f}, loss/g/mel={:.6f}, "
            "loss/g/dur={:.6f}, loss/g/kl={:.6f}, loss/g/fm={:.6f}, loss/g/sup={:.6f}, "
            "loss/g/unsup={:.6f}, loss/g/eegcls={:.6f}, loss/g/emotion={:.6f}".format(
              global_step,
              scalar_dict["eval/loss/d/total"],
              scalar_dict["eval/loss/g/total"],
              scalar_dict["eval/loss/g/mel"],
              scalar_dict["eval/loss/g/dur"],
              scalar_dict["eval/loss/g/kl"],
              scalar_dict["eval/loss/g/fm"],
              scalar_dict["eval/loss/g/sup"],
              scalar_dict["eval/loss/g/unsup"],
              scalar_dict["eval/loss/g/eegcls"],
              scalar_dict["eval/loss/g/emotion"]
            )
          )

        y_hat_lengths = y_hat.shape[2] * torch.ones(1, dtype=torch.long, device=device)
        image_dict = {
          "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].float().cpu().numpy())
        }
        audio_dict = {
          "gen/audio": y_hat[0, :, :y_hat_lengths[0]].float().cpu()
        }
        if global_step == 0:
          image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].float().cpu().numpy())})
          audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]].float().cpu()})

        utils.summarize(
          writer=writer_eval,
          global_step=global_step,
          images=image_dict,
          audios=audio_dict,
          scalars=scalar_dict,
          audio_sampling_rate=hps.data.sampling_rate
        )
        break

    net_g.train()
    net_d.train()

                           
if __name__ == "__main__":
  main()
