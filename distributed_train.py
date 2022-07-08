import os
import random
import numpy as np
import torch
import argparse
from argparse import Namespace
from utils.args_utils import str2bool, str2list, scheduler_type_choice, optim_type_choice
from time import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data.mscoco2014_dataset_manager import MsCocoDataLoader, MsCocoDatasetKarpathy
from utils import language_utils
from utils.saving_utils import load_most_recent_checkpoint, save_last_checkpoint
from losses.loss import LabelSmoothingLoss
from losses.reward import ReinforceCiderReward
from optims.radam import RAdam
from distributed_test import compute_evaluation_phase_loss, evaluate_model_on_set
from models.Expansion_Net import ExpansionNet

# Suppress the following warning:
#  UserWarning: Please also save or load the state of the optimizer when saving or loading the scheduler.
#  warnings.warn(SAVE_STATE_WARNING, UserWarning)
from torch.optim.lr_scheduler import SAVE_STATE_WARNING
import warnings
warnings.filterwarnings("ignore", message=SAVE_STATE_WARNING)
warnings.simplefilter("ignore", category=UserWarning)
import functools
print = functools.partial(print, flush=True)


def convert_time_as_hhmmss(ticks):
    return str(int(ticks / 60)) + " m " + \
           str(int(ticks) % 60) + " s"

def train_xe(rank,
             ddp_model,
             mscoco_dataset,
             data_loader,
             num_accum,
             optimizer,
             sched,
             xe_max_len,
             num_epochs,
             print_every_iter,
             eval_every_iter,
             eval_parallel_batch_size,
             eval_beam_sizes,
             world_size, ddp_sync_port,
             save_every_minutes,
             how_many_checkpoints,
             save_model_path,
             additional_info,
             write_summary=False):

    loss_function = LabelSmoothingLoss(smoothing_coeff=0.1, rank=rank)
    loss_function.to(rank)

    if write_summary:
        writer = SummaryWriter()

    algorithm_start_time = time()
    saving_timer_start = time()
    time_to_save = False
    running_loss = 0
    running_time = 0
    already_trained_steps = data_loader.get_num_batches() * data_loader.get_epoch_it() + data_loader.get_batch_it()
    prev_print_iter = already_trained_steps
    num_iter = data_loader.get_num_batches() * num_epochs
    for it in range(already_trained_steps, num_iter):
        iter_timer_start = time()
        ddp_model.train()

        batch_input_x, batch_target_y, \
        batch_input_x_num_pads, batch_target_y_num_pads, batch_img_idx \
            = data_loader.get_next_batch(verbose=True *
                                                 (((it + 1) % print_every_iter == 0) or
                                                  (it + 1) % data_loader.get_num_batches() == 0),
                                         get_also_image_idxes=True)
        batch_input_x = batch_input_x.to(rank)
        batch_target_y = batch_target_y.to(rank)

        pred_logprobs = ddp_model(enc_x=batch_input_x,
                                  dec_x=batch_target_y[:, :-1],
                                  enc_x_num_pads=batch_input_x_num_pads,
                                  dec_x_num_pads=batch_target_y_num_pads,
                                  apply_softmax=False)

        loss = loss_function(pred_logprobs, batch_target_y[:, 1:], mscoco_dataset.get_pad_token_idx())

        running_loss += loss.item()
        loss.backward()

        if it % num_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        current_rl = sched.get_last_lr()[0]

        running_time += time() - iter_timer_start
        if (it + 1) % print_every_iter == 0:
            avg_loss = running_loss / (it+1 - prev_print_iter)
            tot_elapsed_time = time() - algorithm_start_time
            avg_time_time_per_iter = running_time / (it + 1 - prev_print_iter)
            print('[GPU:' + str(rank) + '] ' + str(round(((it + 1) / num_iter) * 100, 3)) +
                  ' % it: ' + str(it + 1) + ' lr: ' + str(round(current_rl, 12)) +
                  ' n.acc: ' + str(num_accum) +
                  ' avg loss: ' + str(round(avg_loss, 3)) +
                  ' elapsed: ' + convert_time_as_hhmmss(tot_elapsed_time) +
                  ' sec/iter: ' + str(round(avg_time_time_per_iter, 3)))
            running_loss = 0
            running_time = 0
            prev_print_iter = it + 1

        sched.step()

        if ((it + 1) % data_loader.get_num_batches() == 0) or ((it + 1) % eval_every_iter == 0):
            compute_evaluation_phase_loss(loss_function, ddp_model, mscoco_dataset, data_loader,
                                          mscoco_dataset.val_num_images, sub_batch_size=eval_parallel_batch_size,
                                          dataset_split=MsCocoDatasetKarpathy.ValidationSet_ID,
                                          rank=rank, verbose=True)

            if rank == 0:
                print("Evaluation on Validation Set")
            evaluate_model_on_set(ddp_model, mscoco_dataset.caption_idx2word_list,
                                  mscoco_dataset.get_sos_token_idx(), mscoco_dataset.get_eos_token_idx(),
                                  mscoco_dataset.val_num_images, data_loader,
                                  MsCocoDatasetKarpathy.ValidationSet_ID, xe_max_len,
                                  rank, world_size, ddp_sync_port,
                                  parallel_batches=eval_parallel_batch_size,
                                  beam_sizes=eval_beam_sizes)
            time_to_save = True

        # saving
        elapsed_minutes = (time() - saving_timer_start) / 60
        if time_to_save or elapsed_minutes > save_every_minutes or ((it + 1) == num_iter):
            saving_timer_start = time()
            time_to_save = False
            if rank == 0:
                save_last_checkpoint(ddp_model.module, optimizer, sched,
                                     data_loader, save_model_path,
                                     num_max_checkpoints=how_many_checkpoints,
                                     additional_info=additional_info)

            if write_summary and rank == 0 and (it + 1) % print_every_iter:
                ddp_model.module.write_model_summary(writer)


def train_rl(rank, ddp_model,
             mscoco_dataset,
             data_loader,
             num_accum,
             optimizer,
             sched,
             scst_args,
             num_epochs,
             print_every_iter,
             eval_every_iter,
             eval_parallel_batch_size,
             eval_beam_sizes,
             world_size, ddp_sync_port,
             save_every_minutes,
             how_many_checkpoints,
             save_model_path,
             additional_info,
             write_summary=False):

    print("Reinforcement learning Mode")

    num_sampled_captions = 5
    running_logprobs = 0
    running_reward = 0
    running_reward_base = 0

    training_references = mscoco_dataset.get_all_images_captions(MsCocoDatasetKarpathy.TrainSet_ID)
    reinforce_reward = ReinforceCiderReward(training_references, mscoco_dataset.get_eos_token_str(),
                                            num_sampled_captions, rank)

    if write_summary:
        writer = SummaryWriter()

    algorithm_start_time = time()
    saving_timer_start = time()
    time_to_save = False
    running_loss = 0
    running_time = 0
    already_trained_steps = data_loader.get_num_batches() * data_loader.get_epoch_it() + data_loader.get_batch_it()
    prev_print_iter = already_trained_steps
    num_iter = data_loader.get_num_batches() * num_epochs
    for it in range(already_trained_steps, num_iter):
        iter_timer_start = time()
        ddp_model.train()

        batch_input_x, batch_target_y, batch_input_x_num_pads, batch_img_idx \
            = data_loader.get_next_batch(verbose=True *
                                                 (((it + 1) % print_every_iter == 0) or
                                                  (it + 1) % data_loader.get_num_batches() == 0),
                                         get_also_image_idxes=True)
        batch_input_x = batch_input_x.to(rank)

        sampling_search_kwargs = {'sample_max_seq_len': scst_args.scst_max_len,
                                  'how_many_outputs': num_sampled_captions,
                                  'sos_idx': mscoco_dataset.get_sos_token_idx(),
                                  'eos_idx': mscoco_dataset.get_eos_token_idx()}
        all_images_pred_idx, all_images_logprob = ddp_model(enc_x=batch_input_x,
                                                            enc_x_num_pads=batch_input_x_num_pads,
                                                            mode='sampling', **sampling_search_kwargs)

        all_images_pred_caption = [language_utils.convert_allsentences_idx2word(
                                   one_image_pred_idx, mscoco_dataset.caption_idx2word_list) \
                                   for one_image_pred_idx in all_images_pred_idx]

        reward_loss, reward, reward_base \
            = reinforce_reward.compute_reward(all_images_pred_caption=all_images_pred_caption,
                                              all_images_logprob=all_images_logprob,
                                              all_images_idx=batch_img_idx)

        running_logprobs += all_images_logprob.sum().item() / len(torch.nonzero(all_images_logprob, as_tuple=False))
        running_reward += reward.sum().item() / len(reward.flatten())
        running_reward_base += reward_base.sum().item() / len(reward_base.flatten())

        running_loss += reward_loss.item()
        reward_loss.backward()

        if it % num_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        current_rl = sched.get_last_lr()[0]

        running_time += time() - iter_timer_start
        if (it + 1) % print_every_iter == 0:

            avg_loss = running_loss / (it+1 - prev_print_iter)
            tot_elapsed_time = time() - algorithm_start_time
            avg_time_time_per_iter = running_time / (it + 1 - prev_print_iter)
            avg_logprobs = running_logprobs / (it + 1 - prev_print_iter)
            avg_reward = running_reward / (it + 1 - prev_print_iter)
            avg_reward_base = running_reward_base / (it + 1 - prev_print_iter)
            print('[GPU:' + str(rank) + '] ' + str(round(((it + 1) / num_iter) * 100, 3)) +
                  ' % it: ' + str(it + 1) + ' lr: ' + str(round(current_rl, 12)) +
                  ' n.acc: ' + str(num_accum) +
                  ' avg rew loss: ' + str(round(avg_loss, 3)) +
                  ' elapsed: ' + convert_time_as_hhmmss(tot_elapsed_time) +
                  ' sec/iter: ' + str(round(avg_time_time_per_iter, 3)) + '\n'
                  ' avg reward: ' + str(round(avg_reward, 5)) +
                  ' avg base: ' + str(round(avg_reward_base, 5)) +
                  ' avg logprobs: ' + str(round(avg_logprobs, 5)))
            running_loss = 0
            running_time = 0
            running_logprobs = 0
            running_reward = 0
            running_reward_base = 0
            prev_print_iter = it + 1

        sched.step()

        if ((it + 1) % data_loader.get_num_batches() == 0) or ((it + 1) % eval_every_iter == 0):

            if rank == 0:
                with torch.no_grad(), ddp_model.no_sync():
                    print("Evaluation on Validation Set")
                    evaluate_model_on_set(ddp_model, mscoco_dataset.caption_idx2word_list,
                                          mscoco_dataset.get_sos_token_idx(), mscoco_dataset.get_eos_token_idx(),
                                          mscoco_dataset.val_num_images,
                                          data_loader,
                                          MsCocoDatasetKarpathy.ValidationSet_ID, scst_args.scst_max_len,
                                          rank, world_size, ddp_sync_port,
                                          parallel_batches=eval_parallel_batch_size,
                                          beam_sizes=eval_beam_sizes)

            dist.barrier()

            time_to_save = True

        # saving
        elapsed_minutes = (time() - saving_timer_start) / 60
        if time_to_save or elapsed_minutes > save_every_minutes or ((it + 1) == num_iter):
            saving_timer_start = time()
            time_to_save = False
            if rank == 0:
                save_last_checkpoint(ddp_model.module, optimizer, sched,
                                     data_loader, save_model_path,
                                     num_max_checkpoints=how_many_checkpoints,
                                     additional_info=additional_info)

            if write_summary and rank == 0 and (it + 1) % print_every_iter:
                ddp_model.module.write_model_summary(writer)


def distributed_train(rank, world_size,
                      model_args,
                      optim_args,
                      mscoco_dataset,
                      array_of_init_seeds,
                      model_max_len,
                      train_args,
                      scst_args):
    print("GPU: " + str(rank) + "] Process " + str(rank) + " working...")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = train_args.ddp_sync_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Architecture building
    model = ExpansionNet(d_model=model_args.model_dim, N_enc=model_args.N_enc,
                         N_dec=model_args.N_dec, num_heads=8, ff=2048,
                         num_exp_enc=model_args.enc_expand_n,
                         num_exp_dec=model_args.dec_expand_n,
                         output_word2idx=mscoco_dataset.caption_word2idx_dict,
                         max_seq_len=model_max_len, drop_args=model_args.drop_args, rank=rank)
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    if train_args.reinforcement_learning:
        print("Reinforcement learning Mode")
        data_loader = MsCocoDataLoader(mscoco_dataset=mscoco_dataset,
                                       batch_size=train_args.batch_size,
                                       num_procs=world_size,
                                       array_of_init_seeds=array_of_init_seeds,
                                       dataloader_mode='image_wise',
                                       rank=rank,
                                       verbose=True)
    else:
        print("Cross Entropy learning mode")
        data_loader = MsCocoDataLoader(mscoco_dataset=mscoco_dataset,
                                       batch_size=train_args.batch_size,
                                       num_procs=world_size,
                                       array_of_init_seeds=array_of_init_seeds,
                                       dataloader_mode='caption_wise',
                                       rank=rank,
                                       verbose=True)

    if optim_args.sched_type == 'plateau':
        base_lr = optim_args.lr
    else:
        base_lr = 1.0

    if optim_args.optim_type == 'radam':
        optimizer = RAdam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=base_lr,
                          betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=base_lr)

    if optim_args.sched_type == 'annealing':
        sched_func = lambda it: (min(it, optim_args.warmup_iters) / optim_args.warmup_iters) * \
                                optim_args.lr * (0.8 ** (
                    it // (optim_args.anneal_every_epoch * data_loader.get_num_batches())))
    elif optim_args.sched_type == 'noam':
        sched_func = lambda it: pow(model.d_model, -0.5) * min(pow((it + 1), -0.5),
                                                               (it + 1) * pow(optim_args.warmup_iters, -1.5))
    elif optim_args.sched_type == 'custom_warmup_anneal':
        # how warmup and min lr coexist:
        #   if it < warmup iter then is max(0,    custom_warmup_anneal)
        #   if it >= warmup  is max(min_lr, custom_warmup_anneal)
        num_batches = data_loader.get_num_batches()
        sched_func = lambda it: max((it >= optim_args.warmup_iters) * optim_args.min_lr,
                                    (optim_args.lr / (max(optim_args.warmup_iters - it, 1))) * \
                                    (pow(optim_args.anneal_coeff, it // (num_batches * optim_args.anneal_every_epoch)))
                                    )
    elif optim_args.sched_type == 'fixed':
        sched_func = lambda it: optim_args.lr
    else:  # 'fixed'
        sched_func = lambda it: optim_args.lr

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_func)

    change_from_xe_to_rf = False
    # Loading previous state
    if train_args.save_model_path is not None:
        _, additional_info = load_most_recent_checkpoint(ddp_model.module, optimizer, sched,
                                                         data_loader, rank, train_args.save_model_path)
        if additional_info == 'xe' and train_args.reinforcement_learning:
            change_from_xe_to_rf = True
        else:
            print("Training mode still in the same stage: " + additional_info)

    if data_loader.get_batch_size() != train_args.batch_size or change_from_xe_to_rf:
        print("New requested batch size differ from previous checkpoint", end=" ")
        print("- Proceed to reset training session keeping pre-trained weights")
        data_loader.change_batch_size(batch_size=train_args.batch_size, verbose=True)
        if optim_args.optim_type == 'radam':
            optimizer = RAdam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=1,
                              betas=(0.9, 0.98), eps=1e-9)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=1)

        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_func)

    if change_from_xe_to_rf:
        print("Passing from XE training to RL - Optimizer and data loader states are resetted.")
        if optim_args.optim_type == 'radam':
            optimizer = RAdam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=base_lr,
                              betas=(0.9, 0.98), eps=1e-9)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=base_lr)

        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_func)
        data_loader.set_epoch_it(epoch=0, verbose=True)

    if train_args.reinforcement_learning:
        train_rl(rank, ddp_model,
                 mscoco_dataset, data_loader,
                 train_args.num_accum,
                 optimizer, sched,
                 scst_args,
                 train_args.num_epochs,
                 train_args.print_every_iter, train_args.eval_every_iter,
                 train_args.eval_parallel_batch_size, train_args.eval_beam_sizes,
                 world_size, train_args.ddp_sync_port,
                 train_args.save_every_minutes, train_args.how_many_checkpoints,
                 train_args.save_model_path,
                 additional_info='rf',
                 write_summary=train_args.write_summary)

    else:
        train_xe(rank, ddp_model,
                 mscoco_dataset, data_loader,
                 train_args.num_accum,
                 optimizer, sched,
                 model_max_len,
                 train_args.num_epochs,
                 train_args.print_every_iter, train_args.eval_every_iter,
                 train_args.eval_parallel_batch_size, train_args.eval_beam_sizes,
                 world_size, train_args.ddp_sync_port,
                 train_args.save_every_minutes, train_args.how_many_checkpoints,
                 train_args.save_model_path,
                 additional_info='xe',
                 write_summary=train_args.write_summary)

    print("[GPU: " + str(rank) + " ] Closing...")
    dist.destroy_process_group()


def spawn_train_processes(model_args,
                          optim_args,
                          mscoco_dataset,
                          scst_args,
                          train_args
                          ):

    max_sequence_length = mscoco_dataset.max_seq_len + 20
    print("Max sequence length: " + str(max_sequence_length))
    print("y vocabulary size: " + str(len(mscoco_dataset.caption_word2idx_dict)))

    world_size = torch.cuda.device_count()
    print("Using - ", world_size, " processes / GPUs!")
    assert(train_args.num_gpus <= world_size), "requested num gpus higher than the number of available gpus "
    print("Requested num GPUs: " + str(train_args.num_gpus))

    # prepare dataloader: it just need to be a number higher than number of needed epoches
    array_of_init_seeds = [random.random() for _ in range(train_args.num_epochs*2)]
    mp.spawn(distributed_train,
             args=(train_args.num_gpus,
                   model_args,
                   optim_args,
                   mscoco_dataset,
                   array_of_init_seeds,
                   max_sequence_length,
                   train_args,
                   scst_args,
                   ),
             nprocs=train_args.num_gpus,
             join=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ExpansionNet Train')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=1)
    parser.add_argument('--enc_expand_n', type=int, default=64)
    parser.add_argument('--dec_expand_n', type=int, default=16)
    parser.add_argument('--cross_expand_n', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--enc_drop', type=float, default=0.1)
    parser.add_argument('--dec_drop', type=float, default=0.1)
    parser.add_argument('--enc_input_drop', type=float, default=0.1)
    parser.add_argument('--dec_input_drop', type=float, default=0.1)
    parser.add_argument('--drop_other', type=float, default=0.1)

    parser.add_argument('--optim_type', type=optim_type_choice, default='radam')
    parser.add_argument('--sched_type', type=scheduler_type_choice, default='fixed')

    # scheduler arguments that are used or not according to the scheduler type
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--min_lr', type=float, default=5e-7)
    parser.add_argument('--warmup_iters', type=int, default=4000)
    parser.add_argument('--anneal_coeff', type=float, default=0.8)
    parser.add_argument('--anneal_every_epoch', type=float, default=3.0)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_accum', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--ddp_sync_port', type=int, default=12354)
    parser.add_argument('--save_path', type=str, default='./github_ignore_material/saves/')
    parser.add_argument('--save_every_minutes', type=int, default=25)
    parser.add_argument('--how_many_checkpoints', type=int, default=1)
    parser.add_argument('--print_every_iter', type=int, default=1000)

    # eval_every_iter is set to infinite by default as the evaluation is performed
    # at the end of each epoch anyway
    parser.add_argument('--eval_every_iter', type=int, default=999999)
    parser.add_argument('--eval_parallel_batch_size', type=int, default=16)
    parser.add_argument('--eval_beam_sizes', type=str2list, default=[1,3])

    parser.add_argument('--reinforce', type=str2bool, default=False)
    parser.add_argument('--scst_max_len', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--write_tensorboard', type=str2bool, default=False)
    parser.add_argument('--images_path', type=str, default="./github_ignore_material/MS_COCO_2014/")
    parser.add_argument('--mscoco_captions_path', type=str, default='./github_ignore_material/raw_data/dataset_coco.json')
    parser.add_argument('--features_path', type=str, default='./github_ignore_material/raw_data/mscoco2014_features.hdf5')
    parser.add_argument('--train2014_bboxes_path', type=str, default='./github_ignore_material/raw_data/train2014_instances.json')
    parser.add_argument('--val2014_bboxes_path', type=str, default='./github_ignore_material/raw_data/val2014_instances.json')

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()
    args.ddp_sync_port = str(args.ddp_sync_port)

    # Seed setting ---------------------------------------------
    seed = args.seed
    print("seed: " + str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    drop_args = Namespace(enc=args.enc_drop,
                          dec=args.dec_drop,
                          enc_input=args.enc_input_drop,
                          dec_input=args.dec_input_drop,
                          other=args.drop_other)

    model_args = Namespace(model_dim=args.model_dim,
                           N_enc=args.N_enc,
                           N_dec=args.N_dec,
                           enc_expand_n=args.enc_expand_n,
                           dec_expand_n=args.dec_expand_n,
                           cross_expand_n=args.cross_expand_n,
                           dropout=args.dropout,
                           drop_args=drop_args)
    optim_args = Namespace(lr=args.lr,
                           min_lr=args.min_lr,
                           warmup_iters=args.warmup_iters,
                           anneal_coeff=args.anneal_coeff,
                           anneal_every_epoch=args.anneal_every_epoch,
                           optim_type=args.optim_type,
                           sched_type=args.sched_type)

    train_args = Namespace(batch_size=args.batch_size,
                           num_accum=args.num_accum,
                           num_gpus=args.num_gpus,
                           ddp_sync_port=args.ddp_sync_port,
                           save_model_path=args.save_path,
                           save_every_minutes=args.save_every_minutes,
                           how_many_checkpoints=args.how_many_checkpoints,
                           print_every_iter=args.print_every_iter,
                           eval_every_iter=args.eval_every_iter,
                           eval_parallel_batch_size=args.eval_parallel_batch_size,
                           eval_beam_sizes=args.eval_beam_sizes,
                           reinforcement_learning=args.reinforce,
                           num_epochs=args.num_epochs,
                           write_summary=args.write_tensorboard,)

    scst_args = Namespace(scst_max_len=args.scst_max_len)

    print("train batch_size: " + str(args.batch_size))
    print("num_accum: " + str(args.num_accum))
    print("ddp_sync_port: " + str(args.ddp_sync_port))
    print("save_path: " + str(args.save_path))
    print("num_gpus: " + str(args.num_gpus))

    mscoco_dataset = MsCocoDatasetKarpathy(images_path=args.images_path,
                                           mscoco_annotations_path=args.mscoco_captions_path,
                                           train2014_bboxes_annotations_path=args.train2014_bboxes_path,
                                           val2014_bboxes_annotations_path=args.val2014_bboxes_path,
                                           detected_bboxes_hdf5_filepath=args.features_path,
                                           limited_num_train_images=None,
                                           limited_num_val_images=None)

    # train base model
    spawn_train_processes(model_args=model_args,
                          optim_args=optim_args,
                          mscoco_dataset=mscoco_dataset,
                          train_args=train_args,
                          scst_args=scst_args
                          )


