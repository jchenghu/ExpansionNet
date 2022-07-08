import os
import random
import math
import torch
import argparse
from argparse import Namespace
from utils.args_utils import str2list, str2bool
import copy
from time import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from models.ensemble_captioning_model import EsembleCaptioningModel
from data.mscoco2014_dataset_manager import MsCocoDataLoader, MsCocoDatasetKarpathy
from utils import language_utils
from utils.language_utils import compute_num_pads
from utils.parallel_utils import dist_gather_object
from models.Expansion_Net import ExpansionNet
from eval.eval import COCOEvalCap

from torch.optim.lr_scheduler import SAVE_STATE_WARNING
import warnings
warnings.filterwarnings("ignore", message=SAVE_STATE_WARNING)
import functools
print = functools.partial(print, flush=True)

def convert_time_as_hhmmss(ticks):
    return str(int(ticks / 60)) + " m " + \
           str(int(ticks) % 60) + " s"

def compute_evaluation_phase_loss(loss_function,
                                  model,
                                  data_set,
                                  data_loader,
                                  num_samples,
                                  sub_batch_size,
                                  dataset_split,
                                  rank=0,
                                  verbose=False):
    model.eval()

    sb_size = sub_batch_size
    batch_input_x, batch_target_y, batch_input_x_num_pads, batch_target_y_num_pads, batch_rand_indexes \
        = data_loader.get_random_samples(num_samples=num_samples,
                                         dataset_split=dataset_split)

    tot_loss = 0
    num_sub_batch = int(num_samples / sb_size)
    tot_num_tokens = 0
    for sb_it in range(num_sub_batch):
        from_idx = sb_it * sb_size
        to_idx = (sb_it + 1) * sb_size

        sub_batch_input_x = batch_input_x[from_idx: to_idx].to(rank)
        sub_batch_target_y = batch_target_y[from_idx: to_idx].to(rank)
        tot_num_tokens += sub_batch_target_y.size(1)*sub_batch_target_y.size(0) - \
                          sum(batch_target_y_num_pads[from_idx: to_idx])
        pred = model(enc_x=sub_batch_input_x,
                     dec_x=sub_batch_target_y[:, :-1],
                     enc_x_num_pads=batch_input_x_num_pads[from_idx: to_idx],
                     dec_x_num_pads=batch_target_y_num_pads[from_idx: to_idx],
                     apply_softmax=False)
        tot_loss += loss_function(pred, sub_batch_target_y[:, 1:],
                                  data_set.get_pad_token_idx(),
                                  divide_by_non_zeros=False).item()
    tot_loss /= tot_num_tokens
    if verbose and rank == 0:
        print("Validation Loss on " + str(num_samples) + " samples: " + str(tot_loss))

    return tot_loss


def parallel_evaluate_model(ddp_model,
                            validate_x, validate_y,
                            y_idx2word_list,
                            beam_size, max_seq_len,
                            sos_idx, eos_idx,
                            rank, world_size, ddp_sync_port,
                            parallel_batches=16, verbose=True,
                            return_cider=False,
                            stanford_model_path="./eval/get_stanford_models.sh"):
    # avoid synchronization problems in case of ugly numbers
    assert (len(validate_x) % world_size == 0), "to ensure correctness num test sentences must be multiple of world size" \
        + " for the sake of a cleaner code, maybe in future implementation will be allowed."
    start_time = time()

    sub_list_predictions = []

    # divide validate_x and validate_y by the number of gpus
    sub_sample_size = math.ceil(len(validate_x) / float(world_size))
    sub_validate_x = validate_x[sub_sample_size * rank: sub_sample_size * (rank + 1)]

    ddp_model.eval()
    with torch.no_grad():
        sb_size = parallel_batches
        num_iter_sub_batches = math.ceil(len(sub_validate_x) / sb_size)
        for sb_it in range(num_iter_sub_batches):
            last_iter = sb_it == num_iter_sub_batches - 1
            if last_iter:
                from_idx = sb_it * sb_size
                to_idx = len(sub_validate_x)
            else:
                from_idx = sb_it * sb_size
                to_idx = (sb_it + 1) * sb_size

            sub_batch_x = torch.nn.utils.rnn.pad_sequence(sub_validate_x[from_idx: to_idx], batch_first=True).to(rank)
            sub_batch_x_num_pads = compute_num_pads(sub_validate_x[from_idx: to_idx])

            beam_search_kwargs = {'beam_size': beam_size,
                                  'beam_max_seq_len': max_seq_len,
                                  'sample_or_max': 'max',
                                  'how_many_outputs': 1,
                                  'sos_idx': sos_idx,
                                  'eos_idx': eos_idx}

            output_words, _ = ddp_model(enc_x=sub_batch_x,
                                        enc_x_num_pads=sub_batch_x_num_pads,
                                        mode='beam_search', **beam_search_kwargs)

            # take the first element of beam_size sequences
            output_words = [output_words[i][0] for i in range(len(output_words))]

            pred_sentence = language_utils.convert_allsentences_idx2word(output_words, y_idx2word_list)
            for sentence in pred_sentence:
                sub_list_predictions.append(' '.join(sentence[1:-1]))  # remove EOS and SOS

    ddp_model.train()

    dist.barrier()

    list_sub_predictions = dist_gather_object(sub_list_predictions,
                                              rank,
                                              dst_rank=0,
                                              sync_port=ddp_sync_port)

    if rank == 0 and verbose:
        list_predictions = []
        for sub_predictions in list_sub_predictions:
            list_predictions += sub_predictions

        list_list_references = []
        for i in range(len(validate_x)):
            target_references = []
            for j in range(len(validate_y[i])):
                target_references.append(validate_y[i][j])
            list_list_references.append(target_references)

        gts_dict = dict()
        for i in range(len(list_list_references)):
            gts_dict[i] = [{u'image_id': i, u'caption': list_list_references[i][j]}
                           for j in range(len(list_list_references[i]))]

        pred_dict = dict()
        for i in range(len(list_predictions)):
            pred_dict[i] = [{u'image_id': i, u'caption': list_predictions[i]}]

        coco_eval = COCOEvalCap(gts_dict, pred_dict, list(range(len(list_predictions))),
                                get_stanford_models_path=stanford_model_path)
        score_results = coco_eval.evaluate(bleu=True, rouge=True, cider=True, spice=True, meteor=True, verbose=False)
        elapsed_ticks = time() - start_time
        print("Evaluation Phase over " + str(len(validate_x)) + " BeamSize: " + str(beam_size) +
              "  elapsed: " + str(int(elapsed_ticks/60)) + " m " + str(int(elapsed_ticks % 60)) + ' s')
        print(score_results)

    dist.barrier()

    if return_cider:
        cider = score_results[0]
        _, cider = cider
    else:
        cider = None

    if rank == 0:
        return pred_dict, gts_dict, cider

    return None, None, cider


def evaluate_model_on_set(ddp_model,
                          caption_idx2word_list,
                          sos_idx, eos_idx,
                          num_samples,
                          data_loader,
                          dataset_split,
                          eval_max_len,
                          rank, world_size, ddp_sync_port,
                          parallel_batches=16,
                          beam_sizes=[1],
                          stanford_model_path='./eval/get_stanford_models.sh',
                          get_predictions=False):

    with torch.no_grad():
        ddp_model.eval()
        indexes = range(num_samples)
        val_x = [data_loader.get_bboxes_by_idx(i, dataset_split=dataset_split)
                 for i in indexes]
        val_y = [data_loader.get_all_image_captions_by_idx(i, dataset_split=dataset_split)
                 for i in indexes]
        for beam in beam_sizes:
            pred_dict, gts_dict, cider = parallel_evaluate_model(ddp_model, val_x, val_y,
                                                                 y_idx2word_list=caption_idx2word_list,
                                                                 beam_size=beam, max_seq_len=eval_max_len,
                                                                 sos_idx=sos_idx, eos_idx=eos_idx,
                                                                 rank=rank, world_size=world_size,
                                                                 ddp_sync_port=ddp_sync_port,
                                                                 parallel_batches=parallel_batches,
                                                                 verbose=True,
                                                                 return_cider=True,
                                                                 stanford_model_path=stanford_model_path)

            if rank == 0 and get_predictions:
                return pred_dict, gts_dict, cider

    return None, None, cider


def eval_on_cider_only(ddp_model,
                       y_idx2word_list,
                       sos_idx, eos_idx,
                       num_samples,
                       data_loader,
                       dataset_split,
                       max_seq_len,
                       rank,
                       beam_size,
                       parallel_batches=16,
                       stanford_model_path='./eval/get_stanford_models.sh'):

    indexes = range(num_samples)
    validate_x = [data_loader.get_bboxes_by_idx(i, dataset_split=dataset_split)
             for i in indexes]
    validate_y = [data_loader.get_all_image_captions_by_idx(i, dataset_split=dataset_split)
             for i in indexes]

    ddp_model.eval()

    list_predictions = []
    list_list_references = []

    sb_size = parallel_batches
    num_iter_sub_batches = math.ceil(len(validate_x) / sb_size)
    for sb_it in range(num_iter_sub_batches):
        last_iter = sb_it == num_iter_sub_batches - 1
        if last_iter:
            from_idx = sb_it * sb_size
            to_idx = len(validate_x)
        else:
            from_idx = sb_it * sb_size
            to_idx = (sb_it + 1) * sb_size
        sub_batch_x = torch.nn.utils.rnn.pad_sequence(validate_x[from_idx: to_idx], batch_first=True).to(rank)
        sub_batch_x_num_pads = compute_num_pads(validate_x[from_idx: to_idx])

        beam_search_kwargs = {'beam_size': beam_size,
                              'beam_max_seq_len': max_seq_len,
                              'sample_or_max': 'max',
                              'how_many_outputs': 1,
                              'sos_idx': sos_idx,
                              'eos_idx': eos_idx}

        output_words, _ = ddp_model(enc_x=sub_batch_x,
                                    enc_x_num_pads=sub_batch_x_num_pads,
                                    mode='beam_search', **beam_search_kwargs)

        del sub_batch_x

        # take the first element of beam_size sequences
        output_words = [output_words[i][0] for i in range(len(output_words))]

        pred_sentence = language_utils.convert_allsentences_idx2word(output_words, y_idx2word_list)
        for sentence in pred_sentence:
            list_predictions.append(' '.join(sentence[1:-1]))  # remove EOS and SOS

    for i in range(len(validate_x)):
        target_references = []
        for j in range(len(validate_y[i])):
            target_references.append(validate_y[i][j])
        list_list_references.append(target_references)

    gts_dict = dict()
    for i in range(len(list_list_references)):
        gts_dict[i] = [{u'image_id': i, u'caption': list_list_references[i][j]}
                       for j in range(len(list_list_references[i]))]

    pred_dict = dict()
    for i in range(len(list_predictions)):
        pred_dict[i] = [{u'image_id': i, u'caption': list_predictions[i]}]

    coco_eval = COCOEvalCap(gts_dict, pred_dict, list(range(len(list_predictions))),
                            get_stanford_models_path=stanford_model_path)
    score_results = coco_eval.evaluate(bleu=False, rouge=False, cider=True,
                                       spice=False, meteor=False, verbose=False)

    del validate_x, validate_y

    cider = score_results[0]
    _, cider = cider
    return cider


def get_ensemble_model(reference_model,
                       checkpoints_paths,
                       rank=0):
    import re
    z = sorted(range(len(checkpoints_paths)), key=lambda k: re.search('checkpoint(.+?).pth',
                                                                      checkpoints_paths[k]).group(1))
    checkpoints_paths = [checkpoints_paths[k] for k in z]
    print("Actual order: " + str(checkpoints_paths))

    model_list = []
    for i in range(len(checkpoints_paths)):
        model = copy.deepcopy(reference_model)
        model.to(rank)  # transfer memory to device
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoints_paths[i],
                                map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_list.append(model)

    model = EsembleCaptioningModel(model_list, rank).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    return ddp_model


def distributed_test(rank, world_size,
                     model_args,
                     is_ensemble,
                     mscoco_dataset,
                     eval_parallel_batch_size,
                     eval_beam_sizes,
                     show_predictions,
                     array_of_init_seeds,
                     model_max_len,
                     save_model_path,
                     ddp_sync_port,
                     write_summary):
    print("GPU: " + str(rank) + "] Process " + str(rank) + " working...")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = ddp_sync_port
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

    data_loader = MsCocoDataLoader(mscoco_dataset=mscoco_dataset,
                                   batch_size=eval_parallel_batch_size,  # arbitrary, not important
                                   num_procs=world_size,
                                   array_of_init_seeds=array_of_init_seeds,
                                   dataloader_mode='caption_wise',
                                   rank=rank,
                                   verbose=False)

    if not is_ensemble:
        print("Single model Evaluation")
        checkpoint = torch.load(save_model_path)
        ddp_model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Ensembling Evaluation")
        list_checkpoints = os.listdir(save_model_path)
        checkpoints_list = [ save_model_path + elem for elem in list_checkpoints if elem.startswith('checkpoint')]
        print("Detected checkpoints: " + str(checkpoints_list))

        if len(checkpoints_list) == 0:
            print("No checkpoints found")
            dist.destroy_process_group()
            exit(-1)
        ddp_model = get_ensemble_model(model, checkpoints_list, rank=rank)

    print("Evaluation on Validation Set")
    evaluate_model_on_set(ddp_model, mscoco_dataset.caption_idx2word_list,
                          mscoco_dataset.get_sos_token_idx(), mscoco_dataset.get_eos_token_idx(),
                          mscoco_dataset.val_num_images, data_loader,
                          MsCocoDatasetKarpathy.ValidationSet_ID, model_max_len,
                          rank, world_size, ddp_sync_port,
                          parallel_batches=eval_parallel_batch_size,
                          beam_sizes=eval_beam_sizes)

    print("Evaluation on Test Set")
    pred_dict, gts_dict, _ = evaluate_model_on_set(ddp_model, mscoco_dataset.caption_idx2word_list,
                                                   mscoco_dataset.get_sos_token_idx(), mscoco_dataset.get_eos_token_idx(),
                                                   mscoco_dataset.test_num_images, data_loader,
                                                   MsCocoDatasetKarpathy.TestSet_ID, model_max_len,
                                                   rank, world_size, ddp_sync_port,
                                                   parallel_batches=eval_parallel_batch_size,
                                                   beam_sizes=eval_beam_sizes,
                                                   get_predictions=show_predictions)

    print("[GPU: " + str(rank) + " ] Closing...")
    dist.destroy_process_group()


def spawn_train_processes(model_args,
                          is_ensemble,
                          mscoco_dataset,
                          eval_parallel_batch_size,
                          eval_beam_sizes,
                          show_predictions,
                          num_gpus,
                          ddp_sync_port,
                          save_model_path,
                          write_summary
                          ):

    max_sequence_length = mscoco_dataset.max_seq_len + 20
    print("Max sequence length: " + str(max_sequence_length))
    print("y vocabulary size: " + str(len(mscoco_dataset.caption_word2idx_dict)))

    world_size = torch.cuda.device_count()
    print("Using - ", world_size, " processes / GPUs!")
    assert(num_gpus <= world_size), "requested num gpus higher than the number of available gpus "
    print("Requested num GPUs: " + str(num_gpus))

    # prepare dataloader: it just need to be a number higher than number of needed epoches
    array_of_init_seeds = [random.random() for _ in range(10)]
    mp.spawn(distributed_test,
             args=(num_gpus,
                   model_args,
                   is_ensemble,
                   mscoco_dataset,
                   eval_parallel_batch_size,
                   eval_beam_sizes,
                   show_predictions,
                   array_of_init_seeds,
                   max_sequence_length,
                   save_model_path,
                   ddp_sync_port,
                   write_summary),
             nprocs=num_gpus,
             join=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ExpansionNet Test')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--enc_expand_n', type=int, default=64)
    parser.add_argument('--dec_expand_n', type=int, default=16)
    parser.add_argument('--show_predictions', type=str2bool, default=False)

    parser.add_argument('--is_ensemble', type=str2bool, default=False)

    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--ddp_sync_port', type=int, default=12354)
    parser.add_argument('--save_path', type=str, default='./github_ignore_material/saves/')

    parser.add_argument('--eval_parallel_batch_size', type=int, default=16)
    parser.add_argument('--eval_beam_sizes', type=str2list, default=[1,3])

    parser.add_argument('--write_tensorboard', type=str2bool, default=False)
    parser.add_argument('--images_path', type=str, default="./github_ignore_material/MS_COCO_2014/")
    parser.add_argument('--mscoco_captions_path', type=str, default='./github_ignore_material/raw_data/dataset_coco.json')
    parser.add_argument('--features_path', type=str, default='./github_ignore_material/raw_data/mscoco2014_features.hdf5')
    parser.add_argument('--train2014_bboxes_path', type=str, default='./github_ignore_material/raw_data/train2014_instances.json')
    parser.add_argument('--val2014_bboxes_path', type=str, default='./github_ignore_material/raw_data/val2014_instances.json')
    args = parser.parse_args()
    args.ddp_sync_port = str(args.ddp_sync_port)

    assert (args.eval_parallel_batch_size % args.num_gpus == 0), \
        "num gpus must be multiple of the requested parallel batch size"

    print("is_ensemble: " + str(args.is_ensemble))
    print("eval parallel batch_size: " + str(args.eval_parallel_batch_size))
    print("ddp_sync_port: " + str(args.ddp_sync_port))
    print("save_path: " + str(args.save_path))
    print("num_gpus: " + str(args.num_gpus))

    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)

    model_args = Namespace(model_dim=args.model_dim,
                           N_enc=args.N_enc,
                           N_dec=args.N_dec,
                           dec_expand_n=args.dec_expand_n,
                           enc_expand_n=args.enc_expand_n,
                           dropout=0.0,
                           drop_args=drop_args
                           )

    mscoco_dataset = MsCocoDatasetKarpathy(images_path=args.images_path,
                                           mscoco_annotations_path=args.mscoco_captions_path,
                                           train2014_bboxes_annotations_path=args.train2014_bboxes_path,
                                           val2014_bboxes_annotations_path=args.val2014_bboxes_path,
                                           detected_bboxes_hdf5_filepath=args.features_path,
                                           limited_num_train_images=None,
                                           limited_num_val_images=None)

    # train base model
    spawn_train_processes(model_args=model_args,
                          is_ensemble=args.is_ensemble,
                          mscoco_dataset=mscoco_dataset,
                          eval_parallel_batch_size=args.eval_parallel_batch_size,
                          eval_beam_sizes=args.eval_beam_sizes,
                          show_predictions=args.show_predictions,
                          num_gpus=args.num_gpus,
                          ddp_sync_port=args.ddp_sync_port,
                          save_model_path=args.save_path,
                          write_summary=args.write_tensorboard,
                          )

