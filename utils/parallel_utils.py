import os
import pickle

import torch.distributed as dist


def dist_gather_object(my_object, my_rank, dst_rank, sync_port):
    """
    Since no all torch version support gather function, we implement it
    ourselves requiring very basics library such as:
        os and pickle.
    The sync_port serves the purpose of mitigating synchronization conflicts
    """
    # save my obj into file pickle and wait until every one do that step
    save_path_tmp_file = './gatherop' + sync_port + '_' + str(my_rank) + '_.pl'
    with open(save_path_tmp_file, 'wb') as f:
        pickle.dump(my_object, f)
    dist.barrier()
    # make sure rank zero collect all the info from file
    list_objects = []
    if my_rank == dst_rank:
        ls_files = os.listdir('./')
        # sort by ranks
        ls_files.sort()
        for file_name in ls_files:
            if file_name.startswith('gatherop' + sync_port):
                _, detect_rank, _ = file_name.split('_')
                with open('./gatherop' + sync_port + '_' + str(detect_rank) + '_.pl', 'rb') as f:
                    retrieved_obj = pickle.load(f)
                    list_objects.append(retrieved_obj)
    dist.barrier()

    os.remove(save_path_tmp_file)

    if my_rank != dst_rank:
        return None
    else:
        return list_objects
