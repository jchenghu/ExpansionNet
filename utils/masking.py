
import torch


# return a rectangular pad tensor of shape [ batch_size, rows, columns ] for masking which
# padding elments are specified the two arguments
def create_pad_mask(mask_size, pad_along_row_input, pad_along_column_input, rank):
    batch_size, output_seq_len, input_seq_len = mask_size
    # create first mask to remove influence of PADs in attention matrix
    mask = torch.ones(size=(batch_size, output_seq_len, input_seq_len), dtype=torch.int8).to(rank)

    for batch_idx in range(batch_size):
        mask[batch_idx, :, (input_seq_len - pad_along_column_input[batch_idx]):] = 0
        mask[batch_idx, (output_seq_len - pad_along_row_input[batch_idx]):, :] = 0
    return mask


# return a square pad tensor of shape[ batch_size, len, len ] for masking which
# padding elments are specified num_pads and additionally the upper triangular is zero-ed
# to provide a "no_peak" property
def create_no_peak_and_pad_mask(mask_size, num_pads, rank):
    batch_size, seq_len, seq_len = mask_size
    # create first mask to remove influence of PADs in attention matrix
    mask = torch.tril(torch.ones(size=(seq_len, seq_len), dtype=torch.int8),
                      diagonal=0).unsqueeze(0).repeat(batch_size, 1, 1).to(rank)
    for batch_idx in range(batch_size):
        mask[batch_idx, :, seq_len - num_pads[batch_idx]:] = 0
        mask[batch_idx, (seq_len - num_pads[batch_idx]):, :] = 0
    return mask
