##########################
### TRAINING FUNCTIONS ###
##########################
import torch
import random
import os
from .constants import DEVICE, MAX_LENGTH, SOS_token, TEACHER_FORCING_RATIO, HIDDEN_SIZE

# average negative log likelihood of elements corresponding to a 1 in mask tensor


def maskNLLLoss(decoder_out, target, mask):

    nTotal = mask.sum()  # num of elements to consider
    gathered_tensor = torch.gather(
        decoder_out, 1, target.view(-1, 1)).squeeze(1)

    # calculate negative log likelihood loss
    crossEntropy = -torch.log(gathered_tensor)

    # select non-zero elements and calculate the mean and transfer to CUDA
    loss = crossEntropy.masked_select(mask).mean().to(DEVICE)
    return loss, nTotal.item()

# one training step!


def train(input_var, lengths, target_var, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # zero out gradients of previous iterations
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # set device options
    input_var = input_var.to(DEVICE)
    lengths = lengths.to(DEVICE)
    target_var = target_var.to(DEVICE)
    mask = mask.to(DEVICE)

    loss = 0
    print_losses = []
    n_totals = 0

    # STEP 1: forward input batch through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, lengths)

    # STEP 2: Initialize decoder inputs with SOS_token
    decoder_input = torch.LongTensor(
        [[SOS_token for _ in range(batch_size)]]).to(DEVICE)

    # STEP 3: Set initial decoder hidden state to encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    ### Decoding ###
    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False
    for t in range(max_target_len):

        # STEP 4: 1@time input batch sequence forwarding through decoder
        decoder_output, decoder_hidden = decoder(
            decoder_input,
            decoder_hidden,
            encoder_outputs)
        if use_teacher_forcing:
            # Teacher Forcing: next input is current target
            decoder_input = target_var[t].view(1, -1)
        else:
            # NO Teacher Forcing: next input is decoder's current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(DEVICE)

        # STEP 5: calculate & accumulate loss
        mask_loss, nTotal = maskNLLLoss(decoder_output, target_var[t], mask[t])
        if nTotal > 0:
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
            print(mask_loss.item(), nTotal)

    # STEP 6: Perform backpropagation
    loss.backward()

    # STEP 7: Clip gradients (modified in place)
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # STEP 8: Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

# runs n_iterations of training
# saves tarbell of: [to keep and load a checkpoint]
# (1) encoder/decoder state_dicts
# (2) optimizer state_dicts
# (3) loss
# (4) iteration, etc.

########################
### TRAINING PROCESS ###
########################


def trainIters(batch2TrainData, checkpoint, model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # load batches for iteration
    training_batches = [
        batch2TrainData(
            voc,
            [random.choice(pairs) for _ in range(batch_size)]
        ) for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print('Training ...')
    for iteration in range(start_iteration, n_iteration + 1):

        # Batch field extraction
        training_batch = training_batches[iteration - 1]
        input_var, lengths, target_var, mask, max_target_len = training_batch

        # Batch training iteration
        loss = train(input_var, lengths, target_var, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print(loss)
        print_loss += loss

        # Batch finalization
        if iteration % print_every == 0:
            # Print progress
            print_loss_avg = print_loss / print_every
            print('Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}; Loss: {:.4f}'.format(
                iteration,
                iteration / n_iteration * 100,
                print_loss_avg,
                print_loss))
            print_loss = 0

        if iteration % save_every == 0:
            # Save checkpoint
            print('Saving checkpoint ...')
            directory = os.path.join(
                save_dir,
                model_name,
                corpus_name,
                f'{encoder_n_layers}-{decoder_n_layers}_{HIDDEN_SIZE}')
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(
                {'iteration': iteration,
                 'en': encoder.state_dict(),
                 'de': decoder.state_dict(),
                 'en_opt': encoder_optimizer.state_dict(),
                 'de_opt': decoder_optimizer.state_dict(),
                 'loss': loss,
                 'voc_dict': voc.__dict__,
                 'embedding': embedding.state_dict()},
                os.path.join(
                    directory,
                    f'{iteration}_{"checkpoint"}.tar'))
