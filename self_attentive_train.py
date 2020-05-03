import argparse
import logging
import os
import random
import time
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.utils.data.dataset import Dataset
from tqdm import trange, tqdm
from transformers import GPT2Tokenizer

from self_attentive_model import Classifier
from sst_binary import sst_binary

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(self, X, y, tokenizer, block_size=512) -> None:
        self.X = tokenizer.batch_encode_plus(X, add_special_tokens=True, max_length=block_size)["input_ids"]
        self.y = y

    def __getitem__(self, index: int) -> Tuple:
        return torch.tensor(self.X[index], dtype=torch.long), self.y[index]

    def __len__(self) -> int:
        return len(self.X)


def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1, keepdim=True), 2).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


def evaluate(config, eval_dataset, model, tokenizer):
    def collate(examples: List):
        (x, y) = zip(*examples)
        if tokenizer._pad_token is None:
            return pad_sequence(x, batch_first=True), torch.tensor(y, dtype=torch.long)
        return pad_sequence(x, batch_first=True, padding_value=tokenizer.pad_token_id), torch.tensor(y,
                                                                                                     dtype=torch.long)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=config['batch_size'], collate_fn=collate
    )

    criterion = nn.CrossEntropyLoss()

    # Eval!
    logger.info(f'***** Running evaluation *****')
    logger.info(f'  Num examples = {len(eval_dataset)}')
    logger.info(f'  Batch size = {config["batch_size"]}')

    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0

    for (inputs, targets) in tqdm(eval_dataloader, desc='Evaluating'):
        with torch.no_grad():
            inputs = inputs.to(config['device'])
            targets = targets.to(config['device'])

            output, attention = model(inputs)
            output_flat = output.view(inputs.size(1), -1)

            total_loss += criterion(output_flat, targets).item()
            prediction = torch.max(output_flat, 1)[1]
            total_correct += torch.sum((prediction == targets).float())
    return total_loss / (len(eval_dataloader) // config['batch_size']), total_correct / len(eval_dataloader)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def train(config, train_dataset, eval_dataset, model, tokenizer):
    def collate(examples: List):
        (x, y) = zip(*examples)
        if tokenizer._pad_token is None:
            return pad_sequence(x, batch_first=True), torch.tensor(y, dtype=torch.long)
        return pad_sequence(x, batch_first=True, padding_value=tokenizer.pad_token_id), torch.tensor(y,
                                                                                                     dtype=torch.long)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=config['batch_size'], collate_fn=collate
    )

    t_total = len(train_dataloader) // config['num_train_epochs']

    optimizer = Adam(model.parameters(), lr=config['lr'])

    criterion = nn.CrossEntropyLoss()

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info(f"  Num Epochs = {config['num_train_epochs']}")
    logger.info("  Total optimization steps = %d", t_total)

    best_val_loss, best_acc = None, None
    epochs_trained = 0
    global_step = 0

    I = Variable(torch.zeros(config['batch_size'], config['attention_hops'], config['attention_hops']))
    for i in range(config['batch_size']):
        for j in range(config['attention_hops']):
            I.data[i][j][j] = 1

    set_seed(config['seed'])

    train_iterator = trange(
        epochs_trained, int(config['num_train_epochs']), desc="Epoch"
    )
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        total_loss = 0
        total_pure_loss = 0  # without the penalization term
        start_time = time.time()
        for step, batch in enumerate(epoch_iterator):
            data, targets = batch

            data = data.to(config['device'])
            targets = targets.to(config['device'])

            output, attention = model(data)
            loss = criterion(output.view(data.size(0), -1), targets)
            total_pure_loss += loss.item()

            attentionT = torch.transpose(attention, 1, 2).contiguous()
            extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
            loss += config['penalization_coeff'] * extra_loss

            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_step += 1

            total_loss += loss.item()

            if step > 0 and 0 and global_step % config['logging_steps'] == 0:
                elapsed = time.time() - start_time
                print(
                    '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f} | pure loss {:5.4f}'.format(
                        epoch, batch, len(train_dataloader) // config['batch_size'],
                                      elapsed * 1000 / config['logging_steps'],
                                      total_loss[0] / config['logging_steps'],
                                      total_pure_loss[0] / config['logging_steps']))
                total_loss = 0
                total_pure_loss = 0
                start_time = time.time()

        evaluate_start_time = time.time()
        val_loss, acc = evaluate(config, eval_dataset, model, tokenizer)
        print('-' * 89)
        fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
        print(fmt.format((time.time() - evaluate_start_time), val_loss, acc))
        print('-' * 89)
        # Save the model, if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(config['save'], 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:  # if loss doesn't go down, divide the learning rate by 5.
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.2
        if not best_acc or acc > best_acc:
            with open(config['save'][:-3] + '.best_acc.pt', 'wb') as f:
                torch.save(model, f)
            best_acc = acc
        with open(config['save'][:-3] + '.epoch-{:02d}.pt'.format(epoch), 'wb') as f:
            torch.save(model, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention-unit', type=int, default=350,
                        help='number of attention unit')
    parser.add_argument('--attention-hops', type=int, default=1,
                        help='number of attention hops, for multi-hop attention model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')

    parser.add_argument('--nfc', type=int, default=512,
                        help='hidden (fully connected) layer size for classifier MLP')
    parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--logging_steps', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='',
                        help='path to save the final model')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--classes', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--penalization-coeff', type=float, default=1,
                        help='the penalization coefficient')

    parser.add_argument('--model_path', type=str, default='checkpoint-180000')

    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )

    parser.add_argument('--no_cuda', action='store_true', help='Avoid using CUDA when available')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    config = vars(args)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    config['device'] = device
    model = Classifier({
        'dropout': args.dropout,
        'attention-unit': args.attention_unit,
        'attention-hops': args.attention_hops,
        'nfc': args.nfc,
        'classes': args.classes,
        'model_path': args.model_path
    }).to(device)

    tokenizer = GPT2Tokenizer.from_pretrained(config['model_path'])
    gpt_args = torch.load(os.path.join(config['model_path'], 'training_args.bin'))

    trX, vaX, teX, trY, vaY, teY = sst_binary()

    train_dataset = CustomDataset(trX, trY, tokenizer=tokenizer, block_size=gpt_args.block_size)
    eval_dataset = CustomDataset(vaX, vaY, tokenizer=tokenizer, block_size=gpt_args.block_size)

    train(config, train_dataset, eval_dataset, model, tokenizer)
