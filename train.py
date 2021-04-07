import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import WarmupLinearSchedule, AdamW
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset: TensorDataset, model, dev_dataset: TensorDataset = None):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'topic_id': batch[0],
                      'tweet_text': batch[1],
                      'labels': batch[3]}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            cur_loss = loss.item()
            result = 0
            if (step + 1) % args.gradient_accumulation_steps == 0 and not args.tpu:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        new_result, pred_to_write = evaluate(args, dev_dataset, model)
                        if new_result > result:
                            result = new_result
                            with open('./result/results.csv', 'w') as f:
                                for i, r in enumerate(pred_to_write, 1):
                                    f.write('%d,%d\n' % (i, int(r)))

                    logging_loss = tr_loss
                    print('loss=%f' % (tr_loss / global_step))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, dev_dataset, model):
    model.eval()

    results = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset)
    eval_dataloader = DataLoader(dev_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # Eval!

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'topic_id': batch[0],
                      'tweet_text': batch[1],
                      'labels': batch[3]}
            ori_labels = inputs['labels'].detach().cpu().numpy()

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            fst, snd = tuple(logit.detach().cpu().numpy() for logit in logits)
            rank = 1 * (fst > snd)
            preds = rank
            out_label_ids = ori_labels

        else:
            fst, snd = tuple(logit.detach().cpu().numpy() for logit in logits)
            rank = 1 * (fst > snd)
            preds = np.append(preds, rank, axis=0)
            out_label_ids = np.append(out_label_ids, ori_labels, axis=0)

    result = np.equal(preds, out_label_ids).mean()

    print(out_label_ids.tolist())
    print(preds.tolist())

    real = out_label_ids.tolist()
    real_new = []
    for iterator in real:
        if iterator == -1:
            real_new.append(0)
        else:
            real_new.append(1)

    print(real_new)

    print(confusion_matrix(real_new, preds.tolist()))
    print(classification_report(real_new, preds.tolist()))

    return result, preds


def predict_test_dataset(args, test_dataset, model,test_eg):
    # Prediction on test set
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    prediction_sampler = SequentialSampler(test_dataset)
    prediction_dataloader = DataLoader(test_dataset, sampler=prediction_sampler, batch_size=args.eval_batch_size)

    preds = None

    for batch in tqdm(prediction_dataloader, desc="Testing"):
        # Put model in evaluation mode
        model.eval()

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'topic_id': batch[0],
                      'tweet_text': batch[1]}

            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                fst, snd = tuple(logit.detach().cpu().numpy() for logit in logits)
                rank = 1 * (fst > snd)
                preds = rank

            else:
                fst, snd = tuple(logit.detach().cpu().numpy() for logit in logits)
                rank = 1 * (fst > snd)
                preds = np.append(preds, rank, axis=0)

    with open('./Test_Data/test_results.csv', 'w') as f:
        for i, r in enumerate(preds, 0):
            f.write('%s\t%s\t%f\t%s\n' % (test_eg[i].topic_id, test_eg[i].tweet_id, float(r), 'uaics-1'))

    return preds
