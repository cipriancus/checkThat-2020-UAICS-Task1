import torch
import argparse
from transformers import BertConfig
from transformers import BertTokenizer
from task1_prep import DataProcessor
from task1_prep import LABEL2ID
from task1_prep import convert_examples_to_features
from task1_prep import convert_test_to_features
from task1_prep import convert_features_to_test_dataset
from task1_prep import convert_features_to_dataset
from train import predict_test_dataset, evaluate
from train import set_seed
from train import train
from model.bert import Bert

parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", default='exp', type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")

parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")

parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")

parser.add_argument("--max_seq_length", default=121, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")

parser.add_argument("--evaluate_during_training", action='store_true', default=True,
                    help="Rul evaluation during training at each logging step.")

parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for training.")

parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for evaluation.")

parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")

parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")

parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")

parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")

parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")

parser.add_argument("--num_train_epochs", default=5.0, type=float,
                    help="Total number of training epochs to perform.")

parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")

parser.add_argument('--logging_steps', type=int, default=50,
                    help="Log every X updates steps.")

parser.add_argument('--save_steps', type=int, default=100,
                    help="Save checkpoint every X updates steps.")

parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument('--tpu', action='store_true',
                    help="Whether to run on the TPU defined in the environment variables")

parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")

parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")


args = parser.parse_args()

device = torch.device("cpu")
args.n_gpu = torch.cuda.device_count()

print(device)
print(args.n_gpu)

args.device = device

set_seed(args)

dataProcessor = DataProcessor()

dev_eg = dataProcessor.get_dev_examples()

train_eg = dataProcessor.get_train_examples()

test_eg = dataProcessor.get_test_dataset()

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

train_dataset = convert_features_to_dataset(convert_examples_to_features(
    examples=train_eg, label2id=LABEL2ID, max_seq_length=121, tokenizer=tokenizer))

dev_dataset = convert_features_to_dataset(convert_examples_to_features(
    examples=dev_eg, label2id=LABEL2ID, max_seq_length=121, tokenizer=tokenizer))

test_dataset = convert_features_to_test_dataset(convert_test_to_features(examples=test_eg, label2id=LABEL2ID, max_seq_length=121, tokenizer=tokenizer))

config = BertConfig.from_pretrained("bert-large-uncased", num_labels=2)

model = Bert.from_pretrained("bert-large-uncased", config=config, num_labels=2)

model.to(args.device)

pre_trained = False

# if not pre_trained:
#     train(args, train_dataset, model, dev_dataset)
# else:
#     model.load_state_dict(torch.load('./exp/checkpoint-100/pytorch_model.bin'))
#     model.eval()
#
# results = evaluate(args, dev_dataset, model)

predictions = predict_test_dataset(args, test_dataset, model,test_eg)
print(predictions)