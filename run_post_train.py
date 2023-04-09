import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    BartForConditionalGeneration, AutoTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, HfArgumentParser, set_seed
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset
from dataset import DataCollatorForDialougeDenoising

check_min_version("4.9.0")

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")


@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: str = field(default=None)
    # max_length: Optional[int] = field(default=256)
    train_file: str = field(default=None)
    valid_file: str = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    dialogue_max_seq_length: Optional[int] = field(default=1024)
    pad_to_max_length: bool = field(default=True)
    masking_ratio: Optional[float] = field(default=0.3)
    turn_permutation_rate: Optional[float] = field(default=0.1)
    max_span_length: Optional[int] = field(default=1)
    preprocessing_num_workers: Optional[int] = field(default=None)


def main():
    parser = HfArgumentParser((Seq2SeqTrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

    # We use wandb logger: https://wandb.ai/site.
    # if training_args.local_rank == 0:  # only on main process
    #     # Initialize wandb run
    #     wandb.login()
    #     wandb.init(project="Dialogue_Summary", name=training_args.run_name)

    # Set seed before initializing model
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(run_args.model_path)
    model = BartForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
    # tokenizer = BartTokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
    # if run_args.model_path:
    #     model = BartForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
    # else:
    #     model = BartForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
    #print(tokenizer.encode("P03:"))
    #print("-------------")
    data_files = {}
    if run_args.train_file is not None:
        data_files["train"] = run_args.train_file

    extension = run_args.train_file.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files)

    column_names = datasets["train"].column_names
    def tokenize_function(examples):
        return tokenizer(examples["dialogue"], padding="max_length" if run_args.pad_to_max_length else False,
                         truncation=True, max_length=run_args.dialogue_max_seq_length)

    with training_args.main_process_first(desc="train dataset map pre-processing"):
        tokenized_dataset = datasets["train"].map(
            tokenize_function,
            batched=True,
            num_proc=run_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not run_args.overwrite_cache,
        )

        speaker_ids = tokenizer.convert_tokens_to_ids(["P01:", "P02:", "P03:", "P04:", "P05:", "P06:", "P07:", "P08:", "P09:"])
    # Data collator
    data_collator = DataCollatorForDialougeDenoising(
        tokenizer=tokenizer,
        masking_ratio=run_args.masking_ratio,
        permutate_turn_ratio=run_args.turn_permutation_rate,
        speaker_ids=speaker_ids,
        return_tensors="np"
    )

    trainer = Seq2SeqTrainer(model,
                             args=training_args,
                             train_dataset=tokenized_dataset,
                             eval_dataset=None,
                             tokenizer=tokenizer,
                             data_collator=data_collator,
                             )

    trainer.train()
    trainer.save_model(run_args.run_name)


if __name__ == "__main__":
    main()
