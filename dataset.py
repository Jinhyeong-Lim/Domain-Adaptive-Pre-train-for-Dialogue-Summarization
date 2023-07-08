from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.random import poisson

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForDialougeDenoising:
    """Data collator used dialouge-related denoising in BART"""

    tokenizer: PreTrainedTokenizerBase
    masking_ratio: float = 0.15
    poisson_lambda: float = 3.0
    permutate_turn_ratio: float = 0.1  # 논문에서는 없음
    pad_to_multiple_of: int = 16
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    speaker_ids: Optional[list] = None

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError

    def __call__(self, examples, return_tensors=None):
        batch = self.tokenizer.pad(
            examples, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors
        )

        decoder_input_ids = self.shift_tokens_right(batch["input_ids"].copy())
        labels = self.make_labels(batch["input_ids"].copy())
        encoder_input_ids = batch["input_ids"].copy()

        switch = np.random.choice([0, 1])
        if switch == 1:
            encoder_input_ids = self.turn_merging(encoder_input_ids)

        encoder_input_ids = self.text_infilling(encoder_input_ids)
        encoder_input_ids = self.turn_permutation(encoder_input_ids)

        encoder_input_ids = torch.tensor(encoder_input_ids)
        encoder_attention_mask = self.make_attention_mask(encoder_input_ids.clone().numpy())
        decoder_attention_mask = self.make_attention_mask(decoder_input_ids.clone().numpy())

        return {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }

    def speaker_mask(self, inputs):
        pass

    def turn_splitting(self, inputs):
        pass

    def turn_merging(self, inputs):
        bsz, max_length = inputs.shape

        for i in range(bsz):
            speakers = np.in1d(inputs[i], self.speaker_ids)
            speaker_indices = np.where(speakers == True)[0]
            turns = len(speaker_indices)

            if turns > 1:
                merge_length = int(np.random.poisson(lam=self.poisson_lambda))
                while turns < merge_length or merge_length < 2:
                    merge_length = int(np.random.poisson(lam=self.poisson_lambda))

                merge_start = int(np.random.randint(0, turns - merge_length + 1))

                del_indices = speaker_indices[merge_start + 1 : merge_start + 1 + merge_length]
                del_indices = np.concatenate([del_indices, del_indices - 1])
                del_indices = del_indices[del_indices < max_length]
                del_length = len(del_indices)
                input_copy = np.delete(inputs[i], del_indices)

                inputs[i] = np.concatenate([input_copy, [self.tokenizer.pad_token_id] * del_length])

        assert bsz, max_length == inputs.shape

        return inputs

    def text_infilling(self, inputs):
        bsz, max_length = inputs.shape

        for i in range(bsz):
            special_ids = self.tokenizer.all_special_ids + self.speaker_ids
            special_tokens_mask = np.in1d(inputs[i], special_ids)  # 2차원일때는 np.isin 써도 되겠다
            is_token = ~special_tokens_mask

            num_to_mask = int(math.ceil(is_token.astype(float).sum() * self.masking_ratio))

            while num_to_mask != 0:
                pad_indices = np.where(inputs[i] == self.tokenizer.pad_token_id)
                window_length = max_length - np.count_nonzero(pad_indices)

                span_length = np.random.poisson(lam=self.poisson_lambda)
                if span_length > num_to_mask:
                    continue
                # span_start = int(np.random.choice(np.where(is_token==True)[0]))
                span_start = np.random.randint(1, window_length - span_length)
                if not all(is_token[span_start : span_start + span_length]):
                    continue
                if span_start + span_length > window_length:
                    continue

                num_to_mask -= span_length

                if span_length != 0:
                    span = np.arange(span_start, span_start + span_length)
                    inputs[i] = np.concatenate(
                        [np.delete(inputs[i], span), [self.tokenizer.pad_token_id] * span_length]
                    )
                else:
                    while (
                        inputs[i][span_start] in special_ids
                        and inputs[i][span_start - 1] in special_ids
                    ):
                        span_start = np.random.randint(1, window_length - span_length)

                inputs_copy = np.insert(inputs[i], span_start, self.tokenizer.mask_token_id)
                inputs[i] = inputs_copy[:max_length]

                special_tokens_mask = np.in1d(inputs[i], special_ids)  # 2차원일때는 np.isin 써도 되겠다
                is_token = ~special_tokens_mask

        return inputs

    def turn_permutation(self, inputs):
        bsz, max_length = inputs.shape
        for i in range(bsz):
            eos_idx = max_length  # exclusive
            pad_indices = np.where(inputs[i] == self.tokenizer.pad_token_id)
            length_before = max_length - np.count_nonzero(pad_indices)

            if np.count_nonzero(pad_indices) != 0:
                eos_idx = eos_idx - np.count_nonzero(pad_indices)

            speakers = np.in1d(inputs[i], self.speaker_ids)
            seps = inputs[i] == self.tokenizer.sep_token_id

            speaker_indices = np.where(speakers == True)[0]
            sep_indices = np.where(seps == True)[0]  # exclusive
            if len(speaker_indices) == len(sep_indices) + 1:
                sep_indices = np.concatenate([sep_indices, [eos_idx]])

            # assert len(speaker_indices) == len(sep_indices), f"{eos_idx}\n{inputs[i][:eos_idx]} \n {self.tokenizer.decode(inputs[i][:eos_idx])}\n{speaker_indices}\n {sep_indices}"

            turn_list = [
                (speaker_idx, sep_idx) for speaker_idx, sep_idx in zip(speaker_indices, sep_indices)
            ]
            turn_list = np.random.permutation(turn_list)

            token_order = []
            for turn in turn_list:
                token_order.extend(np.arange(turn[0], turn[1]).tolist())

            inputs_copy = np.take(inputs[i], token_order)

            speakers = np.in1d(inputs_copy, self.speaker_ids)
            speaker_indices = np.where(speakers == True)[0]

            if len(speaker_indices) > 1:
                inputs_copy = np.insert(
                    inputs_copy, speaker_indices[1:], self.tokenizer.sep_token_id
                )

            if length_before == len(inputs_copy) + 1:
                inputs_copy = np.insert(inputs_copy, -1, self.tokenizer.sep_token_id)

            assert length_before == len(
                inputs_copy
            ), f"{length_before}, {len(inputs_copy)}, \n{inputs[i][:length_before]} \n{inputs_copy} \n{self.tokenizer.decode(inputs_copy)} \n{self.tokenizer.decode(inputs[i][:length_before])}, {self.tokenizer.decode(self.speaker_ids)}"

            inputs[i] = np.concatenate(
                [inputs_copy, [self.tokenizer.pad_token_id] * np.count_nonzero(pad_indices)]
            )

        return inputs

    def shift_tokens_right(self, inputs):
        """Shift decoder input ids right: https://github.com/huggingface/transformers/issues/7961.
        Examples:
            <s>My dog is cute.</s><s>It loves to play in the park.</s><pad><pad>
            shift to -> </s><s>My dog is cute.</s><s>It loves to play in the park.<pad><pad>
        """

        shifted_inputs = np.roll(inputs, 1, axis=-1)

        # replace first token with eos token
        # -> Seongmin : Nope. I need bos
        # shifted_inputs[:, 0] = self.tokenizer.eos_token_id
        shifted_inputs[:, 0] = self.tokenizer.bos_token_id

        # when there's padding, the last eos tokens will not be rotate to first positon
        # we'll need to replace it with a padding token

        # replace eos tokens at the end of sequences with pad tokens
        end_with_eos = np.where(shifted_inputs[:, -1] == self.tokenizer.eos_token_id)
        shifted_inputs[end_with_eos, -1] = self.tokenizer.pad_token_id

        # find positions where where's the token is eos and its follwing token is a padding token
        last_eos_indices = np.where(
            (shifted_inputs[:, :-1] == self.tokenizer.eos_token_id)
            * (shifted_inputs[:, 1:] == self.tokenizer.pad_token_id)
        )

        # replace eos tokens with pad token
        shifted_inputs[last_eos_indices] = self.tokenizer.pad_token_id
        return torch.tensor(shifted_inputs)

    def make_attention_mask(self, inputs):
        """
        padding 부분 mask
        """
        # padding -> 0
        attention_mask = inputs != self.tokenizer.pad_token_id
        attention_mask = attention_mask.astype(float)

        return torch.tensor(attention_mask)

    def make_labels(self, inputs):
        """
        padding 부분 -100
        마지막에 eos 추가.
        """
        bsz, max_length = inputs.shape
        pad_indices = inputs == self.tokenizer.pad_token_id
        eos_indices = [
            max_length - pad_idx.count(True) if pad_idx.count(True) != 0 else max_length - 1
            for pad_idx in pad_indices.tolist()
        ]

        eos_indices = np.array(eos_indices).reshape(bsz, -1)

        np.put_along_axis(inputs, eos_indices, self.tokenizer.eos_token_id, axis=1)
        pad_indices = inputs == self.tokenizer.pad_token_id

        return torch.tensor(inputs)
