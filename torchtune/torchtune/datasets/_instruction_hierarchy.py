# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional


import numpy as np
from torch.utils.data import Dataset
from torchtune.config._utils import _get_component_from_path
from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    InstructTemplate,
    Message,
    validate_messages,
)
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import ModelTokenizer

def load_IH_dataset(source: str):
    import json 
    with open(source, 'r') as f:
        data = json.load(f)
    return data


def message_converter(sample: Mapping[str, Any], train_on_input: bool) -> List[Message]:

    # Extract message content
    sys_message_raw = sample[0]['content']
    user_message_raw = sample[1]['content']
    data_message_raw = sample[2]['content']
    assistant_message_raw = sample[3]['content']

    # Create Message objects
    sys_message = Message(
        role="system",
        content=sys_message_raw,
        masked=not train_on_input
    ) 

    user_message = Message(
        role="user",
        content=user_message_raw,
        masked=not train_on_input
    )

    data_message = Message(
        role="data",
        content=data_message_raw,
        masked=not train_on_input
    )  if data_message_raw.strip() else None

    assistant_message = Message(
        role="assistant",
        content=assistant_message_raw,
        masked=False
    )

    # Compile the list of messages
    messages = [ message for message in [sys_message, user_message, data_message, assistant_message] if message is not None]
    # messages =  [sys_message, user_message, data_message, assistant_message] 

    return messages



def message_converter_eval(sample: Mapping[str, Any], train_on_input: bool) -> List[Message]:

    # Extract message content
    sys_message_raw = sample[0]['content']
    user_message_raw = sample[1]['content']
    data_message_raw = sample[2]['content']

    # Create Message objects
    sys_message = Message(
        role="system",
        content=sys_message_raw,
        masked=False
    )

    user_message = Message(
        role="user",
        content=user_message_raw,
        masked=False
    )

    data_message = Message(
        role="data",
        content=data_message_raw,
        masked=False
    ) if data_message_raw.strip() else None

    assistant_message = Message(
        role="assistant",
        content="",
        masked=False
    )

    # Create a single turn conversation
    messages = [ message for message in [sys_message, user_message, data_message, assistant_message] if message is not None]

    return messages




class InstructHierarchyDataset(Dataset):
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        convert_to_messages: Callable[[Mapping[str, Any]], List[Message]],
        train_on_input: bool = False,
        max_seq_len: Optional[int] = None,
        fraction: float = 1.0,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:

        self._tokenizer = tokenizer
        self._data = load_IH_dataset(source)
        self.train_on_input = train_on_input
        self.max_seq_len = max_seq_len
        self._convert_to_messages = convert_to_messages
        self._data = self._data[: int(fraction * len(self._data))]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        messages = self._convert_to_messages(sample, self.train_on_input)
        # print(messages)
        tokens, mask = self._tokenizer.tokenize_messages(
            messages, max_seq_len=self.max_seq_len
        )

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)
        return {"tokens": tokens, "labels": labels}
    


class InstructHierarchyDatasetSeg(InstructHierarchyDataset):
    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        messages = self._convert_to_messages(sample, self.train_on_input)
        # print(messages)
        tokens, mask, segment = self._tokenizer.tokenize_messages_segment(
            messages, max_seq_len=self.max_seq_len
        )

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels) == len(segment)
        # print("the length of tokens is ", len(tokens))
        return {"tokens": tokens, "labels": labels, "segment": segment}


def InstructHierarchy(
    *,
    tokenizer: ModelTokenizer,
    source: str,
    train_on_input: bool = False,
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> InstructHierarchyDataset:

    ds = InstructHierarchyDataset(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=message_converter,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )

    # if we want to subsample the dataset ds with subsample rate 
    # if subsample_rate is not None:
    # ## get the indeices of the subsample 
    #     indices = np.random.choice(len(ds), int(len(ds)*subsample_rate), replace=False)
    #     ds = torch.utils.data.Subset(ds, indices)
    
    return (
        PackedDataset(ds, max_seq_len=max_seq_len, padding_idx=tokenizer.pad_id)
        if packed
        else ds
    )

def InstructHierarchySeg(
    *,
    tokenizer: ModelTokenizer,
    source: str,
    train_on_input: bool = False,
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> InstructHierarchyDataset:

    ds = InstructHierarchyDatasetSeg(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=message_converter,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )

    # if we want to subsample the dataset ds with subsample rate 
    # if subsample_rate is not None:
    # ## get the indeices of the subsample 
    #     indices = np.random.choice(len(ds), int(len(ds)*subsample_rate), replace=False)
    #     ds = torch.utils.data.Subset(ds, indices)
    
    return (
        PackedDataset(ds, max_seq_len=max_seq_len, padding_idx=tokenizer.pad_id)
        if packed
        else ds
    )



def InstructHierarchyVal(
    *,
    tokenizer: ModelTokenizer,
    source: str,
    train_on_input: bool = False,
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> InstructHierarchyDataset:

    ds = InstructHierarchyDataset(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=message_converter_eval,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )

    return ds


def InstructHierarchyValSeg(
    *,
    tokenizer: ModelTokenizer,
    source: str,
    train_on_input: bool = False,
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> InstructHierarchyDataset:

    ds = InstructHierarchyDatasetSeg(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=message_converter_eval,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )

    return ds