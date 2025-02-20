# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from functools import partial
from typing import Any, Dict, List, Optional, Union, Tuple


import torch
from omegaconf import DictConfig, ListConfig
from torch import nn

from torchtune import config, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message
from torch.utils.data import DataLoader, DistributedSampler
from torchtune.datasets import ConcatDataset
import json
from tqdm import tqdm
import os




logger = utils.get_logger("DEBUG")



class EvaluateRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        # setup model and tokenizer
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
            enable_kv_cache=cfg.enable_kv_cache,
            batch_size=cfg.batch_size,
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        # setup data
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )



    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
        enable_kv_cache: bool = True,
        batch_size: int = 1,
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        if enable_kv_cache:
            with self._device:
                model.setup_caches(batch_size=batch_size, dtype=self._dtype)

        return model
    



    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)

            packed = False
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                utils.padded_collate_left,
                padding_idx=self._tokenizer.pad_id,
            )
            if not packed
            else None,
        )

        logger.info("Dataset and Sampler are initialized.")

        return sampler, dataloader


    def convert_prompt_to_tokens(
        self,
        prompt: Union[DictConfig, str],
        chat_format: Optional[ChatFormat],
        instruct_template: Optional[InstructTemplate],
    ) -> List[Message]:
        """
        Either:
        (1) a raw string is passed as the prompt, in which case we call tokenizer.encode directly, or
        (2) a DictConfig is passed as the prompt. In this case there are three possibilities:
            (a) an InstructTemplate is provided. Since instruct templates output a string, we will
                call tokenizer.encode on the output of the instruct template.
            (b) a ChatFormat is provided. Since chat formats output a list of messages, we will
                call tokenizer.tokenize_messages on the output of the chat format.
            (c) neither an InstructTemplate nor a ChatFormat is provided. In this case we will
                convert the DictConfig to a list of messages and call tokenizer.tokenize_messages directly.
        """

        # Should only be chat-style prompt or instruct-style prompt
        if chat_format and instruct_template:
            raise ValueError(
                "Cannot pass both chat format and instruct template for generation"
            )

        # If instruct template is provided, assert that the prompt is a DictConfig
        # and apply it
        if instruct_template:
            if not isinstance(prompt, DictConfig):
                raise ValueError("Cannot apply instruct template to raw string")
            instruct_template = _get_component_from_path(instruct_template)
            prompt = instruct_template.format(prompt)

        # To hit this block, either the raw prompt is a string or an
        # instruct template has been provided to convert it to a string
        if isinstance(prompt, str):
            return self._tokenizer.encode(prompt, add_bos=True, add_eos=False)

        # dict.items() will respect order for Python >= 3.7
        else:
            messages = [Message(role=k, content=v) for k, v in prompt.items()]
            messages += [Message(role="assistant", content="")]
            if chat_format:
                chat_format = _get_component_from_path(chat_format)
                messages = chat_format.format(messages)
            return self._tokenizer.tokenize_messages(messages)[0]

    @torch.no_grad()
    def evaluate(self, cfg: DictConfig):

        # for save results
        output_dir= cfg.get("output_dir", "./")
        eval_dataset = cfg.dataset.source
        eval_dataset = eval_dataset.split("/")[-1].split(".")[0]
        checkpoint_name = cfg.checkpointer.checkpoint_dir.split("/")[-2]
        checkpoint_file = cfg.checkpointer.checkpoint_files[0].split(".")[0]
        output_name = checkpoint_name + "_" + checkpoint_file
        save_file_path = output_dir + eval_dataset + "/" + output_name + "_res.json"
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        if os.path.exists(save_file_path):
            logger.info(f"File {save_file_path} already exists, skipping evaluation")
            raise ValueError(f"File {save_file_path} already exists, skipping evaluation")

        results = [] # type: List[str]

        for idx, batch in enumerate(tqdm(self._dataloader)):

            tokens = batch["tokens"]
            prompt = tokens[:,:-2].to(self._device).to(torch.int)
            input_seg = batch.get("segment", None)

            print(f"prompt: {prompt}")
            print(f"input_seg: {input_seg}")

            custom_generate_next_token = None

            t0 = time.perf_counter()
            if input_seg is not None:
                input_seg = input_seg.to(self._device).to(torch.int)
                # set the input_seg to the same length as the prompt
                input_seg = input_seg[:, :prompt.size(1)]
                generated_tokens = utils.generate_segment(
                    model=self._model,
                    prompt=prompt,
                    input_seg=input_seg,
                    max_generated_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    stop_tokens=self._tokenizer.stop_tokens,
                    pad_id=self._tokenizer.pad_id,
                    custom_generate_next_token=custom_generate_next_token,
                )
            else:
                generated_tokens = utils.generate(
                    model=self._model,
                    prompt=prompt,
                    max_generated_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    stop_tokens=self._tokenizer.stop_tokens,
                    pad_id=self._tokenizer.pad_id,
                    custom_generate_next_token=custom_generate_next_token,
                )
            t = time.perf_counter() - t0

            text_prompt = self._tokenizer.decode(prompt[0].tolist(), skip_special_tokens=False)
            response =  self._tokenizer.decode(generated_tokens[0][prompt.size(1):-1])
            single_result = {}
            single_result["idx"] = idx
            single_result["text_prompt"] = text_prompt
            single_result["response"] = response
            print(f"text_prompt: {text_prompt}")
            print(f"response: {response}")
            results.append(single_result)

            ## logging information
            # tokens_generated = len(generated_tokens[0]) - prompt.size(0)
            # tokens_sec = tokens_generated / t
            # logger.info(
            #     f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
            # )
            # logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


        
        
        with open(save_file_path, 'w') as f:
            json.dump(results, f, indent=4)
        

@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="EvaluateRecipe", cfg=cfg)
    recipe = EvaluateRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.evaluate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
