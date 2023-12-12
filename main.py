import os
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from transformers import LlamaTokenizerFast, BitsAndBytesConfig

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from llama.modeling_llama_force_flash_attn2 import LlamaForCausalLM


BASE_MODEL_ID = 'beomi/llama-2-ko-7b'
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj']
TRAIN_ON_INPUTS = True
HF_HUB_REPO = os.environ['HF_HUB_REPO']
if not HF_HUB_REPO:
    raise ValueError(
        'Environment variable `HF_HUB_REPO` must be valid. '
        'Specify `HF_HUB_REPO=entity/repo_name python ...`'
    )


def main():
    gradient_accumulation_steps = 32
    device_map = 'auto'
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK') or 0)
    ddp = world_size != 1
    if ddp:
        device_map = {'': local_rank}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    gradient_accumulation_steps = max(gradient_accumulation_steps, 1)

    training_args = transformers.Seq2SeqTrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=False,
        save_strategy='steps',
        save_steps=100,
        warmup_ratio=0.1,
        num_train_epochs=8,
        learning_rate=1e-5,
        lr_scheduler_type='linear',
        bf16=True,
        logging_steps=10,
        overwrite_output_dir=True,
        output_dir='./llama-code-switching',
        optim='paged_adamw_32bit',
        ddp_find_unused_parameters=False if ddp else None,
        report_to=['wandb'],
        local_rank=local_rank,
    )

    # Load model and tokenizers

    tokenizer = LlamaTokenizerFast.from_pretrained(BASE_MODEL_ID, padding_side='left')
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(base_model, config)
    print(model.eval())

    data_files = {
        'train': './PangyoCorpora.jsonl'
    }
    ds = load_dataset('json', data_files=data_files)

    eos_token_id = tokenizer.eos_token_id

    def _create_text_pair(ko: str, mixed: Optional[str] = None):
        text = ko.strip() + '<|sep|>'
        if mixed:
            text += ' ' + mixed.strip()
        return text

    def _tokenize(text, add_eos_token: bool = True):
        sample = tokenizer(text, truncation=True, max_length=2048, padding=False, return_tensors=None)
        if sample['input_ids'][-1] != eos_token_id and len(sample['input_ids']) < 2048 and add_eos_token:
            sample['input_ids'].append(eos_token_id)
            sample['attention_mask'].append(1)
        sample['labels'] = sample['input_ids'].copy()
        return sample

    def _tokenize_with_masks(sample):
        tokenized = _tokenize(_create_text_pair(sample['korean'], sample['pangyo']))
        if not TRAIN_ON_INPUTS:
            input_tokens = _tokenize(_create_text_pair(sample['korean']), add_eos_token=False)
            input_size = len(input_tokens['input_ids'])
            tokenized['labels'] = [-100] * input_size + tokenized['labels'][input_size:]
        return tokenized

    def _filter_long_texts(tokenized):
        return len(tokenized['input_ids']) < 2048

    train_data = ds['train']
    train_data = train_data.map(_tokenize_with_masks)
    train_data = train_data.filter(_filter_long_texts)
    train_data = train_data.select_columns(['input_ids', 'attention_mask', 'labels'])

    print('Num of training data: {}'.format(len(train_data)))

    model.print_trainable_parameters()

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True,
        ),
    )
    model.config.use_cache = False
    trainer.train()
    model.eval()

    if local_rank == 0:
        print('Pushing the code switching model on HF Hub')
        model.push_to_hub(HF_HUB_REPO, private=True)
        print('All jobs have been finished')


if __name__ == '__main__':
    main()
