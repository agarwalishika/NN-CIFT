from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import torch

class FineTunedModel():
    def __init__(self, model_name) -> None:
        # Load the base model and tokenizer
        self.model_name = model_name  # or any other conversational model
        quant_storage_dtype = torch.bfloat16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=quant_storage_dtype,
            device_map='auto'
        )
        
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.qlora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",  # Target specific modules for LoRA injection
            # target_modules=[               # List of module names to apply LoRA to
            #     'q_proj',                  # Query projection in attention mechanism
            #     'k_proj',                  # Key projection in attention mechanism
            #     'v_proj',                  # Value projection in attention mechanism
            #     'o_proj',                  # Output projection in attention mechanism
            #     'gate_proj',               # Gate projection
            #     'up_proj',                 # Up projection in feed-forward network
            #     'down_proj',               # Down projection in feed-forward network
            #     'lm_head',                 # Language model head
            # ],
            lora_dropout=0.05,
            bias="none"
        )
        self.model = get_peft_model(self.model, self.qlora_config)

    def fine_tune(self, inputs, outputs, model_dir):
        def tokenize_function(sample):
            conversation = sample['input_text']
            response = sample['response_text']
            
            # Tokenize input and response
            input_ids = self.tokenizer(conversation, add_special_tokens=False).input_ids
            response_ids = self.tokenizer(response, add_special_tokens=False).input_ids

            # Create labels: mask input part with -100 to ignore in loss calculation
            labels = [-100] * len(input_ids) + response_ids

            max_length = 1024
            padded_input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding="max_length", max_length=max_length)["input_ids"]
            padded_labels = self.tokenizer.pad({"input_ids": labels}, padding="max_length", max_length=max_length)["input_ids"]

            # Return both padded input_ids and labels
            return {
                "input_ids": padded_input_ids,
                "labels": padded_labels
            }


        dataset = Dataset.from_dict({"input_text": inputs, "response_text": outputs})
        tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=model_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10,
            evaluation_strategy="no",
            save_strategy="no",
            dataloader_num_workers=4,
            fp16=True  # Use mixed precision if supported
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            # eval_dataset=tokenized_dataset["validation"],
        )

        # Fine-tune the model
        trainer.train()
        trainer.save_model()

    def generate(self, prompts, sampling_params, batch_size=4):
        def encode_data_point(prompt):
            encoded = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.model.device)

            inputs = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
            }

            return inputs

        decoded_outputs, confidences = [], []
        prompts = list(prompts)

        for_range = tqdm(range(0, len(prompts), batch_size), desc="batch inference") if len(prompts) > 1 else range(0, len(prompts), batch_size)

        for i in for_range:
            batch_prompts = prompts[i:i+batch_size]
            encoded_input = encode_data_point(batch_prompts)
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded_input,
                    temperature=sampling_params.temperature,
                    do_sample=True,
                    max_new_tokens=100,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            answer_start = len(encoded_input['input_ids'][0])
            decoded_output = [
                self.tokenizer.decode(sequence[answer_start:], skip_special_tokens=True)
                for sequence in outputs.sequences
            ]

            if len(batch_prompts) > 1:
                get_scores = lambda i: torch.stack([outputs.scores[j][i] for j in range(len(outputs.scores))]) 
                scores = [get_scores(i) for i in range(len(batch_prompts))]
            else:
                scores = torch.stack(outputs.scores)

            batch_confs = []
            for score in scores:
                probs = F.softmax(score, dim=-1)
                top_probs, _ = torch.topk(probs, k=2, dim=-1)
                batch_confs.append((top_probs[:, 0] - top_probs[:, 1]).mean())

            decoded_outputs.extend(decoded_output)
            confidences.extend(batch_confs)
            
        return decoded_output, confidences