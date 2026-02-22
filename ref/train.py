import argparse
from config import *
from training.dataset import load_training_data, prepare_training_data, tokenize_training_data, tokenize_sample
from training.model import load_model
import datasets
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description = "Training Script for Hinglish Translation using QLoRA")

    parser.add_argument("--model_name", type = str, default = MODEL_NAME, help = "Base model name or path for fine-tuning")
    parser.add_argument("--data_path", type = str, default = DATA_DIR / "hinge_train.json", help = "Path to the training data JSON file")
    parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR, help = "Directory to save outputs and checkpoints")
    parser.add_argument("--batch_size", type = int, default = BATCH_SIZE, help = "Batch size for training")
    parser.add_argument("--num_epochs", type = int, default = NUM_EPOCHS, help = "Number of training epochs")
    parser.add_argument("--learning_rate", type = float, default = LEARNING_RATE, help = "Learning rate for optimizer")
    parser.add_argument("--max_samples", type = int, default = MAX_SAMPLES, help = "Maximum number of training samples to use")
    parser.add_argument("--quantization", type = str, choices = ["4bit", "8bit", None], default = None, help = "Quantization method to use for model loading")
    parser.add_argument("--eval_split", type = float, default = 0.1, help = "Fraction of data to use for evaluation")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.model_name in ALLOWED_MODELS, f"Model {args.model_name} is not in the list of allowed models."

    # Load and prepare data
    raw_data = load_training_data(args.data_path, max_samples = args.max_samples)
    training_data = prepare_training_data(raw_data)

    print(f"Loaded {len(training_data)} training samples.")

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name)
    print("Model and tokenizer loaded successfully.")

    # Tokenize data using datasets library for efficiency
    dataset = datasets.Dataset.from_list(training_data)
    tokenized_dataset = dataset.map(
        lambda sample: tokenize_sample(sample, tokenizer), batched = False)

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'labels'])

    if args.eval_split > 0:
        train_split, eval_split = tokenized_dataset.train_test_split(test_size = args.eval_split).values()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model = model,
        label_pad_token_id = -100,
        padding = True,
        return_tensors = "pt"
    )
    print(LORA_CONFIG.target_modules)

    if "phi" in args.model_name.lower():
        LORA_CONFIG.target_modules = ["qkv_proj", "o_proj"]

    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir = args.output_dir,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        num_train_epochs = args.num_epochs,
        learning_rate = args.learning_rate,
        warmup_steps = 30,
        lr_scheduler_type = "linear",
        
        logging_dir = LOG_DIR,
        logging_steps = 10,

        save_steps = 200,
        save_total_limit = 2,

        eval_strategy = "steps" if args.eval_split > 0 else "no",
        eval_steps = 200,
        load_best_model_at_end = True if args.eval_split > 0 else False,
        greater_is_better = False,

        fp16 = True,                          # mandatory for T4
        optim = "adamw_torch_fused",          # fastest for FP16
        report_to = "none",

        gradient_checkpointing = True,        # huge VRAM saver
        remove_unused_columns = False,        # for causal LM datasets
    )

    training_args = TrainingArguments(
        output_dir = "./cpu_test",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        max_steps = 5,                 
        logging_steps = 1,
        save_total_limit = 1,

        no_cuda = True,               # force CPU training
        fp16 = False,
        bf16 = False,
        report_to = "none",
        remove_unused_columns = False,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_split,
        eval_dataset = eval_split if args.eval_split > 0 else None,
        data_collator = data_collator,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    model.save_pretrained(args.output_dir + "/lora")
    tokenizer.save_pretrained(args.output_dir + "/lora")
