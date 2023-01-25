import torch
import datasets
import pandas as pd
from datasets import ClassLabel
from transformers import BertTokenizer, EncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments



def main():
    train_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train")
    df = pd.DataFrame(train_data[:1])
    del df["id"]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sample_size = 10000
    
    def map_to_length(x):
    
        x["article_len"] = len(tokenizer(x["article"]).input_ids)
        x["article_longer_512"] = int(x["article_len"] > 512)
        x["summary_len"] = len(tokenizer(x["highlights"]).input_ids)
        x["summary_longer_64"] = int(x["summary_len"] > 64)
        x["summary_longer_128"] = int(x["summary_len"] > 128)
        
        return x 
    
    data_stats = train_data.select(range(sample_size)).map(map_to_length, num_proc=4)


    def compute_and_print_stats(x):
        if len(x["article_len"]) == sample_size:
            print(
                "Article Mean: {}, %-Articles > 512:{}, Summary Mean:{}, %-Summary > 64:{}, %-Summary > 128:{}".format(
                    sum(x["article_len"]) / sample_size,
                    sum(x["article_longer_512"]) / sample_size, 
                    sum(x["summary_len"]) / sample_size,
                    sum(x["summary_longer_64"]) / sample_size,
                    sum(x["summary_longer_128"]) / sample_size,
                )
            )
    
    output = data_stats.map(
        compute_and_print_stats,
        batched=True,
        batch_size=-1
    )

    '''
    Output:
    Article Mean: 805.7411, %-Articles > 512:0.7226, Summary Mean:57.2474, %-Summary > 64:0.2652, %-Summary > 128:0.0

    The average tokens in one article is 805
    3/4 of the articles being longer than the model's max_length 512.
    The summary is on average 57 tokens long. 
    Nearly 30% of our 10000-sample summaries are longer than 64 tokens,
    but none are longer than 128 tokens.


    bert-base-cased is limited to 512 tokens, 
    which means we would have to cut possibly important information from the article.
    Because most of the important imformation is often found at the beginning of articles and because we want to be computationally efficient,
    we decide to stick to bert-base-cased with a max_length of 512 in this 512 in this. 

    Alternatively, one cound leverage long-range sequence models, such as Longformer to be used as the encoder.
    '''

    '''
    Again, we will make use of the .map() function -this time to transform each training batch into a batch of model inputs.

    "article" and "highlights" are tokenized and prepared as the Encoder's "input_ids" and Decoder's "decoder_input_ids" respectively.

    Lastly, it is very important to remember to ignore the loss of the padded labels. In Transformers this can be 
    done by setting the label to -100.    
    '''

    encoder_max_length = 512
    decoder_max_length = 128

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)
        outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because Bert automatically shifts the labels, the labels correspond excatly to `decoder_input_ids`
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

        return batch
    
    '''
    In this notebook, 
    we train and evaluate the model just on a few training examples for demostration 
    and set the batch_size = 4 to prevent out-of-memory issues 
    Only the first 32 examples
    '''
    train_data = train_data.select(range(32))
    batch_size = 4
    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["article", "highlights", "id"]
    )
    
    '''
    OUTPUT: train_data
    -------
    Dataset(
        features: 
        {'attention_mask': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), 
         'input_ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), 
         'labels': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None)
         }, 
        
        num_rows: 32)

    Convert the data to Pytorch Tensors to be trained on GPU.
    '''

    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    '''
    Analogous (相似的)
    We can do the same for the validation data

    10% of the validation dataset and just 8 samples
    '''
    val_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation[:10%]")
    val_data = val_data.select(range(8))

    val_data = val_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=['article', 'highlights', 'id']
    )

    val_data.set_format(
        type='torch', columns=['input_ids', 'attention_mask', 'labels'],
    )

    bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
    '''
    bert2bert.encoder is an instance of BertModel
    bert2bert.decoder is one of BertLMHeadModel.
    Both instances are now combined into a single torch.nn.Module
    and can thus be saved as a single .pt checkpoint file.
    '''
    '''
    Output of bert2bert.config

    EncoderDecoderConfig {
    "_commit_hash": null,
    "decoder": {
        "_name_or_path": "bert-base-uncased",
        "add_cross_attention": true,
        "architectures": [
        "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "bad_words_ids": null,
        "begin_suppress_tokens": null,
        "bos_token_id": null,
        "chunk_size_feed_forward": 0,
        "classifier_dropout": null,
        "cross_attention_hidden_size": null,
        "decoder_start_token_id": null,
        "diversity_penalty": 0.0,
        "do_sample": false,
        "early_stopping": false,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": null,
        "exponential_decay_length_penalty": null,
        "finetuning_task": null,
        "forced_bos_token_id": null,
        "forced_eos_token_id": null,
        "gradient_checkpointing": false,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1"
        },
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "is_decoder": true,
        "is_encoder_decoder": false,
        "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1
        },
        "layer_norm_eps": 1e-12,
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 512,
        "min_length": 0,
        "model_type": "bert",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 12,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 12,
        "num_return_sequences": 1,
        "output_attentions": false,
        "output_hidden_states": false,
        "output_scores": false,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "prefix": null,
        "problem_type": null,
        "pruned_heads": {},
        "remove_invalid_values": false,
        "repetition_penalty": 1.0,
        "return_dict": true,
        "return_dict_in_generate": false,
        "sep_token_id": null,
        "suppress_tokens": null,
        "task_specific_params": null,
        "temperature": 1.0,
        "tf_legacy_loss": false,
        "tie_encoder_decoder": false,
        "tie_word_embeddings": true,
        "tokenizer_class": null,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": null,
        "torchscript": false,
        "transformers_version": "4.25.1",
        "type_vocab_size": 2,
        "typical_p": 1.0,
        "use_bfloat16": false,
        "use_cache": true,
        "vocab_size": 30522
    },
    "encoder": {
        "_name_or_path": "bert-base-uncased",
        "add_cross_attention": false,
        "architectures": [
        "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "bad_words_ids": null,
        "begin_suppress_tokens": null,
        "bos_token_id": null,
        "chunk_size_feed_forward": 0,
        "classifier_dropout": null,
        "cross_attention_hidden_size": null,
        "decoder_start_token_id": null,
        "diversity_penalty": 0.0,
        "do_sample": false,
        "early_stopping": false,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": null,
        "exponential_decay_length_penalty": null,
        "finetuning_task": null,
        "forced_bos_token_id": null,
        "forced_eos_token_id": null,
        "gradient_checkpointing": false,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1"
        },
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "is_decoder": false,
        "is_encoder_decoder": false,
        "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1
        },
        "layer_norm_eps": 1e-12,
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 512,
        "min_length": 0,
        "model_type": "bert",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 12,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 12,
        "num_return_sequences": 1,
        "output_attentions": false,
        "output_hidden_states": false,
        "output_scores": false,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "prefix": null,
        "problem_type": null,
        "pruned_heads": {},
        "remove_invalid_values": false,
        "repetition_penalty": 1.0,
        "return_dict": true,
        "return_dict_in_generate": false,
        "sep_token_id": null,
        "suppress_tokens": null,
        "task_specific_params": null,
        "temperature": 1.0,
        "tf_legacy_loss": false,
        "tie_encoder_decoder": false,
        "tie_word_embeddings": true,
        "tokenizer_class": null,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": null,
        "torchscript": false,
        "transformers_version": "4.25.1",
        "type_vocab_size": 2,
        "typical_p": 1.0,
        "use_bfloat16": false,
        "use_cache": true,
        "vocab_size": 30522
    },
    "is_encoder_decoder": true,
    "model_type": "encoder-decoder",
    "transformers_version": null
}

    The config is similarly composed of an encoder config and a decoder config 
    both of which are instances of BertConfig in our case. 

    We have warm-started a bert2bert model, 
    but we have not defined all the relevant parameters used for beam search decoding yet.

    Let's start by setting the special tokens. Bert-base-cased does not have a 
    decoder_start_token_id or eos_token_id,
    so we will use its cls_token_id and sep_token_id respectively.
    Also, we should define a pad_token_id on the config and make sure the correct vocab_size is set.
    '''

    bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
    bert2bert.config.eos_token_id = tokenizer.sep_token_id
    bert2bert.config.pad_token_id = tokenizer.pad_token_id
    bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size # 30522

    '''
    Next, let's define all parameters related to beam search decoding. 
    '''
    bert2bert.config.max_length = 142
    bert2bert.config.min_length = 56
    bert2bert.config.no_repeat_ngram_size = 3
    bert2bert.config.early_stopping = False
    bert2bert.config.length_penalty = 2.0
    bert2bert.config.num_beams = 4

    '''
    Start fine-tuning the warm-started Bert2Bert model
    One can make use of Seq2SeqTrainer to fine-tune a warm-started encoder-decoder model.

    In short, the seq2seqTrainer allows using the generate(...) function during evaluation. 
    Which is necessary to validate the performance of encoder-decoder models on most sequences-to-sequences tasks,
    such as summarization.

    The argument predict_with_generate should be set to True, so that the Seq2SeqTrainer runs the generate() 
    on the validation data and passes the generated output as predictions to compute_metric function.
    '''

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=False,
        output_dir="./",
        logging_steps=2,
        save_steps=10,
        eval_steps=4)

    '''
    Define a function to correctly compute the ROUGE score during validation.
    Since we activated predict_with_generate, the compute_metrics(...) function expects predictions
    that were obtained using the generate(...) function. Like most summarization tasks.
    '''

    rouge = datasets.load_metric("rouge")

    '''
    The rouge metric computes the score from two lists of strings. Thus we decode both the predictions and labels
    - making sure that -100 is correctly replaced by the pad_token_id and remove all special characters by
    setting skip_sepcial_tokens=True
    '''

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, 
                                     references=label_str, 
                                     rouge_types=['rouge2'])['rouge2'].mid
        
        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4)
        }
    
    trainer = Seq2SeqTrainer(
        model=bert2bert,
        tokenizer=tokenizer,
        args = training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data
    )

    trainer.train()









    




if __name__ == '__main__':
    main()