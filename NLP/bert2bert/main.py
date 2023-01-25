import datasets
import pandas as pd
from datasets import ClassLabel
from transformers import BertTokenizer, EncoderDecoderModel



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

    bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
    '''
    bert2bert.encoder is an instance of BertModel
    bert2bert.decoder is one of BertLMHeadModel.
    Both instances are now combined into a single torch.nn.Module
    and can thus be saved as a single .pt checkpoint file.
    '''
    print(bert2bert.config)

    '''
    Output of bert2bert.config

    '''








    




if __name__ == '__main__':
    main()