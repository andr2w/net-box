import datasets
import pandas as pd
from datasets import ClassLabel
from transformers import BertTokenizer



def main():
    train_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train")
    df = pd.DataFrame(train_data[:1])
    del df["id"]
    tokenizer = BertTokenizer.from_pretrained("bert-base_uncased")
    sample_size = 10000
    
    def map_to_length(x):
    
        x["article_len"] = len(tokenizer(x["article"])).input_ids
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

    


if __name__ == '__main__':
    main()