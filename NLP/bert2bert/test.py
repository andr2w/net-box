import torch 
import datasets
from transformers import BertTokenizer, EncoderDecoderModel

def main():
    bert2bert = EncoderDecoderModel.from_pretrained('./checkpoint-20')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test[:2%]")
    test_data = test_data.select(range(10))

    def generate_summary(batch):
        inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["pred_summary"] = output_str

        return batch
    
    batch_size = 1

    out = generate_summary(test_data[0])
    
    import IPython; IPython.embed(); exit(1)


    results = test_data.map(generate_summary, 
                            batched=True,
                            batch_size=batch_size,
                            remove_columns=["article"])

    # import IPython; IPython.embed(); exit(1)
    print(results)

if __name__ == '__main__':
    main()