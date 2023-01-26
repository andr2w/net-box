import torch 
from transformers import BertTokenizer, BertLMHeadModel

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertLMHeadModel.from_pretrained('bert-base-uncased')

    input_ids = tokenizer.encode("How are you", return_tensors='pt')

    beam_output = model.generate(input_ids,
                                max_length=50,
                                num_beams=5,
                                no_repeat_ngram_size=2,
                                early_stopping=True)
    
    results = tokenizer.decode(beam_output[0], skip_special_tokens=True)

    print(results)

if __name__ == '__main__':
    main()