import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from run_pplm_re import run_pplm_example
from collections import Counter
from nltk.util import ngrams
from transformers import pipeline
def compute_ppl(model, tokenizer, text, stride=512):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


# Example text

# Generate text using your PPLM model
generated_texts = run_pplm_example(
    cond_text="The potato",
    num_samples=20,
    bag_of_words='science',
    discrim="sentiment",
    class_label="very_positive",
    length=50,
    stepsize=0.05,
    sample=True,
    num_iterations=10,
    # window_length=0,
    gamma=1.0,
    gm_scale=0.9,
    kl_scale=0.02,
    colorama=True,
    verbosity='quiet'
)
# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# Assuming the function returns a list of generated texts
for i, text in enumerate(generated_texts):
    ppl = compute_ppl(model, tokenizer, text)
    sentiment_result = classifier(text)[0]
    print(f"Perplexity of generated text {i + 1}: {ppl}，Sentiment of generated text {i + 1}: {sentiment_result['label']} (Score: {sentiment_result['score']:.2f})")


# Compute perplexity


def calculate_distinct_ngrams(samples, n):
    all_ngrams = [ngram for sample in samples for ngram in ngrams(sample.split(), n)]
    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)

#generated_texts 是一个包含生成文本样本的列表
dist_1 = calculate_distinct_ngrams(generated_texts, 1)
dist_2 = calculate_distinct_ngrams(generated_texts, 2)
dist_3 = calculate_distinct_ngrams(generated_texts, 3)

print(f"Dist-1: {dist_1}")
print(f"Dist-2: {dist_2}")
print(f"Dist-3: {dist_3}")


sentiment_results = classifier(generated_texts)
# 加载预训练的情感分类器
positive_samples = [text for text, result in zip(generated_texts, sentiment_results) if result['label'] == 'POSITIVE']
negative_samples = [text for text, result in zip(generated_texts, sentiment_results) if result['label'] == 'NEGATIVE']

print(f"Number of positive samples: {len(positive_samples)}")
print(f"Number of negative samples: {len(negative_samples)}")
