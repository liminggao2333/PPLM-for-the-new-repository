import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Example text
from run_pplm_re import run_pplm_example

# Generate text using your PPLM model
generated_texts = run_pplm_example(
    cond_text="The potato",
    num_samples=3,
    bag_of_words='military',
    discrim="sentiment",
    class_label="very_negative",
    length=50,
    stepsize=0.03,
    sample=True,
    num_iterations=3,
    window_length=5,
    gamma=1.5,
    gm_scale=0.95,
    kl_scale=0.1,
    colorama=True,
    verbosity='quiet'
)


# Assuming the function returns a list of generated texts
for i, text in enumerate(generated_texts):
    ppl = compute_ppl(model, tokenizer, text)
    print(f"Perplexity of generated text {i + 1}: {ppl}")


# Compute perplexity