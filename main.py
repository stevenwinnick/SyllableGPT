from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
from collections import defaultdict
import json

class SyllableGPTModel():

    def __init__(self):
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log(f"Using device: {self.torch_device}\n")
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.eos_token_id = self.gpt2_tokenizer.eos_token_id
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self.eos_token_id).to(self.torch_device)
        self.rhymenet = json.load(open('RhymeNetV1.0'))
        self.softmaxer = torch.nn.Softmax(dim=0)
    
    def log(self, message):
        print(message, end="", flush=True)

    def get_prompt(self, input=None):
        if input:
            return input
        else:
            return "On the next line is a poem with a dense internal rhyme scheme.\n"

    def get_start_of_sequence(self, input=None):
        if input:
            return input
        else:
            return ""

    def get_likely_vocabulary(self, tokenized_input, number_beams=10, max_beam_length=20, tokens_per_beam=10):
        self.log("Determining likely vocabulary")
        likely_vocabulary = defaultdict(int)
        outputs = self.gpt2_model.generate(
            input_ids=tokenized_input,
            max_new_tokens=max_beam_length,
            num_beams=number_beams,
            num_return_sequences=number_beams,
            early_stopping=True,
            no_repeat_ngram_size=5
        )
        for beam in outputs:
            for end in range(len(tokenized_input[0]), len(beam)):
                input = beam[:end]
                output = self.gpt2_model(input)
                next_token_logits = output.logits[-1]
                next_tokens_best = np.argsort(next_token_logits.detach())
                next_tokens_best = next_tokens_best[-tokens_per_beam:]
                next_words_best = self.gpt2_tokenizer.convert_ids_to_tokens(next_tokens_best, skip_special_tokens=True)
                for word in next_words_best:
                    if word[0] == "Ġ":
                        likely_vocabulary[word[1:]] += 1
                    else:
                        likely_vocabulary[word] += 1
            self.log(".")
        self.log("\n")
        return set(likely_vocabulary)
    
    def _rescore(self, tokens, logits, prior_tokens, likely_vocabulary):
        """
        Returns the softmax probability of each word after rescoring it based on its matches with the likely vocabulary
        """
        
        prior_words = self.gpt2_tokenizer.convert_ids_to_tokens(prior_tokens)
        next_words = self.gpt2_tokenizer.convert_ids_to_tokens(tokens)
        prior_words = set([word[1:].upper() if word[0] == "Ġ" else word.upper() for word in prior_words])
        next_words = [word[1:].upper() if word[0] == "Ġ" else word.upper() for word in next_words]
        
        for idx, word in enumerate(next_words):
            
            # Add 1 point for each word with a matching syllable in likely vocabulary and 2 for prior words
            try:
                syllables = self.rhymenet["words"][word]["phoneme_syllables"]
                for syllable in syllables:
                    syllable_combined = "".join([phoneme[:-1] + "_" if phoneme[-1].isnumeric() else phoneme + "_" for phoneme in syllable])
                    for matching_word in self.rhymenet["syllables"][syllable_combined]:
                        if matching_word in likely_vocabulary:
                            logits[idx] += 1
                        if matching_word in prior_words:
                            logits[idx] += 2
            except:
                pass
            
            # Add 2 points for each word with a matching last syllable rhyme in likely vocabulary and 4 for prior words
            try:
                last_syllable = self.rhymenet["words"][word]["phoneme_syllables"][-1]
                last_syllable_combined = "".join([phoneme[:-1] + "_" if phoneme[-1].isnumeric() else phoneme + "_" for phoneme in last_syllable])
                for matching_word in self.rhymenet["syllables"][last_syllable_combined]:
                    if matching_word in likely_vocabulary:
                        logits[idx] += 2
                    if matching_word in prior_words:
                        logits[idx] += 4
            except:
                pass

            # Add 3 points for each word with a matching after stress rhyme in likely vocabulary and 6 for prior words
            try:
                after_stress_rhyme = self.rhymenet["words"][word]["after_stress_rhyme"]
                for matching_word in self.rhymenet["after_stress_rhymes"][after_stress_rhyme]:
                    if matching_word in likely_vocabulary:
                        logits[idx] += 3
                    if matching_word in prior_words:
                        logits[idx] += 6
            except:
                pass
            
        softmaxed_logits = self.softmaxer(torch.tensor(logits))
        return list(zip(tokens, softmaxed_logits))

    def generate(self, tokenized_input, likely_vocabulary, number_beams=10, max_beam_length=50, tokens_per_beam=15):
        """
        Generate new text using beam search, with text rescored to increase rhyming
        """
        self.log("Generating text")
        cur_beams = [(tokenized_input[0], 1)]
        next_beams = []
        for iteration in range(max_beam_length):
            for beam_tokens, beam_probability in cur_beams:
                if beam_tokens[-1] == self.eos_token_id:
                    next_beams.append((beam_tokens, beam_probability))
                else:
                    output = self.gpt2_model(beam_tokens)
                    next_token_logits = output.logits[-1]
                    next_tokens_best = np.argsort(next_token_logits.detach())[-tokens_per_beam:]
                    next_tokens_best_logits = [next_token_logits[idx].item() for idx in next_tokens_best]
                    next_tokens_best_rescored = self._rescore(next_tokens_best, next_tokens_best_logits, beam_tokens, likely_vocabulary)
                    next_tokens_best_rescored.sort(key=lambda x: x[1])
                    for token, probability in next_tokens_best_rescored:
                        next_beams += [(torch.cat((beam_tokens, torch.tensor([token])), dim=0), beam_probability * probability)]
            cur_beams = sorted(next_beams, key=lambda x: x[1])[:number_beams]
            next_beams = []
            self.log(".")
        max_prob = 0
        max_idx = -1
        for idx, beam in enumerate(cur_beams):
            if beam[1] >= max_prob:
                max_prob = beam[1]
                max_idx = idx
        self.log("\n")
        return self.gpt2_tokenizer.decode(cur_beams[max_idx][0])

    def interactive(self):
        input_prompt = input("Enter prompt (optional): ")
        prompt = self.get_prompt(input_prompt)
        start_of_sequence = self.get_start_of_sequence(input("Enter start of sequence (optional): "))
        full_input = prompt + start_of_sequence
        tokenized_input = self.gpt2_tokenizer.encode(full_input, return_tensors='pt')
        likely_vocabulary = self.get_likely_vocabulary(tokenized_input)
        generated_output = self.generate(tokenized_input, likely_vocabulary)
        print("Generated output:\n" + 100 * '-')
        print(generated_output)
        

def main():
    model = SyllableGPTModel()
    model.interactive()

if __name__ == "__main__":
    main()