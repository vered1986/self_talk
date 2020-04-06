"""
Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
"""
import re
import torch
import logging

from typing import List
from collections import defaultdict
from transformers import GPT2Tokenizer, XLNetTokenizer, TransfoXLTokenizer, OpenAIGPTTokenizer
from transformers import GPT2LMHeadModel, XLNetLMHeadModel, TransfoXLLMHeadModel, OpenAIGPTLMHeadModel

logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

logger = logging.getLogger(__name__)

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

MODEL_CLASSES = {
    'distilgpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-medium': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-large': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-xl': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet-base-cased': (XLNetLMHeadModel, XLNetTokenizer),
    'xlnet-large-cased': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl-wt103': (TransfoXLLMHeadModel, TransfoXLTokenizer)
}


class LMTextGenerator:
    """
    Generating text with a language model using the HuggingFace implementation.
    """
    def __init__(self,
                 model_name: str,
                 device: torch.device = torch.device("cpu")) -> None:

        logger.info("Loading the language model")
        self.model_name = model_name
        self.lm_head, self.tokenizer = init_model(model_name, device)
        self.device = device

    def generate(self,
                 prefixes: List[str],
                 p: float = 0.0,
                 k: float = 0.0,
                 temperature: float = 1.0,
                 length: int = 25,
                 num_samples: int = 1,
                 stop_token=None):
        """
        Generate an ending for the beginning of the text
        :param prefixes: text on which the generation is conditioned
        :param p: p for nucleus sampling
        :param k: k for top k sampling
        :param temperature: default = 1
        :param length: the maximum length to sample
        :param num_samples: how many texts to generate at once
        :param stop_token: if this token was generated, it's the end of the generated text.
        :return: the text
        """
        if "transfo-xl" in self.model_name or "xlnet" in self.model_name:
            prefixes = [PADDING_TEXT + prefix for prefix in prefixes]

        generated_strings = defaultdict(list)
        reduce_spaces = lambda s: ' '.join(s.split())

        for index, prefix in enumerate(prefixes):
            out = self.generate_texts(
                prompt_text=prefix, length=length, temperature=temperature,
                k=k, p=p, num_samples=num_samples, stop_token=stop_token)

            generated_strings[index] = [reduce_spaces(t) for t in out]

        return generated_strings

    def generate_texts(self,
                       length: int,
                       prompt_text: str,
                       num_samples: int = 1,
                       temperature: float = 1.0,
                       p: float = 0.0,
                       k: float = 0.0,
                       stop_token='?'):
        """
        Generate an ending for the beginning of the text
        :param prompt_text: text on which the generation is conditioned
        :param p: p for nucleus sampling
        :param temperature: default = 1
        :param length: the maximum length to sample
        :return: the text
        """
        eos_token_ids = self.tokenizer.encode(f"{stop_token} <eop> <eod>", add_special_tokens=False)

        if "xlnet" in self.model_name and len(eos_token_ids) > 1:
            eos_token_ids = eos_token_ids[1:]

        k = k if k > 0 else None
        p = p if p > 0 else None

        context_tokens = self.tokenizer.encode(prompt_text)
        max_length = length + len(context_tokens)
        input_ids = torch.tensor(context_tokens, device=self.device).unsqueeze(0)

        outputs = self.lm_head.generate(
            input_ids=input_ids, max_length=max_length, do_sample=True, temperature=temperature,
            num_return_sequences=num_samples, top_p=p, top_k=k, eos_token_ids=eos_token_ids, repetition_penalty=2.0)

        if len(outputs.shape) == 3:
            outputs = outputs[0]

        outputs = outputs[:, len(context_tokens):]
        outputs = [self.tokenizer.decode(text, clean_up_tokenization_spaces=True) for text in outputs]

        if stop_token is not None:
            outputs = [text[:text.find(stop_token)+1] for text in outputs if stop_token in text]

        outputs = [re.sub(" +", " ", text).strip() for text in outputs]
        outputs = set([text for text in outputs if len(text) > 0])
        return outputs


def init_model(model_name: str,
               device: str):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    logger.info(f'Initializing {model_name}')
    model_class, tokenizer_class = MODEL_CLASSES[model_name]
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer






