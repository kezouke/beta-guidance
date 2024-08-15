import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..utils.early_stopping import EosListStoppingCriteria
from ..utils.substring_utils.substring_engine import SubstringEngine
from transformers import PreTrainedModel
from vllm import LLM


class GuidanceBeta:
    """
    Class for generating guidance using a 
    pretrained language model.

    Args:
        model Option[PreTrainedModel, LLM]: Either pretrained model 
            from Hugging Face model hub or LLM model from vllm library
        mode (bool): Mode for the guidance generation (whether to print log messages or not).

    Attributes:
        model (AutoModelForCausalLM): Pretrained model for generating guidance.
        tokenizer (AutoTokenizer): Tokenizer for tokenizing inputs.
    """

    def __init__(self, llm, mode=True, is_fast=True):

        self.llm = llm
        self.llm_engine = self.llm.engine
        if hasattr(self.llm_engine, "get_tokenizer"):
            self.tokenizer = self.llm_engine.get_tokenizer()
        elif hasattr(self.llm_engine, "tokenizer"):
            if hasattr(self.llm_engine.tokenizer, "tokenizer"):
                self.tokenizer = self.llm_engine.tokenizer.tokenizer
            else:
                self.tokenizer = self.llm_engine.tokenizer
        else:
            raise ValueError(
                "The provided LLM instance in RegexLogitsProcessor neither has a "
                "`tokenizer` attribute or a get_tokenizer method.")

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.mode = mode
        self.is_fast = is_fast

    def _tokenize_inputs(self, texts, choices_list):
        inputs = []
        for text, choices in zip(texts, choices_list):
            for choice in choices:
                inputs.append(f"{text}{choice}")

        tokenized_inputs = self.tokenizer(inputs,
                                          return_tensors="pt",
                                          padding=True,
                                          add_special_tokens=True)

        return tokenized_inputs

    def select(self, input_batches, choices_list, return_full_text=False):
        """
        Select the most appropriate choice for each text.

        Args:
            texts (list): List of input texts or one string with input.
            choices_list (list of lists): List of lists of choices corresponding to each text.

        Returns:
            list: List of selected choices.
        """
        if isinstance(input_batches, str):
            input_batches = [input_batches]

        tokenized_inputs = self._tokenize_inputs(input_batches, choices_list)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            logits = outputs.logits

            # Apply log softmax to convert logits to probabilities
            probabilities = F.log_softmax(logits, dim=-1)

        returned_text = []

        logits_slice_begin = 0

        for text_idx, pair in enumerate(zip(input_batches, choices_list)):
            text, choices = pair

            # Number of tokens to skip, since they are common in given text
            skip_logits = len(self.tokenizer.encode(text)) - 1

            # Get the number of different variants to select
            number_of_options = len(choices)

            logits_slice_end = logits_slice_begin + number_of_options

            # Extracting logits for specific tokens
            probabilities_slice = probabilities[
                logits_slice_begin:logits_slice_end, skip_logits - 1:-1, :]

            # Getting indices of tokens from input_ids
            input_ids_slice = tokenized_inputs['input_ids'][
                logits_slice_begin:logits_slice_end, skip_logits:]

            # Create a mask tensor in order not to count probability of special tokens
            mask = torch.where(
                torch.isin(input_ids_slice, self.special_tokens),
                torch.tensor(0), torch.tensor(1))

            # Adding a dimension to input_ids_slice
            input_ids_slice_expanded = input_ids_slice.unsqueeze(-1)

            # Gathering logits for the specified tokens
            selected_probabilities = probabilities_slice.gather(
                dim=-1, index=input_ids_slice_expanded).squeeze(-1)
            selected_probabilities_masked = selected_probabilities * mask

            # Getting log probabilities of
            log_probs = torch.mean(selected_probabilities_masked, dim=-1)
            choice_idx = torch.argmax(log_probs).item()

            if return_full_text:
                returned_text.append(f"{text}{choices[choice_idx]}")
            else:
                returned_text.append(f"{choices[choice_idx]}")

            logits_slice_begin = logits_slice_end

        return returned_text

    def gen(self,
            input_batches,
            stop_keywords=None,
            return_full_text=False,
            **kwargs):
        """
        Generate text based on input batches.

        Args:
            input_batches (str or list of str): Input text or list of input texts.
            stop_keywords (str, optional): String at which to stop generation.
            return_full_text (bool, optional): Whether to return the full generated text.
            **kwargs: Additional keyword arguments for the generation method.
                details on kwargs:
                https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig

        Returns:
            list of str: List of generated texts.
        """

        self.model.eval()

        # Ensure input_batches is a list
        if isinstance(input_batches, str):
            input_batches = [input_batches]

        # Tokenize all input batches at once
        inputs = self.tokenizer(input_batches,
                                return_tensors="pt",
                                padding=True,
                                add_special_tokens=True)

        inputs['input_ids'] = inputs['input_ids'].to('cuda')

        # Convert the stop_token to its token ID if provided
        if stop_keywords:
            keyword_token_ids = self.tokenizer.encode(stop_keywords,
                                                      add_special_tokens=False)
            stopping_criteria = EosListStoppingCriteria(
                eos_sequence=keyword_token_ids)
            kwargs['stopping_criteria'] = [stopping_criteria]

        with torch.no_grad():
            # Generate text for all input batches in a single call
            generated_text = self.model.generate(inputs['input_ids'], **kwargs)
            # Decode the generated text
            generated_texts = self.tokenizer.batch_decode(
                generated_text, skip_special_tokens=True)

        if not return_full_text:
            return [
                result[len(batch):]
                for result, batch in zip(generated_texts, input_batches)
            ]
        return generated_texts

    def substring(self,
                  input_text,
                  context: str,
                  k=1,
                  max_substring_length=35):
        substring_engine = SubstringEngine(self.model,
                                           self.tokenizer,
                                           mode=self.mode)
        # return res_text, working_list, initial_root
        result = substring_engine.substring(input_text, context, k,
                                            max_substring_length)

        return result
