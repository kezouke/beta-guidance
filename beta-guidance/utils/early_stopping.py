from transformers import StoppingCriteria
import torch


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence):
        self.eos_sequence = eos_sequence

    def __call__(self,
                 input_ids: torch.LongTensor,
                 scores: torch.FloatTensor,
                 **kwargs) -> bool:
        # Check each batch item if the sequence ends with the specified eos_sequence
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        # Check if all elements in eos_sequence match for any item in the batch
        return self.eos_sequence in last_ids
