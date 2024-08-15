# !pip install flash-attn --no-build-isolation

# In[8]:

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from typing import List, Tuple
import time
import re
import torch.nn.functional as F
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import logging
from aiostream import stream

logging.basicConfig(level=logging.WARNING)

# In[9]:


class Node:
    """
    A class representing a node in a tree structure. Each node contains information about its token ID, its parent node,
    its children nodes, its depth in the tree, and its cumulative log probability.

    Attributes:
        token_id (int): The ID of the token associated with this node.
        parent_node (Node): The parent node of this node. None if this is the root node.
        children (list): A list of child nodes.
        depth (int): The depth of this node in the tree.
        cum_log_probability (float): The cumulative log probability of this node.
        token_sequence (torch.Tensor): A tensor representing the sequence of tokens from the root to this node.
    """

    def __init__(self, token_id: int, parent_node: 'Node', depth: int):
        """
        Initializes a new Node instance.

        Args:
            token_id (int): The ID of the token associated with this node.
            parent_node (Node): The parent node of this node. None if this is the root node.
            depth (int): The depth of this node in the tree.
        """
        self.token_id = token_id
        self.parent_node = parent_node
        self.children = []
        self.depth = depth
        self.cum_log_probability = None

        # Initialize the token_sequence based on the parent node's token_sequence and the current token_id
        if depth:
            self.token_sequence = torch.cat((parent_node.token_sequence,
                                             torch.tensor([self.token_id],
                                                          dtype=torch.long)))
        else:
            self.token_sequence = torch.tensor([], dtype=torch.long)

    def __str__(self) -> str:
        """
        Returns a string representation of the node, including its token sequence.

        Returns:
            str: A string representation of the node.
        """
        return f"Nodes: {self.token_sequence}, {self.cum_log_probability}"

    def __eq__(self, other) -> bool:
        """
        Checks if this node is equal to another node or a tensor.

        Args:
            other (Node or torch.Tensor): The other node or tensor to compare with.

        Returns:
            bool: True if the nodes are equal, False otherwise.
        """
        if isinstance(other, Node):
            return torch.equal(self.token_sequence, other.token_sequence)
        return False

    def __hash__(self):
        """
        Return the hash based on an immutable attribute. Here, we use the string representation of the token_sequence
        because tensors themselves are not hashable and should not be used directly in hash computations if their content
        may change.
        """
        return hash(tuple(self.token_sequence.tolist()))


# In[10]:

from vllm.sampling_params import SamplingParams


class SubstringEngine:

    def __init__(self, llm, tokenizer, mode=False):
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_token_len = min(
            len(
                sorted(list(self.tokenizer.vocab.keys()),
                       key=lambda x: len(x),
                       reverse=True)[0]), 16)
        self.mode = mode

    def _expand_tree(self,
                     parent: Node,
                     tokenized_candidates: List[torch.Tensor],
                     height: int,
                     position: int = 0,
                     special_ids: List[int] = []) -> Node:
        """
        Expands the tree from a given parent node by adding child nodes based on the tokenized context.
    
        Args:
            parent (Node): The parent node from which to expand the tree.
            tokenized_candidates (List[torch.Tensor]): The tokenized context for the prompt.
            height (int): The height of the tree to expand to.
            position (int, optional): The current position in the tokenized context. Defaults to 0.
            special_ids (List[int], optional): A list of special token IDs to exclude from the tree. Defaults to an empty list.
    
        Returns:
            Node: The parent node with its children expanded.
        """
        # Iterate over each context in the tokenized context
        for candidate in tokenized_candidates:
            # Get the token at the current position
            if position < len(candidate):
                token = candidate[position].item()
                # Check if the token is not a special token and if it's not already a child of the parent
                if (torch.equal(candidate[:position], parent.token_sequence)
                        and all(token != child.token_id
                                for child in parent.children)
                        and token not in special_ids):

                    # Create a new node with the current token and add it as a child to the parent
                    new_node = Node(token, parent, parent.depth + 1)
                    parent.children.append(new_node)

                    # Recursively expand the tree if the current position is less than the height
                    if new_node.depth < height:
                        self._expand_tree(new_node, tokenized_candidates,
                                          height, position + 1, special_ids)
        # Return the parent node with its children expanded
        return parent

    def _build_tree(
            self, tokenized_context: List[torch.Tensor]
    ) -> Tuple[Node, torch.Tensor]:
        """
        Builds the entire tree for a given prompt using the tokenized context.
    
        Args:
            promt (str): The prompt for which the tree is being built.
            tokenized_context (List[torch.Tensor]): The tokenized context for the prompt.    
        Returns:
            Tuple[Node, torch.Tensor]: The root node of the tree and the tokenized prompt.
        """

        s = time.time()

        # Initialize the root node and tokenize the prompt
        root = Node(-1, None, 0)
        # Expand the tree from the root node to the specified height, excluding special tokens
        root = self._expand_tree(root,
                                 tokenized_context,
                                 len(tokenized_context[0]),
                                 special_ids=self.tokenizer.all_special_ids)
        # Set the cumulative log probability of the root node to 0
        root.cum_log_probability = 0

        if self.mode:
            print(f"build tree for first tokens: {time.time() - s}")

        # Return the root node and the tokenized prompt
        return root

    def _candidate_sequences(self, context, max_token_length, prompt=''):
        """
        Generates a set of candidate sequences from the given context by considering all
        possible substrings within a specified length limit.
        
        These candidates are then prefixed with the provided prompt to form complete sequences.
    
        Args:
            context (str): The input context from which to generate candidate sequences.
            max_token_length (int): The maximum length of a candidate sequence in terms of tokens.
            prompt (str, optional): A prefix to be added to each candidate sequence. Defaults to an empty string.
    
        Returns:
            list: A list of candidate sequences, each starting with the provided prompt.
    
        The function first calculates the restriction based on the length of the context and the maximum token length.
        It then iterates over the text to generate all possible substrings within this restriction.
        These substrings are added to a set to ensure uniqueness. The set is then sorted for reproducibility,
        and each candidate is prefixed with the prompt to form complete
        sequences. These sequences are returned as a list.
        """
        s = time.time()

        # Calculate the restriction based on the length of the text and the maximum token length
        restriction = min(len(context) + 1, max_token_length)
        # Initialize an empty set to store unique substring candidates
        substring_candidates = set()
        # Iterate over the text to generate all possible substrings within the restriction
        for i in range(len(context)):
            for j in range(i + 1, i + 1 + restriction):
                substring_candidates.add(context[i:j])

        # Sort the set of substring candidates for reproducibility
        substring_candidates = sorted(substring_candidates)
        # Prefix each candidate with the prompt to form complete sequences
        sequences = [prompt + candidate for candidate in substring_candidates]

        if self.mode:
            print(
                f"get candidates: last 2 tokens + all substring candidates {time.time() - s}"
            )

        return sequences

    async def _compute_logprob(self, common_part, nodes):
        """
        Computes the cumulative log probabilities for each node in the tree structure,
        given a common part of the text (user prompt wihtout last 2 tokens) and a list of nodes.
    
        This function is crucial for evaluating the likelihood of each candidate sequence generated from the context text.
        It does so by leveraging the transformer model to predict the next token in the sequence and then calculating
        the log probability of each token. 
        
        Args:
            common_part (str): A common part of the text (user prompt wihtout last 2 tokens) that is shared by all
                               nodes in the tree. This is used to ensure that the model's predictions are relevant
                               to the context of the input prompt.
            nodes (List[Node]): A list of nodes for which the cumulative log probabilities are to be computed.
    
        Returns:
            None: The function modifies the nodes in-place, updating their cumulative log probabilities.
    
        The function begins by initializing an empty list for the input batch and two empty lists for mapping nodes
        to their corresponding log probabilities and input sequences. It then iterates over each node, checking if
        its cumulative log probability has been set. If not, it constructs the input sequence for the model by 
        concatenating the common part of the text with the token sequence of the node. This input sequence is then
        added to the input batch and the node is mapped to its corresponding log probability.
    
        Once all nodes have been processed, the function tokenizes the input batch using the tokenizer and feeds
        it into the model to get the logits. The log probabilities are then calculated using the log_softmax function.
    
        Finally, the function iterates over the nodes again, this time updating their cumulative log probabilities
        based on the log probabilities of their tokens and the cumulative log probabilities of their parent nodes.
        
        This process ensures that each node's cumulative log probability reflects the likelihood
        of the sequence of tokens leading up to it.
        """

        s = time.time()

        # Calculate the number of tokens in the common part of the text
        loop = asyncio.get_running_loop()
        skip_logits = len(self.tokenizer.encode(common_part))

        log_probs_mapping = {}
        node_output_map = {}
        nodes_map = []
        results = None
        sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)

        nodes_map = [
            node for node in nodes if node.cum_log_probability is None
        ]

        if self.mode:
            print("log prob nodes: ")
            print(len(nodes))
            for node in nodes_map:
                print(self.tokenizer.decode(node.token_sequence))
            print()

        if not nodes_map:
            return

        # # Iterate over each node
        # # for node in nodes:
        for node in nodes_map:
            inp = self.tokenizer.encode(
                common_part) + node.token_sequence.tolist()
            request_id = random_uuid()
            # Feed the input into the model to get the logits
            res = (engine.generate(request_id=request_id,
                                   prompt=None,
                                   prompt_token_ids=inp,
                                   sampling_params=sampling_params))

            node_output_map[request_id] = [node]

            if not results:
                results = res
            else:
                results = stream.merge(results, res)

        async for item in results:
            node_output_map[item.request_id].append(item)

        # Iterate over the nodes again to update their cumulative log probabilities
        for idx in node_output_map:
            node = node_output_map[idx][0]
            output = node_output_map[idx][1]

            # It is possible, that parent node cum_log_probability is not calculated yet
            # Thus, if it is a such situation, we will calculate log_probability for parent nodes also

            # Initialize lists for the parent nodes and their log probabilities
            parents_log_probs = []
            parents_sequence_without_logprob = []
            # Iterate over the parent nodes
            parent_tmp = node.parent_node

            i = 2
            while parent_tmp.cum_log_probability is None:
                parents_sequence_without_logprob.append(parent_tmp)

                if self.mode:
                    print("parent without logprob: ")
                    print(parent_tmp.token_id)
                    print(output.prompt_logprobs[-i])

                parents_log_probs.append(
                    output.prompt_logprobs[-i][parent_tmp.token_id].logprob)
                i += 1
                parent_tmp = parent_tmp.parent_node

            # Calculate the cumulative log probability for each parent node
            number_of_parents_without_logbrob = len(
                parents_sequence_without_logprob)
            for n_id in range(number_of_parents_without_logbrob - 1, -1, -1):
                if n_id == number_of_parents_without_logbrob - 1:
                    parents_sequence_without_logprob[
                        n_id].cum_log_probability = (
                            parents_log_probs[n_id] +
                            parents_sequence_without_logprob[n_id].parent_node.
                            cum_log_probability)
                else:
                    parents_sequence_without_logprob[
                        n_id].cum_log_probability = (
                            parents_log_probs[n_id] +
                            parents_sequence_without_logprob[n_id].parent_node.
                            cum_log_probability)

            # Update the node's cumulative log probability
            log_probs_mapping[node] = node.parent_node.cum_log_probability

            node.cum_log_probability = log_probs_mapping[
                node] + output.prompt_logprobs[-1][node.token_id].logprob

        if self.mode:
            print(f"compute log_probs call {time.time() - s}")

    async def _get_topk_nodes(self, nodes, k):
        """
        Selects the top `k` nodes from a given list of nodes based on their cumulative log probabilities, normalized by their depth.
    
        This function is used to prune the tree and focus on the most promising candidates for further evaluation or output.
        By normalizing the cumulative log probabilities by the depth of each node,
        it ensures that nodes deeper in the tree are not overly favored simply because they are longer.
    
        Args:
            nodes (List[Node]): A list of nodes from which to select the top `k` nodes.
            k (int): The number of top nodes to select.
    
        Returns:
            List[Node]: A list of the top `k` nodes, sorted by their normalized cumulative log probabilities.
    
        The function begins by calculating the normalized scores for each node.
        This is done by dividing the cumulative log probability of each node by its depth.
        The scores are then converted into a tensor and the indices of the top `k` scores 
        are determined using the `torch.topk` function. These indices are used to select 
        the corresponding nodes from the original list.
    
        The selected nodes are returned as a list, which can then be used for further processing
        or output. This function is particularly useful in scenarios where the tree is large and 
        contains many nodes, allowing the script to efficiently focus on the most likely candidates.
        """
        s = time.time()
        # Calculate the normalized scores for each node
        scores = torch.tensor(
            [node.cum_log_probability / node.depth for node in nodes])
        # Determine the indices of the top k scores
        top_k_indices = torch.topk(scores, k=k, largest=True,
                                   sorted=True).indices

        if self.mode:
            print(f"get top k nodes call: {time.time() - s}")
        # Select the top k nodes using the indices
        return [nodes[i] for i in top_k_indices]

    def _candidate_sequences_exp(self,
                                 context,
                                 chosen_options,
                                 max_candidate_length,
                                 prompt=''):
        """
        Generates a set of candidate sequences from the given context,
        with each candidate starting with one of the chosen options.
    
        This function is useful for scenarios where the context or the prompt suggests
        specific starting points for the sequences, allowing for more targeted generation.
        
        It iterates over the text to generate all possible substrings within a specified length
        limit and checks if each candidate starts with one of the chosen options. Only those 
        candidates that meet this condition are added to the set of substring candidates.
    
        Args:
            context (str): The input context from which to generate candidate sequences.
            chosen_options (List[str]): A list of options that each candidate sequence must start with.
            max_candidate_length (int): The maximum length of a candidate sequence in terms of tokens.
            prompt (str, optional): A prefix to be added to each candidate sequence. Defaults to an empty string.
    
        Returns:
            list: A list of candidate sequences, each starting with one of the chosen options and prefixed with the provided prompt.
    
        The function first calculates the restriction based on the length of the text and the maximum candidate length. It then iterates over the text to generate all possible substrings within this restriction. For each substring, it checks if the substring starts with one of the chosen options. If so, the substring is added to the set of substring candidates. The set is then sorted for reproducibility, and each candidate is prefixed with the prompt to form complete sequences. These sequences are returned as a list.
        """
        s = time.time()
        # Calculate the restriction based on the length of the text and the maximum candidate length
        restriction = min(len(context) + 1, max_candidate_length)
        # Initialize an empty set to store unique substring candidates
        substring_candidates = set()
        # Iterate over the text to generate all possible substrings within the restriction
        for i in range(len(context)):
            for j in range(i + 1, i + 1 + restriction):
                candidate = prompt + context[i:j]
                tokenized_candidate = self.tokenizer.encode(
                    candidate, return_tensors="pt",
                    add_special_tokens=False)[0]
                # Check if the candidate starts with one of the chosen options
                if any(
                        torch.equal(tokenized_candidate[:len(option)], option)
                        for option in chosen_options):
                    substring_candidates.add(candidate)

        # Sort the set of substring candidates for reproducibility
        sequences = sorted(list(substring_candidates))

        if self.mode:
            print(
                f"get expanded candidates starts with top k first tokens: {time.time() - s}"
            )
        return sequences

    def _get_nodes_seq_before_branch(self, node):
        """
        Traverses the tree structure from a given node and returns the node that is just before a branching point.
    
        Args:
            node (Node): The starting node from which to traverse the tree.
    
        Returns:
            Node: The node that is just before a branching point in the tree.
    
        The function begins by entering a loop that continues until it finds a node with more than one child.
        It starts with the given node and checks its children. If the node has exactly one child, the function 
        moves to that child and continues the process. This ensures that the function traverses down the tree 
        until it reaches a node that is about to branch into multiple paths.
    
        Once the branching point is found, the function breaks out of the loop and returns the node that led to 
        this branching. This node is the one just before the branching point, and it can be used for further 
        processing or analysis.
        """
        while True:
            children = node.children
            # If the node has exactly one child, move to that child
            if len(children) == 1:
                node = children[0]
            else:
                # If the node has more than one child, it's a branching point
                break
        # Return the node just before the branching point
        return node

    async def _iteration(self, working_list, common_part, k):
        """
        Performs an iteration of the sequence generation process by computing the cumulative log probabilities
        of the nodes in the working list and selecting the top `k` nodes.
    
        Args:
            working_list (List[Node]): The list of nodes for which the cumulative log probabilities are to be computed
                                       and from which the top `k` nodes are to be selected.
            common_part (str): A common part of the text that is shared by all nodes in the tree. This is used to ensure
                               that the model's predictions are relevant to the context of the input text.
            k (int): The number of top nodes to select.
    
        Returns:
            List[Node]: The top `k` nodes from the working list, sorted by their normalized cumulative log probabilities.
        """
        # Add children to wotking list for every node in it
        working_list = self._update_working_list_with_children(working_list)

        # Compute the cumulative log probabilities for each node in the working list
        await self._compute_logprob(common_part, working_list)
        # Select the top k nodes from the working list based on their cumulative log probabilities
        return await self._get_topk_nodes(working_list, k)

    def _update_working_list_with_children(self, working_list):
        """
        Updates the working list of nodes by adding the children of each node in the list,
        specifically those that are just before a branching point in the tree.
    
        Args:
            working_list (List[Node]): The current working list of nodes to be updated.
    
        Returns:
            List[Node]: The updated working list of nodes, including the children of each node in the original list.
        """
        s = time.time()
        for node in working_list:
            for c in node.children:
                candidate_to_add = self._get_nodes_seq_before_branch(c)
                if candidate_to_add not in working_list:
                    working_list.append(candidate_to_add)

        if self.mode:
            print(
                f"add children sequences before found branch into working list: {time.time() - s}"
            )
        return working_list

    async def substring(self,
                        prompt,
                        context,
                        k,
                        max_substring_length,
                        return_full_text=False):
        """
        Generates and evaluates candidate sequences based on a given prompt and context, selecting the top `k` nodes.
    
        Args:
            prompt (str): The prompt for which the tree is being built and from which the last part and the common part are extracted.
            context (str): The context from which candidate sequences are generated.
            k (int): The number of top nodes to select.
            max_substring_length (int): The maximum length of a candidate sequence in terms of tokens.
    
        Returns:
            Tuple[List[Node], Node]: The top `k` nodes from the working list and the initial root node of the tree.
        """
        # Tokenize the prompt and extract the last part and the common part
        tokenized_prompt = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False)['input_ids']
        last_part = self.tokenizer.decode(tokenized_prompt[0, -2:])
        common_part = self.tokenizer.decode(tokenized_prompt[0, :-2])

        if len(context) == 1:
            if return_full_text:
                return prompt + context
            return context

        # Generate candidate sequences from the context
        substring_candidates = self._candidate_sequences(
            context, self.max_token_len, last_part)
        tokenized_s_cand = self.tokenizer(
            substring_candidates,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False)['input_ids']

        # Build the tree structure from the tokenized candidate sequences
        initial_root = self._build_tree(tokenized_s_cand)

        # Expand the tree from the root node
        node_before_branch = self._get_nodes_seq_before_branch(initial_root)
        first_branch_children = node_before_branch.children

        if not first_branch_children:
            first_branch_children = [node_before_branch]

        # Last token of the prompt can be changed
        # Therefore, we have to capture not tokens before first branch, but all its children after branch

        working_list = []
        for node in first_branch_children:
            wl_len = len(working_list)

            for c in node.children:
                candidate_to_add = self._get_nodes_seq_before_branch(c)
                if candidate_to_add not in working_list:
                    working_list.append(candidate_to_add)

            if wl_len == len(working_list):
                working_list.append(node)

        await self._compute_logprob(common_part, working_list)
        working_list = await self._get_topk_nodes(working_list, k)

        # Generate expanded candidate sequences based on the chosen candidates
        chosen_candidates = list(map(lambda x: x.token_sequence, working_list))

        if self.mode:
            print("chosen_candidates for explansion: ")
            for c in chosen_candidates:
                print(self.tokenizer.decode(c))
            print()

        expanded_candidates = self._candidate_sequences_exp(
            context, chosen_candidates, max_substring_length, last_part)

        tokenized_expanded_candidates = self.tokenizer(
            expanded_candidates,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False)['input_ids']

        # Update the working list with children nodes
        working_list[0].parent_node.children = working_list

        # Expand the tree with the expanded candidate sequences
        for node in working_list:
            self._expand_tree(node,
                              tokenized_expanded_candidates,
                              len(tokenized_expanded_candidates[0]),
                              position=node.depth,
                              special_ids=self.tokenizer.all_special_ids)

        # Iteratively refine the set of candidate sequences based on their likelihood
        while True:
            prev_w_l = deepcopy(working_list)
            working_list = await self._iteration(working_list, common_part, k)
            if prev_w_l == working_list:
                break

        res_text = []
        for node in working_list:
            substirng_choice = self.tokenizer.decode(node.token_sequence)

            if return_full_text:
                res_text.append(common_part + substirng_choice)
            else:
                res_text.append(substirng_choice[len(last_part):])

        return res_text, working_list, initial_root


# In[11]:

from transformers import StoppingCriteria


class EosListStoppingCriteria(StoppingCriteria):

    def __init__(self, eos_sequence, len_tokenized_eos_sequence, tokenizer):
        self.eos_sequence = eos_sequence
        self.len_tokenized_eos_sequence = len_tokenized_eos_sequence
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:

        # Check each batch item if the sequence ends with the specified eos_sequence
        last_ids = self.tokenizer.decode(
            input_ids[:, -self.len_tokenized_eos_sequence:])
        # Check if all elements in eos_sequence match for any item in the batch
        return self.eos_sequence == last_ids


# In[12]:


class GuidanceBeta:
    """
    Class for generating guidance using a pretrained language model.

    Args:
        model_name (str): Pretrained model identifier from Hugging Face model hub.
        mode (bool): Mode for the guidance generation (whether to print log messages or not).
        model_kwargs (dict): Additional keyword arguments to pass to the model initialization.
        tokenizer_kwargs (dict): Additional keyword arguments to pass to the tokenizer initialization.

    Attributes:
        model (AutoModelForCausalLM): Pretrained model for generating guidance.
        tokenizer (AutoTokenizer): Tokenizer for tokenizing inputs.
    """

    def __init__(
        self,
        llm,
        mode=True,
    ):

        self.llm = llm

        if hasattr(llm, "get_tokenizer"):
            self.tokenizer = self.llm.get_tokenizer()
        elif hasattr(llm, "tokenizer"):
            if hasattr(llm.tokenizer, "tokenizer"):
                self.tokenizer = self.llm.tokenizer.tokenizer
            else:
                self.tokenizer = self.llm.tokenizer
        else:
            raise ValueError(
                "The provided LLM instance in RegexLogitsProcessor neither has a "
                "`tokenizer` attribute or a get_tokenizer method.")

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.mode = mode

    async def substring(self,
                        input_text,
                        context: str,
                        k=1,
                        max_substring_length=35):
        substring_engine = SubstringEngine(self.llm,
                                           self.tokenizer,
                                           mode=self.mode)
        # return res_text, working_list, initial_root
        result = await substring_engine.substring(input_text, context, k,
                                                  max_substring_length)

        return result[0]


# In[13]:

# import time

# import os
# import torch
# from vllm import LLM

# model_name = "meta-llama/Meta-Llama-3-8B"
# access_token = "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"

# Specify the primary GPU to use (GPU 0, which has more available memory)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Setting PyTorch environment variable for better memory management
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# llm = LLM(
#     model=model_name,
#     tensor_parallel_size=1,  # Consider adjusting if still facing issues
#     gpu_memory_utilization=0.7,
#     enforce_eager=True
# )

# In[ ]:

# guidance_system = GuidanceBeta(llm, mode=False)

# Сейчас на сабстринге включены принты, чтобы следить за временем каждого этапа алогоритма.
# Если хочешь выключить, то передай в инициализацию GuidanceBeta аргумент `mode=False`

# In[ ]:

# import time

# prompt = "How many parameters does BLOOM have? "
# context = "NL"

# s = time.time()
# res = guidance_system.substring(prompt, context, 3)
# time.time() - s

# In[ ]:

# In[ ]:

# for node in res[1]:
#     print(node)
#     print(guidance_system.tokenizer.decode(node.token_sequence))

#  _______________________________
# / Don't want to self-host?      \
# \ Try .json at http://dottxt.co /
#  -------------------------------
#        \   ^__^
#         \  (oo)\_______
#            (__)\       )\/\
#                ||----w |
#                ||     ||
#
#
# Copyright 2024- the Outlines developers
# Copyright 2023 the vLLM developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
from typing import AsyncGenerator, Optional
import outlines
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from pydantic import BaseModel

from outlines.integrations.vllm import JSONLogitsProcessor, RegexLogitsProcessor

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None

# class GenerationItem(BaseModel):
#     prompt: str
#     substring: Optional[str] = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/logits")
async def logits(request: Request) -> Response:
    """
    Endpoint used to test that logit generation works in vLLM
    """
    assert engine is not None

    return JSONResponce({'text': ''})


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - schema: the JSON schema to use for the generation (if regex is not provided).
    - regex: the regex to use for the generation (if schema is not provided).
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    assert engine is not None

    request_dict = await request.json()
    prompt = request_dict.pop('prompt', None)
    stream = request_dict.pop("stream", False)
    json_schema = request_dict.pop("schema", None)
    regex_string = request_dict.pop("regex", None)
    choice_list = request_dict.pop("choice", None)
    temperature = request_dict.pop("temperature", 1)
    stop = request_dict.pop("stop", None)
    substring = request_dict.pop('substring', None)
    if substring is None:
        if json_schema is not None:
            logits_processors = [
                JSONLogitsProcessor(json_schema, engine.engine)
            ]
        elif regex_string is not None:
            logits_processors = [
                RegexLogitsProcessor(regex_string, engine.engine)
            ]
        elif choice_list is not None:
            regex_str = r"(" + r"|".join(choice_list) + r")"
            logits_processors = [
                RegexLogitsProcessor(regex_str, engine.engine)
            ]
        else:
            logits_processors = []
        sampling_params = SamplingParams(
            **request_dict,
            logits_processors=logits_processors,
            stop=stop,
            include_stop_str_in_output=True,
            temperature=temperature  # type: ignore
        )
        request_id = random_uuid()

        results_generator = engine.generate(prompt, sampling_params,
                                            request_id)  # type: ignore

        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for request_output in results_generator:
                prompt = request_output.prompt
                text_outputs = [
                    prompt + output.text for output in request_output.outputs
                ]
                ret = {"text": text_outputs}
                yield (json.dumps(ret) + "\0").encode("utf-8")

        if stream:
            return StreamingResponse(stream_results())

        # Non-streaming case
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await engine.abort(request_id)  # type: ignore
                return Response(status_code=499)
            final_output = request_output

        assert final_output is not None
        prompt = final_output.prompt
        text_outputs = [output.text for output in final_output.outputs]
        return JSONResponse(text_outputs)
    else:
        res = await guidance_system.substring(prompt, substring, 1)
        return JSONResponse(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Adds the `engine_use_ray`,  `disable_log_requests` and `max_log_len`
    # arguments
    engine_args: AsyncEngineArgs = AsyncEngineArgs.from_cli_args(
        args)  # type: ignore

    # Sets default for the model (`facebook/opt-125m`)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    guidance_system = GuidanceBeta(engine.engine, mode=True)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
