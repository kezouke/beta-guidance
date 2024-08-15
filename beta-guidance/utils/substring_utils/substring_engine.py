import torch
import time

from vllm import LLM
from transformers import PreTrainedModel
from utils.tokens_tree_node import Node
from utils.substring_utils.compute_logprobs import hf_compute_logprob, vllm_compute_logprob
from typing import *
from copy import deepcopy


class SubstringEngine:

    def __init__(self, llm, tokenizer, mode=False, is_fast=True):
        self.llm = llm
        self.tokenizer = tokenizer
        longest_token_in_vocab_len = len(
            sorted(list(self.tokenizer.vocab.keys()),
                   key=lambda x: len(x),
                   reverse=True)[0])

        self.max_token_len = min(longest_token_in_vocab_len, 16)
        self.mode = mode
        self.is_fast = is_fast
        self.for_loop_counter = 0

        if isinstance(self.llm, LLM):
            self.compute_logprob_func = vllm_compute_logprob
        elif isinstance(self.llm, PreTrainedModel):
            self.compute_logprob_func = hf_compute_logprob
        else:
            raise Exception("Unsupported llm object passed: {self.llm}")

    def _expand_tree(self,
                     parent: Node,
                     tokenized_candidates: List[torch.Tensor],
                     max_depth: int,
                     position: int = 0,
                     special_ids: List[int] = []) -> Node:
        """
        Expands the tree from a given parent node by adding 
        child nodes based on the tokenized context.
    
        Args:
            parent (Node): The parent node from which to
                expand the tree.
            tokenized_candidates (List[torch.Tensor]): The
                tokenized context for the prompt.
            max_depth (int): Limit on the depth of the 
                constructed tree
            position (int, optional): The current position
                in the tokenized context. Defaults to 0.
            special_ids (List[int], optional): A list of
                special token IDs to exclude from the tree.
                Defaults to an empty list.
    
        Returns:
            Node: The parent node with its children expanded.
        """
        if position >= max_depth:
            return

        tokenized_candidates_to_expand = tokenized_candidates.copy()
        # Iterate over each context in the tokenized context
        self.for_loop_counter += 1
        for idx in range(len(tokenized_candidates) - 1, -1, -1):
            candidate = tokenized_candidates[idx]
            # Get the token at the current position
            token = candidate[position].item()
            # Check if the token is not a special token and if it's
            # not already a child of the parent
            if (torch.equal(candidate[:position], parent.token_sequence)
                    and all(token != child.token_id
                            for child in parent.children)
                    and token not in special_ids):

                # Create a new node with the current token and add it
                # as a child to the parent
                new_node = Node(token, parent, parent.depth + 1)
                parent.children.append(new_node)

                # Recursively expand the tree if the current position
                # is less than the max_depth
                if new_node.depth < max_depth:
                    self._expand_tree(new_node, tokenized_candidates_to_expand,
                                      max_depth, position + 1, special_ids)
            else:
                # Since the parent sequence does not match,
                # this token will definitely not participate
                # in recursive tree expansion calls
                tokenized_candidates_to_expand.pop(idx)

        # Return the parent node with its children expanded
        return parent

    def _build_tree(
            self, tokenized_context: List[torch.Tensor]
    ) -> Tuple[Node, torch.Tensor]:
        """
        Builds the entire tree for a given prompt using 
        the tokenized context.
    
        Args:
            promt (str): The prompt for which the tree 
                is being built.
            tokenized_context (List[torch.Tensor]): The
                tokenized context for the prompt.    
        Returns:
            Tuple[Node, torch.Tensor]: The root node of
                the tree and the tokenized prompt.
        """

        s = time.time()

        # Initialize the root node and tokenize the prompt
        root = Node(-1, None, 0)
        # Expand the tree from the root node to the
        # specified height, excluding special tokens
        special_tokens_ids = self.tokenizer.all_special_ids
        root = self._expand_tree(root,
                                 tokenized_context,
                                 len(tokenized_context[0]),
                                 special_ids=special_tokens_ids)
        # Set the cumulative log probability of the root node to 0
        root.cum_log_probability = 0

        if self.mode:
            print(f"{self.for_loop_counter} - self.for_loop_counter")
            print(f"build tree for first tokens: {time.time() - s}")

        # Return the root node and the tokenized prompt
        return root

    def _candidate_sequences(self, context, max_token_length, prompt=''):
        """
        Generates a set of candidate sequences 
        from the given context by considering all
        possible substrings within a specified 
        length limit.
        
        These candidates are then prefixed with the
        provided prompt to form complete sequences.
    
        Args:
            context (str): The input context from 
                which to generate candidate sequences.
            max_token_length (int): The maximum length 
                of a candidate sequence in terms of tokens.
            prompt (str, optional): A prefix to be added 
                to each candidate sequence. Defaults to an
                empty string.
    
        Returns:
            list: A list of candidate sequences, each 
                starting with the provided prompt.
        """
        s = time.time()

        # Calculate the restriction based on the
        # length of the text and the maximum token
        # length
        restriction = min(len(context) + 1, max_token_length)
        # Initialize an empty set to store unique
        # substring candidates
        substring_candidates = set()

        if self.is_fast:
            # if we want a narrow candidate space
            # as a sub-string response, but faster
            # execution time

            substring_context = context.split(' ')
            word_lengths = [len(word) for word in substring_context]
            avg_length = (sum(word_lengths) +
                          len(word_lengths)) / len(word_lengths)
            step_size = int(restriction // avg_length) + 1

            # Iterate over the text to generate all possible
            # substrings within the restriction
            for i in range(len(substring_context)):
                loop_boundary = min(
                    len(substring_context) + 1, i + 1 + step_size)
                for j in range(i + 1, loop_boundary):
                    candidate = " ".join(substring_context[i:j])
                    substring_candidates.add(candidate)
        else:
            # all possible character-by-character context splits
            for i in range(len(context)):
                loop_boundary = min(len(context) + 1, i + 1 + restriction)
                for j in range(i + 1, loop_boundary):
                    substring_candidates.add(context[i:j])

        # Sort the set of substring candidates
        # for reproducibility
        substring_candidates = sorted(substring_candidates)
        # Prefix each candidate with the prompt to
        # form complete sequences
        sequences = [prompt + candidate for candidate in substring_candidates]

        if self.mode:
            print(
                f"get candidates: last 2 tokens + all substring candidates {time.time() - s}"
            )
            sequences

        return sequences

    def _compute_logprob(self, common_part, nodes):
        self.compute_logprob_func(common_part, nodes, self.llm, self.tokenizer,
                                  self.mode)

    def _get_topk_nodes(self, nodes, k):
        """
        Selects the top `k` nodes from a given 
        list of nodes based on their cumulative
        log probabilities, normalized by their 
        depth.
    
        Args:
            nodes (List[Node]): A list of nodes
                from which to select the top `k` nodes.
            k (int): The number of top nodes 
                to select.
    
        Returns:
            List[Node]: A list of the top `k` nodes,
                sorted by their normalized cumulative 
                log probabilities.
    
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
        
        Args:
            context (str): The input context from which 
                to generate candidate sequences.
            chosen_options (List[str]): A list of options 
                that each candidate sequence must start with.
            max_candidate_length (int): The maximum length 
                of a candidate sequence in terms of tokens.
            prompt (str, optional): A prefix to be added to 
                each candidate sequence. Defaults to an 
                empty string.
    
        Returns:
            list: A list of candidate sequences, each 
                starting with one of the chosen options 
                and prefixed with the provided prompt.
        """
        s = time.time()
        # Calculate the restriction based on the
        # length of the text and the maximum candidate length
        restriction = min(len(context) + 1, max_candidate_length)
        # Initialize an empty set to store unique substring candidates
        substring_candidates = set()

        # Iterate over the text to generate all
        # possible substrings within the restriction

        if self.is_fast:
            substring_context = context.split(' ')
            word_lengths = [len(word) for word in substring_context]
            avg_length = (sum(word_lengths) +
                          len(word_lengths)) / len(word_lengths)
            step_size = int(restriction // avg_length) + 1

            for i in range(len(substring_context)):
                for j in range(
                        i + 1,
                        min(len(substring_context) + 1, i + 1 + step_size)):
                    candidate = prompt + " ".join(substring_context[i:j])
                    tokenized_candidate = self.tokenizer.encode(
                        candidate,
                        return_tensors="pt",
                        add_special_tokens=False)[0]
                    # Check if the candidate starts with one of the chosen options
                    if any(
                            torch.equal(tokenized_candidate[:len(option)],
                                        option) for option in chosen_options):
                        substring_candidates.add(candidate)
        else:
            for i in range(len(context)):
                for j in range(i + 1, i + 1 + restriction):
                    candidate = prompt + context[i:j]
                    tokenized_candidate = self.tokenizer.encode(
                        candidate,
                        return_tensors="pt",
                        add_special_tokens=False)[0]
                    # Check if the candidate starts with one of the chosen options
                    if any(
                            torch.equal(tokenized_candidate[:len(option)],
                                        option) for option in chosen_options):
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
        Args:
            node (Node): The starting node from which
                to traverse the tree.
    
        Returns:
            Node: The node that is just before a branching
                point in the tree.
    
        The function begins by entering a loop that continues 
        until it finds a node with more than one child.
        It starts with the given node and checks its children. 
        If the node has exactly one child, the function 
        moves to that child and continues the process. This 
        ensures that the function traverses down the tree 
        until it reaches a node that is about to branch into 
        multiple paths.
    
        Once the branching point is found, the function breaks
        out of the loop and returns the node that led to 
        this branching. This node is the one just before the 
        branching point, and it can be used for further 
        processing or analysis.
        """
        nodes_sequence = [node]
        while True:
            children = node.children
            # If the node has exactly one child,
            # move to that child
            if len(children) == 1:
                node = children[0]
                nodes_sequence.append(node)
            else:
                # If the node has more than one child,
                # it's a branching point
                break
        # Return the node just before the branching point
        return nodes_sequence

    def _iteration(self, working_list, common_part, k):
        """
        Performs an iteration of the sequence
        generation process by computing the 
        cumulative log probabilities
        of the nodes in the working list and 
        selecting the top `k` nodes.
    
        Args:
            working_list (List[Node]): The list of 
                nodes for which the cumulative log 
                probabilities are to be computed
                and from which the top `k` nodes
                are to be selected.
            common_part (str): A common part of the 
                text that is shared by all nodes in 
                the tree. This is used to ensure that 
                the model's predictions are relevant to 
                the context of the input text.
            k (int): The number of top nodes to select.
    
        Returns:
            List[Node]: The top `k` nodes from the working 
                list, sorted by their normalized cumulative 
                log probabilities.
        """
        # Add children to wotking list for every node in it
        working_list = self._update_working_list_with_children(working_list)

        # Compute the cumulative log probabilities for each node in the working list
        self._compute_logprob(common_part, working_list)
        # Select the top k nodes from the working list based on their cumulative log probabilities
        return self._get_topk_nodes(working_list, k)

    def _update_working_list_with_children(self, working_list):
        """
        Updates the working list of nodes by adding 
        the children of each node in the list,
        specifically those that are just before a 
        branching point in the tree.
    
        Args:
            working_list (List[Node]): The current 
                working list of nodes to be updated.
    
        Returns:
            List[Node]: The updated working list of 
                nodes, including the children of each 
                node in the original list.
        """
        s = time.time()
        for node in working_list:
            for c in node.children:
                candidate_seq_to_add = self._get_nodes_seq_before_branch(c)
                if candidate_seq_to_add[-1] not in working_list:
                    working_list.extend(candidate_seq_to_add)

        if self.mode:
            print(
                f"add children sequences before found branch into working list: {time.time() - s}"
            )
        return working_list

    def substring(self,
                  prompt,
                  context,
                  k,
                  max_substring_length,
                  return_full_text=False):
        """
        Generates and evaluates candidate sequences 
        based on a given prompt and context, selecting 
        the top `k` nodes.
    
        Args:
            prompt (str): The prompt for which the 
                tree is being built and from which 
                the last part and the common part 
                are extracted.
            context (str): The context from which 
                candidate sequences are generated.
            k (int): The number of top nodes to select.
            max_substring_length (int): The maximum 
                length of a candidate sequence in 
                terms of tokens.
    
        Returns:
            Tuple[List[Node], Node]: The top `k` nodes 
                from the working list and the initial 
                root node of the tree.
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

        tokenized_s_cand = list(
            self.tokenizer(substring_candidates,
                           return_tensors="pt",
                           padding=True,
                           add_special_tokens=False)['input_ids'])

        # Build the tree structure from the tokenized candidate sequences
        initial_root = self._build_tree(tokenized_s_cand)

        # Expand the tree from the root node
        node_before_branch = self._get_nodes_seq_before_branch(
            initial_root)[-1]
        first_branch_children = node_before_branch.children

        if not first_branch_children:
            first_branch_children = [node_before_branch]

        # Last token of the prompt can be changed
        # Therefore, we have to capture not tokens before first
        # branch, but all its children after branch

        working_list = []
        for node in first_branch_children:
            wl_len = len(working_list)

            for c in node.children:
                candidate_seq_to_add = self._get_nodes_seq_before_branch(c)
                if candidate_seq_to_add[-1] not in working_list:
                    working_list.extend(candidate_seq_to_add)

            if wl_len == len(working_list):
                working_list.append(node)

        self._compute_logprob(common_part, working_list)
        working_list = self._get_topk_nodes(working_list, k)

        # Generate expanded candidate sequences based
        # on the chosen candidates
        chosen_candidates = list(map(lambda x: x.token_sequence, working_list))

        if self.mode:
            print("chosen_candidates for explansion: ")
            for c in chosen_candidates:
                print(self.tokenizer.decode(c))
            print()

        expanded_candidates = self._candidate_sequences_exp(
            context, chosen_candidates, max_substring_length, last_part)

        tokenized_expanded_candidates = list(
            self.tokenizer(expanded_candidates,
                           return_tensors="pt",
                           padding=True,
                           add_special_tokens=False)['input_ids'])

        # Update the working list with children nodes
        working_list[0].parent_node.children = working_list

        # Expand the tree with the expanded candidate sequences
        for node in working_list:
            self._expand_tree(node,
                              tokenized_expanded_candidates,
                              len(tokenized_expanded_candidates[0]),
                              position=node.depth,
                              special_ids=self.tokenizer.all_special_ids)

        # Iteratively refine the set of candidate sequences based
        # on their likelihood
        while True:
            prev_w_l = deepcopy(working_list)
            working_list = self._iteration(working_list, common_part, k)
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
