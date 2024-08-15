import torch
import torch.nn.functional as F
import time
from vllm import SamplingParams
from vllm.utils import random_uuid
from .utils.tokens_tree_node import Node


def vllm_compute_logprob(common_part,
                         nodes,
                         llm,
                         tokenizer,
                         mode):
    """
    When vllm llm model is used:

    Computes the cumulative log probabilities
    for each node in the tree structure,
    given a common part of the text (user 
    prompt wihtout last 2 tokens) and a list 
    of nodes.

    This function is crucial for evaluating the 
    likelihood of each candidate sequence generated 
    from the context text.
    It does so by leveraging the transformer model
    to predict the next token in the sequence and 
    then calculating the log probability of each 
    token. 
    
    Args:
        common_part (str): A common part of the text
            (user prompt wihtout last 2 tokens) that 
            is shared by all nodes in the tree. This 
            is used to ensure that the model's 
            predictions are relevant to the context 
            of the input prompt.
        nodes (List[Node]): A list of nodes for which 
            the cumulative log probabilities are to 
            be computed.

    Returns:
        None: The function modifies the nodes in-place, 
            updating their cumulative log probabilities.

    The function begins by initializing an empty list for
    the input batch and two empty lists for mapping nodes
    to their corresponding log probabilities and input 
    sequences. It then iterates over each node, checking if
    its cumulative log probability has been set. If not, 
    it constructs the input sequence for the model by 
    concatenating the common part of the text with the token 
    sequence of the node. This input sequence is then
    added to the input batch and the node is mapped to its 
    corresponding log probability.

    Once all nodes have been processed, the function tokenizes 
    the input batch using the tokenizer and feeds it into the 
    model to get the logits. The log probabilities are then 
    calculated using the log_softmax function.

    Finally, the function iterates over the nodes again, this time 
    updating their cumulative log probabilities based on the log 
    probabilities of their tokens and the cumulative log 
    probabilities of their parent nodes.
    
    This process ensures that each node's cumulative log 
    probability reflects the likelihood of the sequence of 
    tokens leading up to it.
    """
    def process_log_probs(prompt_tokens,
                          generated_tokens, 
                          logits):
        log_probs = F.log_softmax(logits, dim=0)
        log_probs_mapping[tuple(prompt_tokens)] = log_probs
        return logits

    s = time.time()
    
    # Calculate the number of tokens in the 
    # common part of the text
    inference_counter = 0
    log_probs_mapping = {}
    node_output_map = {}
    nodes_map = []
    results = None
    sampling_params = SamplingParams(max_tokens=1,
                                     prompt_logprobs=1,
                                     logits_processors=[process_log_probs])

    # get all nodes without log probability computed yet
    nodes_map = set(node for node in nodes if node.cum_log_probability is None)

    # get all parents of nodes from nodes_map whose log
    # probability has not been calculated yet
    parents_without_logprob = set()
    for node in nodes_map:
        parent_tmp = node.parent_node
        while parent_tmp.cum_log_probability is None:
            if parent_tmp not in nodes_map:
                parents_without_logprob.add(parent_tmp)
            parent_tmp = parent_tmp.parent_node

    # concatenate two sets of nodes in order to pass to the llm
    nodes_map = list(nodes_map) + list(parents_without_logprob)

    if not nodes_map:
        # if all log probs already computed
        return
    
    s1 = time.time()

    common_part_encoded = tokenizer.encode(common_part) 
    inputs_map = {}
    for node in nodes_map:
        inp = common_part_encoded + node.parent_node.token_sequence.tolist()
        # we save nodes with the same 
        # parent node, because they will
        # have the same input prompt for 
        # the lmm
        inputs_map.setdefault(tuple(inp), []).append(node)
    
    if mode:
        print(f"\ncreate inputs map: {time.time() - s1}\n")
        
        
    s2 = time.time()    
    for input_prompt in inputs_map:
        # Feed the input into the model to get the logits
        llm.generate(prompt=None,
                     prompt_token_ids=list(input_prompt),
                     sampling_params=sampling_params)
        inference_counter += 1

    # await until all prompts will be processed

    s2e = time.time()

    if mode:
        print(f"inference counter: {inference_counter}")
        print(f"\nawait for model response: {s2e - s2}\n")
        print(f"inference_counter/time: {inference_counter/(s2e - s2)}")

    s3 = time.time()
    
    for input_prompt in inputs_map:
        for node in inputs_map[input_prompt]:
            # save logprobs of concrete tokens
            # (we saved logprobs for all tokens
            #  during `.generate()` call)
            node_output_map[node] = log_probs_mapping[input_prompt][node.token_id]

    if mode:
        print(f"save log probs: {time.time() - s3}")
    
            
    # Iterate over the nodes again to
    # update their 
    # cumulative log probabilities
    for node in node_output_map:
        node_log_prob = node_output_map[node]

        if mode:
            print("children")
            print(node.token_id)
            print(node_log_prob)
        
        if node.parent_node.cum_log_probability is None:
            # for parent nodes without stored logprobability 
            
            parents_sequence_without_logprob = []
            parent_tmp = node.parent_node
            while parent_tmp.cum_log_probability is None:
                # get sequence of parent nodes with None log prob
                parents_sequence_without_logprob.append(parent_tmp)
                parent_tmp = parent_tmp.parent_node

            parents_log_probs = []

            for parent_tmp in parents_sequence_without_logprob:
                key = common_part_encoded + parent_tmp \
                                            .parent_node \
                                            .token_sequence \
                                            .tolist()
                parent_logprob = log_probs_mapping[tuple(key)][parent_tmp.token_id]
                parents_log_probs.append(parent_logprob)
        
                if mode:
                    print("parent without logprob: ")
                    print(parent_tmp.token_id)
                    print(parents_log_probs[-1])


            # Calculate the cumulative log probability for each parent node
            number_of_parents_without_logbrob = len(parents_sequence_without_logprob)
            for n_id in range(number_of_parents_without_logbrob - 1, -1, -1):
                parents_sequence_without_logprob[n_id].cum_log_probability = \
                (parents_log_probs[n_id] + parents_sequence_without_logprob[n_id]
                                            .parent_node.cum_log_probability)
            

        # Update the node's cumulative log probability
        node.cum_log_probability = node.parent_node.cum_log_probability + \
                                    node_log_prob

    if mode:
        print("log prob nodes: ")
        print(len(nodes_map))
        for node in nodes_map:
            print(f"'{tokenizer.decode(node.token_sequence)}';
                    '{node.cum_log_probability}';
                    '{node.depth}'")
        
        print()
        print(f"compute log_probs call {time.time() - s}")



def hf_compute_logprob(common_part,
                       nodes,
                       model,
                       tokenizer,
                       mode):
    """
    When Hugging Face llm model is used:

    Computes the cumulative log probabilities
    for each node in the tree structure,
    given a common part of the text (user 
    prompt wihtout last 2 tokens) and a list 
    of nodes.

    This function is crucial for evaluating the 
    likelihood of each candidate sequence generated 
    from the context text.
    It does so by leveraging the transformer model
    to predict the next token in the sequence and 
    then calculating the log probability of each 
    token. 
    
    Args:
        common_part (str): A common part of the text
            (user prompt wihtout last 2 tokens) that 
            is shared by all nodes in the tree. This 
            is used to ensure that the model's 
            predictions are relevant to the context 
            of the input prompt.
        nodes (List[Node]): A list of nodes for which 
            the cumulative log probabilities are to 
            be computed.

    Returns:
        None: The function modifies the nodes in-place, 
            updating their cumulative log probabilities.

    The function begins by initializing an empty list for
    the input batch and two empty lists for mapping nodes
    to their corresponding log probabilities and input 
    sequences. It then iterates over each node, checking if
    its cumulative log probability has been set. If not, 
    it constructs the input sequence for the model by 
    concatenating the common part of the text with the token 
    sequence of the node. This input sequence is then
    added to the input batch and the node is mapped to its 
    corresponding log probability.

    Once all nodes have been processed, the function tokenizes 
    the input batch using the tokenizer and feeds it into the 
    model to get the logits. The log probabilities are then 
    calculated using the log_softmax function.

    Finally, the function iterates over the nodes again, this time 
    updating their cumulative log probabilities based on the log 
    probabilities of their tokens and the cumulative log 
    probabilities of their parent nodes.
    
    This process ensures that each node's cumulative log 
    probability reflects the likelihood of the sequence of 
    tokens leading up to it.
    """
    if mode:
        print("log prob nodes: ")
        print(len(nodes))
        for node in nodes:
            print(tokenizer.decode(node.token_sequence))
        print()

    s = time.time()

    # Calculate the number of tokens in the common part of the text
    skip_logits = len(tokenizer.encode(common_part))
    # Initialize dicts for the (key: input text, value: token)
    # and log probabilities mapping (key: node,
    # value: prob)
    node_mapping = {}
    log_probs_mapping = {}

    # Iterate over each node
    for node in nodes:
        # Check if the node's cumulative log probability has been set
        if node.cum_log_probability is None:
            # Construct the input sequence for the model
            # we get tokens of the parent, since we need only
            # them for getting log probability for current considered node token
            inp = common_part + tokenizer.decode(node.parent_node.token_sequence)
            # Map the node to its input sequence
            node_mapping.setdefault(inp, []).append(node)

    # If there are no nodes to process, return
    if not node_mapping:
        return

    # Tokenize the input batch
    tokenized_model_input = tokenizer(list(node_mapping.keys()),
                                           return_tensors="pt",
                                           padding=True,
                                           add_special_tokens=True)

    # Feed the tokenized input into the model to get the logits
    with torch.no_grad():
        outputs = model(**tokenized_model_input)
        logits = outputs.logits[:, skip_logits - 1:, :]
        # Calculate the log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Iterate over the nodes again to update their cumulative log probabilities
    for idx, inp in enumerate(list(node_mapping.keys())):
        for node in node_mapping[inp]:
            # Get the tokens for the current node
            tokens = tokenized_model_input['input_ids'][idx, skip_logits - 1:]

            # Calculate the number of tokens before padding
            first_padding = torch.sum(tokens != tokenizer.pad_token_id).item()

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
                parents_log_probs.append(log_probs[idx, first_padding - i, tokens[first_padding - i + 1]])
                i += 1
                parent_tmp = parent_tmp.parent_node

            # Calculate the cumulative log probability for each parent node
            number_of_parents_without_logbrob = len(parents_sequence_without_logprob)
            for n_id in range(number_of_parents_without_logbrob - 1, -1, -1):
                (parents_sequence_without_logprob[n_id]
                    .cum_log_probability) = (parents_log_probs[n_id] +
                                            parents_sequence_without_logprob[n_id]
                                            .parent_node.cum_log_probability)

            # Update the node's cumulative log probability
            log_probs_mapping[node] = node.parent_node.cum_log_probability
            node.cum_log_probability = log_probs_mapping[node] + log_probs[idx, first_padding - 1, node.token_id]

    if mode:
        print(f"compute log_probs call {time.time() - s}")