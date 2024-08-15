import torch

class Node:
    """
    A class representing a node in a tree structure. Each node contains 
    information about its token ID, its parent node, its children nodes, 
    its depth in the tree, and its cumulative log probability.

    Attributes:
        token_id (int): The ID of the token associated 
            with this node.
        parent_node (Node): The parent node of this node.
            None if this is the root node.
        children (list): A list of child nodes.
        depth (int): The depth of this node in the tree.
        cum_log_probability (float): The cumulative log 
            probability of this node.
        token_sequence (torch.Tensor): A tensor representing
            the sequence of tokens from the root to this node.
    """

    def __init__(self,
                 token_id: int,
                 parent_node: 'Node',
                 depth: int):
        """
        Initializes a new Node instance.

        Args:
            token_id (int): The ID of the token
                associated with this node.
            parent_node (Node): The parent node
                of this node. None if this is 
                the root node.
            depth (int): The depth of this node 
                in the tree.
        """
        self.token_id = token_id
        self.parent_node = parent_node
        self.children = []
        self.depth = depth
        self.cum_log_probability = None

        # Initialize the token_sequence based on the 
        # parent node's token_sequence and the current
        # token_id
        if depth:
            token_id_tensor = torch.tensor([self.token_id],
                                            dtype=torch.long)
            parent_sequence = self.parent_node.token_sequence
            self.token_sequence = torch.cat((parent_sequence,
                                             token_id_tensor))
        else:
            self.token_sequence = torch.tensor([], dtype=torch.long)

    def __str__(self) -> str:
        """
        Returns a string representation of the node,
        including its token sequence.

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
        
        