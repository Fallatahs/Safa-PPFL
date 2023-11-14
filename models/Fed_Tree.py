class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_binary_tree(node_values):
    if not node_values:
        return None

    root = TreeNode(node_values[0])
    queue = [root]
    i = 1
    while i < len(node_values):
        current_node = queue.pop(0)
        if node_values[i] is not None:
            current_node.left = TreeNode(node_values[i])
            queue.append(current_node.left)
        i += 1
        if i < len(node_values) and node_values[i] is not None:
            current_node.right = TreeNode(node_values[i])
            queue.append(current_node.right)
        i += 1

    return root

def FedAvg_binary_tree(root):
    if not root:
        return None

    if not root.left and not root.right:
        return [root.value]

    left_result = FedAvg_binary_tree(root.left)
    right_result = FedAvg_binary_tree(root.right)

    if left_result and right_result:
        # Aggregate logic for internal nodes (example: sum)
        return left_result + right_result
    elif left_result:
        # If only the left child has a result
        return left_result
    elif right_result:
        # If only the right child has a result
        return right_result

# Example usage
node_values = [50, 13, None, 88, 77, 71, 12, 72, None, None, None, None, 48, 7, None, 76]
binary_tree_root = build_binary_tree(node_values)
accumulated_info = FedAvg_binary_tree(binary_tree_root)

print("Accumulated Information for Each Path:")
print(accumulated_info)
