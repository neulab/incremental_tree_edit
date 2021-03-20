# coding=utf-8
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree, AbstractSyntaxNode, SyntaxToken, DummyReduce


def find_by_id(candidate_list, candidate):
    for idx, candidate_i in enumerate(candidate_list):
        if id(candidate_i) == id(candidate):
            return idx

    return -1 # not found


def get_field_repr(field):
    field_string = ""
    parent_node = field.parent_node
    cur_field = field
    while parent_node:
        # which field of parent_node
        field_idx = find_by_id(parent_node.fields, cur_field)
        field_string = "%s-%d" % (str(parent_node), field_idx) + "-SEP-" + field_string
        parent_field = parent_node.parent_field
        if parent_field:
            node_idx = find_by_id(parent_field.as_value_list, parent_node)
            field_string = "%s-%d" % (str(parent_field), node_idx) + "-SEP-" + field_string
            parent_node = parent_field.parent_node
            cur_field = parent_field
        else:
            break

    field_string += str(field)

    return field_string


def get_field_node_queue(node):
    output = []

    field = node.parent_field
    while field is not None:
        node_idx = find_by_id(field.as_value_list, node)
        assert node_idx != -1
        output.append((node_idx, node))

        node = field.parent_node
        field_idx = find_by_id(node.fields, field)
        assert field_idx != -1
        output.append((field_idx, field))

        field = node.parent_field

    output = output[::-1]
    return output


def copy_tree_field(tree: AbstractSyntaxTree, field: RealizedField, bool_w_dummy_reduce=False):
    if bool_w_dummy_reduce:
        new_tree = tree.copy_and_reindex_w_dummy_reduce()
    else:
        new_tree = tree.copy_and_reindex_wo_dummy_reduce()

    root_to_field_trace = []
    cur_field = field
    while cur_field:
        cur_parent_node = cur_field.parent_node
        cur_field_idx = find_by_id(cur_parent_node.fields, cur_field)
        assert cur_field_idx != -1
        root_to_field_trace.append(('field', cur_field_idx))

        cur_parent_node_parent_field = cur_parent_node.parent_field
        if cur_parent_node_parent_field:
            cur_parent_node_idx = find_by_id(cur_parent_node_parent_field.as_value_list, cur_parent_node)
            assert cur_parent_node_idx != -1
            root_to_field_trace.append(('node', cur_parent_node_idx))

        cur_field = cur_parent_node_parent_field

    pointer = new_tree.root_node
    while root_to_field_trace:
        trace = root_to_field_trace.pop()
        if trace[0] == 'field':
            assert isinstance(pointer, AbstractSyntaxNode)
            field_idx = trace[1]
            pointer = pointer.fields[field_idx]
        else:
            assert trace[0] == 'node'
            assert isinstance(pointer, RealizedField)
            node_idx = trace[1]
            pointer = pointer.as_value_list[node_idx]

    assert isinstance(pointer, RealizedField)
    new_field = pointer

    # assert new_tree == tree # not necessary since DummyReduce may have been inserted
    # assert new_field == field

    return new_tree, new_field


def stack_subtrees(tree_node, bool_repr=False, bool_stack_syntax_token=False):
    if isinstance(tree_node, AbstractSyntaxNode):
        # safety check: need depth >= 2
        bool_has_child = False
        for field in tree_node.fields:
            if len(field.as_value_list):
                bool_has_child = True
                break
        if not bool_has_child:
            return []

        if bool_repr:
            new_memory = [tree_node.to_string()]
        else:
            new_memory = [tree_node]

        for field in tree_node.fields:
            for val in field.as_value_list:
                new_memory.extend(stack_subtrees(val, bool_repr=bool_repr,
                                                 bool_stack_syntax_token=bool_stack_syntax_token))
    elif bool_stack_syntax_token:
        if bool_repr:
            new_memory = [str(tree_node).replace(' ', '-SPACE-')]
        else:
            new_memory = [tree_node]

    else:
        new_memory = []

    return new_memory


def get_productions_str(tree_node):
    productions = dict()

    if isinstance(tree_node, AbstractSyntaxNode):
        productions[str(tree_node)] = productions.get(str(tree_node), 0) + 1
        for field in tree_node.fields:
            for val in field.as_value_list:
                for k,v in get_productions_str(val).items():
                    productions[k] = productions.get(k, 0) + v
    elif not isinstance(tree_node, DummyReduce): # dummy nodes are excluded
        productions[str(tree_node)] = productions.get(str(tree_node), 0) + 1

    return productions


def calculate_tree_prod_f1(tree_prod_pred, tree_prod_gold):
    all_preds = 0
    true_pos = 0
    for prod, count in tree_prod_pred.items():
        all_preds += count
        if prod in tree_prod_gold:
            true_pos += min(count, tree_prod_gold[prod])

    precision = true_pos * 1.0 / all_preds

    all_golds = sum([count for prod, count in tree_prod_gold.items()])
    recall = true_pos * 1.0 / all_golds

    if true_pos == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_sibling_ids(field, anchor_node, bool_rm_dummy=False):
    assert id(anchor_node.parent_field) == id(field) # has to be the same field instance

    left_sibling_ids, right_sibling_ids = [], []

    parent_node = field.parent_node
    all_sibling_nodes = [node for field in parent_node.fields for node in field.as_value_list]
    bool_left = True
    for node in all_sibling_nodes:
        if node.id == anchor_node.id:  # anchor node will be shifted right
            bool_left = False

        if bool_left:
            if bool_rm_dummy and isinstance(node, DummyReduce):
                continue
            left_sibling_ids.append(node.id)
        else:
            if bool_rm_dummy and isinstance(node, DummyReduce):
                continue
            right_sibling_ids.append(node.id)

    return left_sibling_ids, right_sibling_ids

