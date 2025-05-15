from read_jzp import get_jzp_string
import os

# relations: Right, Left, Middle, Above, Below, Inside, Left_down_surround, Right_up_surround, Left_up_surround
# ⿱ : Above, Below A:B
# ⿰ : Left, Right L:R
# ⿸ : Left_up_surround, Inside Lus:I
# ⿺ : Right_down_surround, Inside Rds:I
# ⿹ ：Right_up_surround, Inside Rus:I
# ⿲ : Left, Middle, Right L:M:R
# ⿳ : Above, Middle, Below A:M:B
# '⿱', '⿰', 'LEFT_FINGER_DA', '5', '⿺', 'RIGHT_TIAO', '6'
# O,⿱_1, ⿱, O
# O,⿰_2, ⿰, OA
# O,LEFT_FINGER_DA_3, LEFT_FINGER_DA, OAL
# O,5_4, 5, OALR
# O,⿺_5, ⿺, OALRB
# O,RIGHT_TIAO_6, RIGHT_TIAO, OALRBRds
# O,6_7, 6, OALRBRdsI


relation_dict = {"⿱" : ["Above", "Below", "A", "B"],
                 "⿰" : ["Left", "Right", "L", "R"],
                 "⿸" : ["Left_up_surround", "Inside", "Lus", "I"],
                 "⿺" : ["Left_down_surround", "Inside", "Rds", "I"],
                 "⿹" : ["Right_up_surround", "Inside", "Rus", "I"],
                 "⿲" : ["Left", "Middle", "Right", "L", "M", "R"],
                 "⿳" : ["Above", "Middle", "Below", "A", "M", "B"]}

def get_object(items, paths, index, res):
    if index == len(items):
        return len(items)
    item = items[index]
    if item in ['⿱','⿰','⿸','⿺','⿹']:
        left = relation_dict[item][2]
        right = relation_dict[item][3]
        res.append(",".join(["O", item + '^' + str(index + 2), item, "".join(paths)]))
        paths.append(left)
        index = get_object(items, paths, index + 1, res)
        paths.pop()
        paths.append(right)
        index = get_object(items, paths, index, res)
        paths.pop()
    elif item in ['⿲','⿳']:
        first = relation_dict[item][3]
        second = relation_dict[item][4]
        third = relation_dict[item][5]
        res.append(",".join(["O", item + '^' + str(index + 2), item, "".join(paths)]))
        paths.append(first)
        index = get_object(items, paths, index + 1, res)
        paths.pop()
        paths.append(second)
        index = get_object(items, paths, index, res)
        paths.pop()
        paths.append(third)
        index = get_object(items, paths, index, res)
        paths.pop()
    else:
        res.append(",".join(["O", item + '^' + str(index + 2), item, "".join(paths)]))
        index = index + 1
    return index 

def get_relations(items, index, last_index, relation, relations):
    if index == len(items):
        return len(items)
    item = items[index]
    if item in ['⿱','⿰','⿸','⿺','⿹']:
        left_rel = relation_dict[item][0]
        right_rel = relation_dict[item][1]
        left_item=items[last_index]
        right_item=item
        if relation:
            relations.append(",".join(["R", left_item + '^' + str(last_index + 2), right_item + '^' + str(index + 2), relation]))
        next_index = get_relations(items, index + 1, index ,left_rel, relations)

        next_index = get_relations(items, next_index, index, right_rel, relations)
    elif item in ['⿲','⿳']:
        first_rel = relation_dict[item][0]
        second_rel = relation_dict[item][1]
        third_rel = relation_dict[item][2]
        left_item=items[last_index]
        right_item=item
        if relation:
            relations.append(",".join(["R", left_item + '^' + str(last_index + 2), right_item + '^' + str(index + 2), relation]))
        next_index = get_relations(items, index + 1, index, first_rel, relations)
        next_index = get_relations(items, next_index, index, second_rel, relations)
        next_index = get_relations(items, next_index, index, third_rel, relations)
    else:
        left_item=items[last_index]
        right_item=item
        if relation:
            relations.append(",".join(["R", left_item + '^' + str(last_index + 2), right_item + '^' + str(index + 2), relation]))
        next_index = index + 1
    return next_index

def write_to_file(file_name, content):
    with open(file_name, 'w') as f:
        for item in content:
            f.write(item)
            f.write('\n')
            
def create_tree():
    labels, image_paths = get_jzp_string()
    os.makedirs(os.path.dirname("./data/wushen_jzp/tree"), exist_ok=True)
    for label, img_path in zip(labels, image_paths):
        text = label.split(' ')
        items = text[1:]
        path = img_path.split('/')
        img = path[-1]
        if items[0] in relation_dict.keys():
            objs = []
            paths = ['O']
            rels = []
            get_object(items, paths, 0, objs)
            get_relations(items, 0, 0, None, rels)

            write_to_file(os.path.join('./data/WUSHENG_JZP/tree/'+ img + ".lg"), objs + rels)
        else:
            write_to_file(os.path.join('./data/jzpdata/tree/' + img + ".lg"), label)

# items = ['⿲','⿱', 'LEFT_FINGER_DA', '5', '⿰', 'LEFT_FINGER_DA', '5', '⿺', 'RIGHT_TIAO', '6']
# paths = ['O']
# objs = []
# get_object(items, paths, 0, objs)
# rels = []
# get_relations(items, 0, 0, None, rels)

# print(objs)
# print(rels)