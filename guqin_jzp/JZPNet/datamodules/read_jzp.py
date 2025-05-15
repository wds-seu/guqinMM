import json


file_path = "data/jzpdata/dataset.json"
file = open(file_path, "r")
data = json.load(file)
# {image path, type, annotation}
# {type, {content: {content, children: {content, children}}}

def parse_annotation(annotation):
    annotation_type = annotation["type"]
    content = annotation["content"]
    
    if annotation_type == "STRING_NUMBER":
        return str(content)

    elif annotation_type == "FULL_JIANZIPU":
        return parse_full_jianzipu(content)
    
    elif annotation_type == "LEFT_HAND":
        return ' '.join(content)
    
    else:
        return str(content)

def parse_full_jianzipu(content):
    result = content["content"]
    if "children" in content:
        for child in content["children"]:
            result += " " + str( parse_full_jianzipu(child))
    return result


def get_jzp_string():
    result_strings = []
    image_paths = []
    for item in data:
        type_ = item["type"]
        image_path = "./" + item["image_path"]
        annotation = item["annotation"]
        annotation_str = parse_annotation(annotation)
        path_string = f"{image_path}"
        image_paths.append(image_path)
        result_string = f"{type_} {annotation_str}"
        result_strings.append(result_string)
    return result_strings, image_paths

def get_jzp_character(): 
    #"LeftHandAnnotation"{}"Hui" "LH Fingers" "LH Slides"}
    #"FullJianzipuAnnotation"{"Technique" "Numbers" "LH Fingers" "LH Slides" "RH Plucks"  "RH Misc" "LR Technique"  "Other Symbols"} 
    #"StringNumberAnnotation"{"String Number"}
    alphabet_path = "data/jzpdata/gui_config.json"
    alphabet_file = open(alphabet_path, "r")
    alphabet = json.load(alphabet_file)
    jzp_character_list = []
    for items in alphabet["FullJianzipuAnnotation"]:
        sub_alphabet = alphabet["FullJianzipuAnnotation"][items]
        for item in sub_alphabet:
            if item is not None:
                jzp_character_list.append(item)
    jzp_character_list = jzp_character_list +  ["⿱", "⿰", "⿸", "⿺", "⿹","⿲","⿳"]

    return jzp_character_list


def get_jzp_structure(content):
    structure_list = []
    for jz_notation in content:
        structure = ""
        for symbol in jz_notation[6:]:
            if symbol in ["⿱", "⿰", "⿸", "⿺", "⿹", "⿲", "⿳"]:
                structure = structure + symbol
        if structure == "":
            structure = "None"
        structure_list.append(structure)
    return structure_list
    

