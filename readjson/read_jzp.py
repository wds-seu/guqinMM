import json

# Specify the path to your JSON file
file_path = "dataset.json"

# Open the JSON file in read mode
with open(file_path, "r") as file:
    # Load the JSON data
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

result_strings = []
for item in data:
    type_ = item["type"]
    annotation = item["annotation"]
    annotation_str = parse_annotation(annotation)
    
    result_string = f"{type_} {annotation_str}"
    result_strings.append(result_string)

# 拼接结果字符串
final_result = "\n".join(result_strings)
print(final_result)



