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
image_paths = []
for item in data:
    type_ = item["type"]
    image_path = item["image_path"]
    annotation = item["annotation"]
    annotation_str = parse_annotation(annotation)
    path_string = f"{image_path}"
    image_paths.append(image_path)
    result_string = f"{type_} {annotation_str}"
    result_strings.append(result_string)

# 拼接结果字符串

path_result = "\n".join(image_paths)
print(path_result)
final_result = "\n".join(result_strings)
print(final_result)






