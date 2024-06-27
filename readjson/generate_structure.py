import random
from PIL import ImageTk, Image, ImageChops
import os
def open_images():
    images = {}
    path = "../jianzipu"
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1] == ".png":
            name = os.path.splitext(filename)[0]
            img = Image.open(os.path.join(path, filename))
            images[name] = {}
            images[name]["pil"] = img
            images[name]["tk"] = ImageTk.PhotoImage(image=img)
    images[""] = images["placeholder"]
    return images


def combine_left_right(img1, img2):
    left = img1.resize((img1.size[0] // 2, img1.size[1]))
    right = img2.resize((img2.size[0] // 2, img2.size[1]))
    new_im = Image.new('RGB', (left.size[0] + right.size[0], left.size[1]))
    x_offset = 0
    for im in [left, right]:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def combine_top_bottom(img1, img2):
    top = img1.resize((img1.size[0], img1.size[1] // 2))
    bottom = img2.resize((img2.size[0], img2.size[1] // 2))
    new_im = Image.new('RGB', (top.size[0], top.size[1] + bottom.size[1]))
    y_offset = 0
    for im in [top, bottom]:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


def combine_left_middle_right(img1, img2, img3):
    left = img1.resize((img1.size[0] // 3, img1.size[1]))
    middle = img2.resize((img2.size[0] // 3, img2.size[1]))
    right = img3.resize((img3.size[0] // 3, img3.size[1]))
    new_im = Image.new('RGB', (left.size[0] + middle.size[0] + right.size[0], left.size[1]))
    x_offset = 0
    for im in [left, middle, right]:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def combine_top_middle_bottom(img1, img2, img3):
    top = img1.resize((img1.size[0], img1.size[1] // 3))
    middle = img2.resize((img2.size[0], img2.size[1] // 3))
    bottom = img3.resize((img3.size[0], img3.size[1] // 3))
    new_im = Image.new('RGB', (top.size[0], top.size[1] + middle.size[1] + bottom.size[1]))
    y_offset = 0
    for im in [top, middle, bottom]:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


def surround_upper_left(img1, img2):
    outer = img1
    inner = img2.resize((img2.size[0] * 2 // 3, img2.size[1] * 2 // 3))
    new_im = Image.new('RGB', outer.size)
    new_im.paste(outer, (0, 0))
    whitespace = Image.new('RGB', ((outer.size[0] * 2) // 3, (outer.size[1] * 2) // 3), (255, 255, 255))
    new_im.paste(whitespace, (outer.size[0] // 3, outer.size[1] // 3))
    new_inner = Image.new('RGB', outer.size, (255, 255, 255))
    new_inner.paste(inner, (outer.size[0] // 4, outer.size[1] // 4))
    new_im = ImageChops.multiply(new_im, new_inner)
    return new_im


def surround_lower_left(img1, img2):
    outer = img1
    inner = img2.resize((img2.size[0] * 2 // 3, img2.size[1] * 2 // 3))
    new_im = Image.new('RGB', outer.size)
    new_im.paste(outer, (0, 0))
    whitespace = Image.new('RGB', ((outer.size[0] * 2) // 3, (outer.size[1] * 2) // 3), (255, 255, 255))
    new_im.paste(whitespace, (outer.size[0] // 3, 0))
    new_inner = Image.new('RGB', outer.size, (255, 255, 255))
    new_inner.paste(inner, (outer.size[0] // 4, outer.size[1] // 8))
    new_im = ImageChops.multiply(new_im, new_inner)
    return new_im


def surround_upper_right(img1, img2):
    outer = img1
    inner = img2.resize((img2.size[0] * 2 // 3, img2.size[1] * 2 // 3))
    new_im = Image.new('RGB', outer.size)
    new_im.paste(outer, (0, 0))
    whitespace = Image.new('RGB', ((outer.size[0] * 2) // 3, (outer.size[1] * 2) // 3), (255, 255, 255))
    new_im.paste(whitespace, (0, outer.size[1] // 3))
    new_inner = Image.new('RGB', outer.size, (255, 255, 255))
    new_inner.paste(inner, (outer.size[0] // 8, outer.size[1] // 4))
    new_im = ImageChops.multiply(new_im, new_inner)
    return new_im




def get_node_image(self, node):
        children = self.musical_var.get_children(node)
        content = str(self.musical_var.item(node)["values"][0])
        if len(children):  # child is no leaf node
            for action in self.actions_list:
                if action["symbol"] == content:
                    break
            return action["combine"](*[self.get_node_image(child) for child in children])
        else:
            return self.annotation_images[content]["pil"]

actions_list = [
            {"symbol": "⿰", "children": ["Left", "Right"], "combine": combine_left_right},
            {"symbol": "⿱", "children": ["Top", "Bottom"], "combine": combine_top_bottom},
            # {"symbol": "⿲", "children": ["Left", "Middle", "Right"], "combine": combine_left_middle_right},
            # {"symbol": "⿳", "children": ["Top", "Middle", "Bottom"], "combine": combine_top_middle_bottom},
            {"symbol": "⿸", "children": ["Outer", "Inner"], "combine": surround_upper_left},
            {"symbol": "⿺", "children": ["Outer", "Inner"], "combine": surround_lower_left},
            {"symbol": "⿹", "children": ["Outer", "Inner"], "combine": surround_upper_right}
        ]
# Example usage
left_path = "./jianzipu/left_fingers/"
num_path = "./jianzipu/string_nums/"
right_path = "./jianzipu/right_fingers/"
# img1 = Image.open("./jianzipu/RIGHT_GOU.png")
# img2 = Image.open("./jianzipu/3.png")
def random_combine():
    img1_files = os.listdir(num_path)
    random_img1_file = random.choice(img1_files)
    img2_files = os.listdir(right_path)
    random_img2_file = random.choice(img2_files)
    print(random_img1_file, random_img2_file)
    img1 = Image.open(os.path.join(num_path, random_img1_file))
    img2 = Image.open(os.path.join(right_path, random_img2_file))
    random_choice = random.choice(actions_list)
    print(random_choice["symbol"])
    combined_image = random_choice["combine"](img1, img2)
    combined_image.show()

random_combine()
# random_symbols = [action["symbol"] for action in actions_list]
# random_list = random.choices(random_symbols, k=4)
# print(random_list)