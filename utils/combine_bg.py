import copy
import random
import os
from tqdm import tqdm
import concurrent.futures
from PIL import Image

def remove_part_bg(img,color=None):

    img = img.convert('RGBA')
    image = copy.deepcopy(img)
    newImage = []
    if not color:
        for item in image.getdata():
            if all(0 <= i <= 21 for i in item[:3]):
                newImage.append((0, 0, 0, 0))
            else:
                newImage.append(item)
    else:
        for item in image.getdata():
            if item[:3] == (0, 0, 0):
                newImage.append((0, 0, 0, 0))
            else:
                newImage.append(item)
    image.putdata(newImage)
    return image

def resize_with_padding(img, expected_size):
    from PIL import Image, ImageOps
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)
def combine_bg_fg(image1, save_path):
    img_size=(1920, 1080)
    bg_path = os.path.join('hands_images')
    bg_img_list = os.listdir(bg_path)
    bg_path = random.choice(bg_img_list)
    bg = Image.open(os.path.join('hands_images', bg_path))
    bg = bg.convert('RGBA')
    bg = bg.resize(img_size)
    l=[0.6,0.7,.8,.9,1]
    scale=random.choice(l)
    image1 = image1.resize((int(scale*img_size[0]), int(scale*img_size[1])))
    image1=resize_with_padding(image1, img_size)
    bg.paste(image1, (0,0), image1)
    bg.convert('RGB').save(save_path)

def process_image(img_path, save_path):
    with Image.open(img_path) as img:
        rgba_img = remove_part_bg(img)
        combine_bg_fg(rgba_img, save_path)

def combine_hands(x_train):


    num_cores = os.cpu_count()-1

    total_img_num = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for img_path in x_train:
            total_img_num += 1
            base_path = os.path.join('combine_hands')
            os.makedirs(name=os.path.join(base_path), mode=0o777, exist_ok=True)
            folder_path = img_path.split('/')[-1].split('\\')[0]
            img_save_path=img_path.split('/')[-1].split('\\')[-1]
            os.makedirs(name=os.path.join(base_path, folder_path), mode=0o777, exist_ok=True)
            output_img_path = os.path.join(base_path, folder_path, img_save_path)
            # process_image(img_path, output_img_path)
            futures.append(executor.submit(process_image, img_path, output_img_path))
        pbar = tqdm(total=total_img_num, ascii=True)
        for future in concurrent.futures.as_completed(futures):
            pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    pass


