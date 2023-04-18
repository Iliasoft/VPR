# https://openbase.com/python/easyocr
# https://www.jaided.ai/easyocr/documentation/
# https://github.com/JaidedAI/EasyOCR

import sys
import pickle
import easyocr
from tqdm import tqdm
from os import listdir
from os.path import isfile, isdir
from files import join


def valid_len(ss):
    musor = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return 4 <= sum(map(lambda x: 0 if x in musor else 1, ss))

reader = easyocr.Reader(['en', 'de', 'fr', 'it'], gpu=False)

working_dir = sys.argv[1]
'''
with open(join(working_dir, "images_ocrs.pkl"), 'rb') as f:
    ocr_results = pickle.load(f)

print(len(ocr_results))
for r in ocr_results.keys():
    print(r, ocr_results[r])
assert False
'''
imgs_names = listdir(working_dir)
working_dir_ocr_results = {}
for img_fn in tqdm(imgs_names):
    if isdir(join(working_dir, img_fn)) or img_fn[-4:] != ".jpg":
        continue

    ocr_result = reader.readtext(join(working_dir, img_fn), decoder="wordbeamsearch", batch_size=256, paragraph=True)
    valid_image_ocrs = []
    for r in ocr_result:

        if valid_len(r[1]):
            # print(f"{r[0]}, {r[1]}")
            valid_image_ocrs.append(r)
    if valid_image_ocrs:
        working_dir_ocr_results[img_fn] = valid_image_ocrs

with open(join(working_dir, "images_ocrs.pkl"), 'wb') as f:
    pickle.dump(working_dir_ocr_results, f)
print(f"OCRs extracted from {len(working_dir_ocr_results)} files")