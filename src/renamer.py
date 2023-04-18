import shutil
from files import join

tmp_dir = 'E:/AITests/EbaySet/parsingTest24/primers/tmp'
dir = 'E:\AITests\EbaySet\parsingTest24\primers\goods2'

image_suffix = '__0.jpg'
files_to_move = [
    17574,#15866?
]

for fn in files_to_move:

    try:
        shutil.move(
            join(dir, str(fn) + image_suffix),
            join(tmp_dir, str(fn) + image_suffix)
        )
    except:
        print("Warning: can't rename similarity file", fn)
