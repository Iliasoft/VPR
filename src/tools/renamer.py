import shutil
from files import join

dst_dir = 'E:/AITests/EbaySet/parsingTest24/primers/tmp'
src_dir = 'E:\AITests\EbaySet\parsingTest24\primers\skews'

image_suffix = '__0.jpg'
files_to_move = [
8086,
14237,
17646,
20767,
20954,

]

for fn in files_to_move:

    try:

        shutil.move(
            join(src_dir, str(fn) + image_suffix),
            join(dst_dir, str(fn) + image_suffix)
        )
    except:
        print("Warning: can't rename similarity file", fn)
