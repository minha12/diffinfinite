# %%
from openslide import OpenSlide

# %%
# Load a test WSI file (adjust the path to point to your TIGER dataset)
wsi = OpenSlide('../pathology-datasets/TIGER//wsibulk/images/103S.tif')

# Print image properties to make sure it's loaded correctly
print("Dimensions:", wsi.dimensions)
print("Levels:", wsi.level_count)
print("Downsampling factors:", wsi.level_downsamples)

# Extract a patch from the WSI
patch = wsi.read_region((0, 0), 4, (63488, 63488))  # (x, y), level, size
# Display the patch

# %%
patch

# %%
from wholeslidedata.image.wholeslideimage import WholeSlideImage

# %%
wsi = WholeSlideImage("../pathology-datasets/TIGER//wsibulk/images/103S.tif")

# %%
patch = wsi.get_patch(2000, 2000, 512, 512, 0.5)

# %%
patch

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
plt.imshow(patch)
plt.axis('off')  # Hide axes for cleaner display
plt.title("Image from Numpy Array")  # Add a title (optional)
plt.show()

# %%
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation


# %%
wsa = WholeSlideAnnotation("../pathology-datasets/TIGER//wsibulk/annotations-tumor-bulk/xmls/103S.xml")

# %%
annotations = wsa.select_annotations(2000, 2000, 512, 512)

# %%
annotations

# %%
wsi = OpenSlide("../pathology-datasets/TIGER//wsibulk/annotations-tumor-bulk/masks/103S.tif")

# %%
# Print image properties to make sure it's loaded correctly
print("Dimensions:", wsi.dimensions)
print("Levels:", wsi.level_count)
print("Downsampling factors:", wsi.level_downsamples)




