
from PIL import Image
import matplotlib.pyplot as plt

# Define the image path
image_path = '../pathology-datasets/DRSK/image_patches_512_20x/images/0a7f82b5b77b6574129b584750e80959b9e681deb6c3a41288b5f68f9fa55732.png'

# Load the image
image = Image.open(image_path)

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()

# Print image information
print(f'Image size: {image.size}')
print(f'Image mode: {image.mode}')