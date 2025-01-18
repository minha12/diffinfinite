LMD on different datasets

| Dataset                     | Approximate Size | Iterations | Batch Size | Notes                                     |
|-----------------------------|------------------|------------|------------|-------------------------------------------|
| CelebA-HQ                   | ~30,000          | 410k       | 48         | Unconditional Model                       |
| FFHQ                        | ~70,000          | 635k       | 42         | Unconditional Model                       |
| LSUN-Churches               | ~126,000         | 500k       | 96         | Unconditional Model                       |
| LSUN-Bedrooms               | ~3,000,000       | 1.9M       | 48         | Unconditional Model                       |
| ImageNet (LDM-1)            | ~1.2 million     | 2M         | 7          | Conditional Model, downsampling factor 1  |
| ImageNet (LDM-2)            | ~1.2 million     | 2M         | 9          | Conditional Model, downsampling factor 2  |
| ImageNet (LDM-4)            | ~1.2 million     | 2M         | 40         | Conditional Model, downsampling factor 4  |
| ImageNet (LDM-8)            | ~1.2 million     | 2M         | 64         | Conditional Model, downsampling factor 8  |
| ImageNet (LDM-16)           | ~1.2 million     | 2M         | 112        | Conditional Model, downsampling factor 16 |
| ImageNet (LDM-32)           | ~1.2 million     | 2M         | 112        | Conditional Model, downsampling factor 32 |
| CelebA (LDM-1)              | ~30,000          | 500k       | 9          | Unconditional Model, downsampling factor 1|
| CelebA (LDM-2)              | ~30,000          | 500k       | 11         | Unconditional Model, downsampling factor 2|
| CelebA (LDM-4)              | ~30,000          | 500k       | 48         | Unconditional Model, downsampling factor 4|
| CelebA (LDM-8)              | ~30,000          | 500k       | 96         | Unconditional Model, downsampling factor 8|
| CelebA (LDM-16)             | ~30,000          | 500k       | 128        | Unconditional Model, downsampling factor 16|
| CelebA (LDM-32)             | ~30,000          | 500k       | 128        | Unconditional Model, downsampling factor 32|
| LAION                       | ~400 million pairs| 390k       | 680        | Text-to-Image                           |
| OpenImages                  | ~9 million       | 4.4M       | 24         | Layout-to-Image                         |
| COCO                        | ~120,000         | 170k       | 48         | Layout-to-Image                         |
| ImageNet                    | ~1.2 million     | 178k       | 1200       | Super Resolution                        |
| Places                      | ~1.8 million     | 860k       | 64         | Inpainting                              |
| Landscapes                  | ~1.5 million     | 360k       | 48         | Semantic-Map-to-Image                  |

