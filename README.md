# Transformer Diffusion
Denoising diffusion probabilistic models, with transformers instead of a U-Net.

Preliminary testing with 64x64 images shows promising results using a swin-transformer as a backbone. Examples of LSUN Bedroom images generated during testing:

![media_images_examples_678_0902ce013c5dd32af289](https://user-images.githubusercontent.com/28935064/185817684-6eda0619-9ef5-41e5-9ba4-5f7848ad4360.png)
![media_images_examples_678_e1cb3cdad10d00073bb5](https://user-images.githubusercontent.com/28935064/185817685-257afca7-682d-4838-be2e-7d59025b2495.png)
![media_images_examples_678_81efd020efaa09bfa59c](https://user-images.githubusercontent.com/28935064/185817686-9c33a84b-8f4d-40f8-9950-003188ba4085.png)
![media_images_examples_678_894a471ea534b41dfb71](https://user-images.githubusercontent.com/28935064/185817687-f6ca90c5-5a65-4dfb-82ab-bea15e9fc299.png)
![media_images_examples_678_12e3c4bd2fbaad07c199](https://user-images.githubusercontent.com/28935064/185817688-454eca53-7b35-4534-a507-b0b18d077936.png)

This test used very simple hyper parameters that are almost certainly not ideal, but were easy to debug. Additionally, recent developments such as cascaded diffusion models with conditioning augmentation were not used.
