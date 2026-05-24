#default is resolution 256x256, 512 is for a single ablative experiment on celebhq
PRECOMPUTED_LATENTS_PATH_CELEBHQ = 'data/celebhq/precomputed_latents/vq_train.pt'
PRECOMPUTED_LATENTS_PATH_CELEBHQ_512 = 'data/celebhq/precomputed_latents/vq_train_512.pt'
PRECOMPUTED_LATENTS_PATH_MALE = 'data/celebhq/precomputed_latents/vq_celebhq_male.pt'
PRECOMPUTED_LATENTS_PATH_FEMALE = 'data/celebhq/precomputed_latents/vq_celebhq_female.pt'
PRECOMPUTED_LATENTS_PATH_IMAGENET = 'data/imagenet/precomputed_latents/vq_imagenet.pt'
PRECOMPUTED_LATENTS_PATH_FFHQ = 'data/ffhq/precomputed_latents/vq_ffhq.pt'

#downloaded using hugging_face
CACHE_DIR =  "/p/project1/generativeaims/briq/datasets"
# downloaded manually  
FFHQ_DATAROOT = '/p/scratch/generativeaims/briq/dataset/ffhq'

celebhq_cluster_path = 'data/celebhq/clip_features/train/cluster_clip_24.pth'
imagenet_clusters_path = 'data/imagenet/clip_features/train/cluster_clip_1000.pth'
ffhq_cluster_path = 'data/ffhq/clip_features/train/cluster_24.pth'


celebhq_vq_vae_path = 'vae/vqvae_celebhq.pt'
celebhq_512_vq_vae_path = 'vae/vqvae_celebhq_512.pt'
ffhq_vq_vae_path = 'vae/vqvae_ffhq.pt'
imagenet_vq_vae_path = 'vae/vqvae_imagenet.pt' 



scale_factor_map = {'celebhq': 0.8077, 'ffhq': 0.8011, 'imagenet': 0.8260898637, 'vae_pretrained' :0.18215}       
in_channels_map = {'celebhq': 4, 'ffhq': 4, 'imagenet': 8, 'vae_pretrained': 4}  

