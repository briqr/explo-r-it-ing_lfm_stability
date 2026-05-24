import torch
import torch.nn.functional as F
from PIL import Image
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import onnxruntime as ort
import numpy as np
import cv2
import argparse
from tqdm import tqdm



def load_arcface_model(model_dir):
    model_path = f"{model_dir}/glintr100.onnx"
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session, input_name

def preprocess_face(img, size=112):
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    img = img.astype(np.float32) / 127.5 - 1.0  
    img = np.transpose(img, (2,0,1))  
    img = np.expand_dims(img, axis=0)
    return img

def get_arcface_embedding(session, input_name, img):
    emb = session.run(None, {input_name: img})[0]
    emb = emb / np.linalg.norm(emb)
    return emb


def build_path(epoch, ckpt, results_dir):
    ckpt_path = os.path.join(*[results_dir, ckpt, 'checkpoints', epoch]) 
    exp_dir = ckpt_path.split('/')[1]
    epoch = ckpt_path.split('/')[-1].split('.')[0]
    save_dir = os.path.join(*[results_dir, exp_dir, 'vis', 'vis_%s'%(epoch)])
    return save_dir

def get_samples(epoch, ckpt, results_dir, added_str=''): 
    save_dir = build_path(epoch, ckpt, results_dir)
    samples_save_path = os.path.join(save_dir, 'generated_samples%s.pth'%added_str)
    all_samples = torch.load(samples_save_path, weights_only=False)
    return all_samples


def arcface_features_from_samples(images, session, input_name, save_path, n=4096):

    if os.path.exists(save_path):
        print(f"Loading cached ArcFace embeddings from {save_path}")
        embs = np.load(save_path)
    else:
        print(f"Extracting ArcFace embeddings and saving to {save_path}")
        embs = []
        for im in tqdm(images, desc="ArcFace embedding"):
            arr = (np.transpose(im.numpy(), (1,2,0)) * 255).astype(np.uint8)
            img = preprocess_face(arr)
            emb = get_arcface_embedding(session, input_name, img)
            embs.append(emb)
        embs = np.vstack(embs).squeeze()
        np.save(save_path, embs)
    return embs[:n]


def arcface_similarity(images1, images2, cache_path1, cache_path2, n=4096):

    model_dir = "/p/project1/cache/.insightface/models/antelopev2"
    arcface_sess, arcface_input = load_arcface_model(model_dir)

    embs1 = arcface_features_from_samples(images1, arcface_sess, arcface_input, cache_path1, n)
    embs2 = arcface_features_from_samples(images2, arcface_sess, arcface_input, cache_path2, n)

    
    if False: #for unrelated pairs
        perm = torch.randperm(len(embs2))
        embs2 = embs2[perm]
    sims = np.sum(embs1 * embs2, axis=1)
    mean_sim = sims.mean()
    std_sim  = sims.std()

    print(f"ArcFace similarity: {mean_sim:.4f} ± {std_sim:.4f}")
    return float(mean_sim)



def main(args):

    results_dir1 = args.results1
    path1 = build_path(args.epoch, args.ckpt1, results_dir1)
    all_samples1 = get_samples(args.epoch, args.ckpt1, results_dir1, added_str='')
    results_dir2 = args.results2

    path2 = build_path(args.epoch2, args.ckpt2, results_dir2)
    all_samples2 = get_samples(args.epoch2, args.ckpt2, results_dir2, added_str='')
    n = 4096

    arcface_path1 = os.path.join(path1, 'arcface_features.npy')
    arcface_path2 = os.path.join(path2, 'arcface_features.npy')
        
    sims = arcface_similarity(all_samples1[:n], all_samples2[:n], arcface_path1, arcface_path2,n=n)   
    print(f"ArcFace cosine similarity: {sims:.4f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0) 
    parser.add_argument("--results1", type=str, default="celebhq_models")
    parser.add_argument("--results2", type=str, default="celebhq_models")
    parser.add_argument("--epoch", type=str, default='0130000.pt')
    parser.add_argument("--epoch2", type=str, default='0130000.pt')
    parser.add_argument("--ckpt1", type=str, default="celebhq_fm_unpruned_s2_pruned00")
    parser.add_argument("--ckpt2", type=str, default="celebhq_fm_random_pruned05")
    args = parser.parse_args()
    main(args)
