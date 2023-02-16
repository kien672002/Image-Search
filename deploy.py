import streamlit as st
from backbone.resnet import feature_extractor
from utils.dataloader import get_transformation
import torch
import faiss
import pathlib
from PIL import Image


ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']

@st.cache_resource   # to avoid reruning
def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in image_root.iterdir():
        if image_path.exists() and image_path.suffix.lower() in ACCEPTED_IMAGE_EXTS:
            image_list.append(image_path)
    image_list = sorted(image_list, key=lambda x: int(
        x.name.split('.')[0].split('_')[-1]))
    return image_list

faiss_bin_path = "data_index.bin"
image_root = "images"
top_k = 5

@st.cache_resource   # to avoid generate model multiple time
def get_model():
    print("Generate new model")
    model = feature_extractor()
    model.to(device)
    return model

if __name__ == "__main__":

    device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model()

    uploaded_image = st.file_uploader("Upload image here")

    if (uploaded_image != None):
        if '.'+uploaded_image.name.split('.')[-1] not in ACCEPTED_IMAGE_EXTS:
            st.header(f"Your file must be one of {ACCEPTED_IMAGE_EXTS} format")
        else:
            pass
            img_list = get_image_list(image_root)
            transform = get_transformation()
            pil_image = Image.open(uploaded_image).convert('RGB')

            st.image(pil_image, caption="Your input image")

            image_tensor = transform(pil_image)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                feat = model(image_tensor)
                feat = feat.view((image_tensor.size(0), -1))
                
            indexer = faiss.read_index(faiss_bin_path)

            distances, indices = indexer.search(
                feat.cpu().detach().numpy(), k=top_k)
            
            print(distances, indices)
            indices = indices[0]
            distances = distances[0]
            
            if (indices != []):
                st.subheader("Your query images: ")
            for ii, index in enumerate(indices):
                # print(img_list[index], distances[ii])
                # st.write(img_list[index])
                st.image(Image.open(img_list[index]))
