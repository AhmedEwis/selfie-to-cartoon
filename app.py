import os
import cv2
import torch
import numpy as np
from models import ResnetGenerator
import argparse
from utils import Preprocess
from PIL import Image
import ssl
import certifi

urlopen(request, context=ssl.create_default_context(cafile=certifi.where()))
import streamlit as st
import gdown
from io import BytesIO
import base64

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title='Photo to Cartoon', layout='centered', initial_sidebar_state='expanded')

def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


class Photo2Cartoon:
    def __init__(self):
        self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
        
        params = self.download_data()
        self.net.load_state_dict(params['genA2B'])
        print('[Step1: load weights] success!')
        

    def download_data(self):

        url = 'https://drive.google.com/uc?id=1wrx5qaDxPtNoyQDmiPljF1WJo3HRyUaB'
        if not os.path.exists('./photo2cartoon_weights.pt'):
            gdown.download(url, quiet=False)
        
        params = torch.load('./photo2cartoon_weights.pt', map_location='cpu')
        
        return params
    
    def inference(self, img):
        # face alignment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None
        
        print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]

        # post-process
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')
        return cartoon

    
def get_image_download_link(img):
            """Generates a link allowing the PIL image to be downloaded
            in:  PIL image
            out: href string
            """
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
            return href
        
if __name__ == '__main__':

    c2p = Photo2Cartoon()
    local_css("css/styles.css")
    st.markdown('<h1 align="center">Photo to Cartoon</h1>', unsafe_allow_html=True)


    st.markdown("### Upload your selfie here ⬇")
    image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])  # upload image


    if image_file is not None:
            #img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        img = Image.open(image_file).convert('RGB') 
        img = np.array(img)
        col1, col2 = st.beta_columns(2)

        col1.header("Original")
        cartoon = c2p.inference(img)
        try:
            if cartoon == None:
                img = Image.open(image_file).convert('RGB') 
                img = img.rotate(270)
                img = np.array(img)
                cartoon = c2p.inference(img)
        except:
            pass
        col1.image(img, use_column_width=True)

        st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
        col2.header("Cartoon")
        col2.image(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB), use_column_width=True)
            
       # cv2.imwrite('./images/result.png', cartoon)
        
        
        result = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR))

    
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)
        
    else:

        img = Image.open('./images/your_image.png').convert('RGB') 
        img = np.array(img)
    
  
