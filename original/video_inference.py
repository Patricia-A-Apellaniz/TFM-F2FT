import torch
import cv2
from matplotlib import pyplot as plt

from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
from webcam_demo.webcam_extraction_conversion import *

"""Init"""

#Paths
#path_to_model_weights = 'model_weights.tar'
path_to_model_weights = "finetuned_model.tar"
path_to_embedding = 'e_hat_video.tar'
path_to_mp4 = 'Pati_4.MOV'

device = torch.device("cuda:0")
cpu = torch.device("cpu")

print("Loading checkpoint...")
checkpoint = torch.load(path_to_model_weights, map_location=cpu) 
e_hat = torch.load(path_to_embedding, map_location=cpu)
e_hat = e_hat['e_hat'].to(device)

print("Initializing generator...")
G = Generator(256, finetuning=True, e_finetuning=None)
G.eval()

"""Training Init"""
G.load_state_dict(checkpoint['G_state_dict'])
G.to(device)


"""Main"""
print("Loading video...")
cap = cv2.VideoCapture(path_to_mp4)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
ret = True
i = 0
size = (256*3,256)
video = cv2.VideoWriter('video_generated.avi',cv2.VideoWriter_fourcc(*'XVID'), fps, size)

with torch.no_grad():
    while ret:
        x, g_y, ret = generate_landmarks(cap=cap, device=device, pad=50)

        if ret:
	        g_y = g_y.unsqueeze(0)
	        x = x.unsqueeze(0)

	        x_hat = G(g_y, e_hat)

	        plt.clf()
	        out1 = x_hat.transpose(1,3)[0]/255
	        out1 = out1.to(cpu).numpy()
	        #plt.imshow(out1)
	        #plt.show()
	        
	        #plt.clf()
	        out2 = x.transpose(1,3)[0]/255
	        out2 = out2.to(cpu).numpy()
	        #plt.imshow(out2)
	        #plt.show()

	        #plt.clf()
	        out3 = g_y.transpose(1,3)[0]/255
	        out3 = out3.to(cpu).numpy()
	        #plt.imshow(out3)
	        #plt.show()

	        me = cv2.cvtColor(out2*255, cv2.COLOR_BGR2RGB)
	        landmark = cv2.cvtColor(out3*255, cv2.COLOR_BGR2RGB)
	        fake = cv2.cvtColor(out1*255, cv2.COLOR_BGR2RGB)

	        img = np.concatenate((me, landmark, fake), axis=1)
	        img = img.astype("uint8")
	        video.write(img)

	        i+=1
	        print(i,'/',n_frames)

cap.release()
video.release()
cv2.destroyAllWindows()