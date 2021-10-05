# TFM-F2FT

Throughout this Master's Thesis, a study of the state of the art in the different techniques that exist for 
the translation of facial gestures between audiovisual media will be carried out. Specifically, it will be 
focused on image animation based on facial expressions recorded on video.

From this study, a system based on convolutional networks will be implemented, in particular based on GANs, 
Generative Adversarial Networks, which will train using the VoxCeleb2 dataset, which has been pre-processed 
to acquire the audiovisual features to be used as input to the network. As visual features, image reference 
points also known as landmarks, will be used and as audio features, MFCCs, Mel Frequency Cepstral Coefficients, 
and the sequence spectrogram. To develop this network, to train it and to later evaluate it, several Python libraries 
and frameworks will be used, highlighting PyTorch against others of expanded use in this area as TensorFlow.

Taking advantage of the individual audio and video channels in a single sequence, the aim is therefore to obtain 
a system that generates clear and realistic movement patterns or facial expression in originally static images.
