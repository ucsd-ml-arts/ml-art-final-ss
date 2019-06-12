import torch
import numpy as np
from torch.autograd import Variable
from AttnGAN.miscc.config import cfg
from AttnGAN.miscc.utils import build_super_images2
from AttnGAN.model import RNN_ENCODER, G_NET
import matplotlib.pyplot as plt
import pickle

class TextToImage(object):
    def __init__(self, captionsPath='./AttnGAN/data/captions.pickle'):
        # import dictionaries for use
        x = pickle.load(open('./AttnGAN/data/captions.pickle', 'rb'))
        self.ixtoword = x[2]
        self.wordtoix = x[3]
        del x

        # generate model for use
        word_len = len(self.wordtoix)
        self.text_encoder = RNN_ENCODER(word_len, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        self.text_encoder.eval()
        
        self.netG = G_NET()
        state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(state_dict)
        self.netG.eval()

    def vectorize_caption(self, caption, copies=2):
        # create caption vector
        tokens = caption.split(' ')
        cap_v = []
        for t in tokens:
            t = t.strip().encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in self.wordtoix:
                cap_v.append(self.wordtoix[t])

        # expected state for single generation
        captions = np.zeros((copies, len(cap_v)))
        for i in range(copies):
            captions[i,:] = np.array(cap_v)
        cap_lens = np.zeros(copies) + len(cap_v)

        return captions.astype(int), cap_lens.astype(int)

    def generate(self, caption, copies=2, plotExample=False):
        # load word vector
        captions, cap_lens  = self.vectorize_caption(caption, copies)
        n_words = len(self.wordtoix)

        # only one to generate
        batch_size = captions.shape[0]

        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions))
        cap_lens = Variable(torch.from_numpy(cap_lens))
        noise = Variable(torch.FloatTensor(batch_size, nz))

        #######################################################
        # (1) Extract text embeddings
        #######################################################
        hidden = self.text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = self.text_encoder(captions.to(torch.long), cap_lens, hidden)
        mask = (captions == 0)
            

        #######################################################
        # (2) Generate fake images
        #######################################################
        noise.data.normal_(0, 1)
        fake_imgs, attention_maps, _, _ = self.netG(noise, sent_emb, words_embs, mask)

        # G attention
        cap_lens_np = cap_lens.cpu().data.numpy()

        imgs = []
        for j in range(batch_size):
            for k in range(len(fake_imgs)):
                im = fake_imgs[k][j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                imgs.append(im)
        
        # Plot an example if requested
        if(plotExample):
            plt.figure()
            plt.imshow(imgs[-1])
            plt.title('Example Image')
        
        return imgs