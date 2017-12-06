import torch
import torch.optim as optim
import os
from torch.autograd import Variable
import numpy as np
from scipy.misc import imsave

class PGGAN():
    def __init__(self, D, G, data, config):
        self.D = D
        self.G = G
        self.data = data
        self.config = config
        
        self.use_cuda = self.config['gpu_option']
        
        self.batchsize_map = {2**R: self.get_batchsize(2**R) for R in range(2, 11)}
        self.rows_map = {32: 8, 16: 4, 8: 4, 4: 2, 2: 2}
        
    # optimization
    def build_optim(self):
        self.d_optimization = optim.Adam(self.D.parameters(), lr=self.config['learning_rate'], betas=(self.config['beta1'], self.config['beta2']))
        self.g_optimization = optim.Adam(self.G.parameters(), lr=self.config['learning_rate'], betas=(self.config['beta1'], self.config['beta2']))
    
    def get_batchsize(self, resolution):
        R = int(np.log2(resolution))
        if R < 7:
            bs = 32 / 2**(max(0, R-4))
        else:
            bs = 8 / 2**(min(2, R-7))
        return int(bs)
    
    # loss
    def create_criterion(self):
        if self.config['gan'] == 'lsgan':
            self.criterion = lambda logits, label, w: torch.mean((logits-label)**2)
        elif self.config['gan'] == 'wgan':
            self.criterion = lambda logits, label, w: (-2*label+1)*torch.mean(logits)
        elif self.config['gan'] == 'gan':
            self.criterion = lambda logits, label, w: -w*(torch.mean(label*torch.log(logits + 1e-8)) + torch.mean((1-label)*torch.log(1-logits + 1e-8)))
        else:
            raise ValueError('Invalid values:%s'%self.config['gan'])
    def compute_loss(self, logits, label, w):
        return self.criterion(logits, label, w)

    def build_d_loss(self):
        d_real_loss = self.compute_loss(self.d_real, 1, 0.5)
        d_fake_loss = self.compute_loss(self.d_fake, 0, 0.5)
        d_loss = d_real_loss + d_fake_loss
        #self.d_loss = d_loss.data[0] if isinstance(d_loss, Variable) else d_loss
        return d_loss
    
    def build_g_loss(self):
        g_loss = self.compute_loss(self.d_fake, 1, 1)
        return g_loss
    
    # forward
    def build_d_forward(self,cur_level, detach=True):
        self.fake = self.G(self.z, cur_level=cur_level)
        # def forward(self, input_x, input_y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
        self.d_real = self.D(self.input_x, cur_level=cur_level, gdrop_strength=0.0)
        self.d_fake = self.D(self.fake.detach() if detach else self.fake,cur_level=cur_level)
        
    def build_g_forward(self, cur_level):
        # return self.output_layer(input_x, input_y, cur_level, insert_y_at)
        self.d_fake = self.D(self.fake, cur_level=cur_level)
    
    # backward
    def build_d_backward(self, retain_graph=False):
        d_loss = self.build_d_loss()
        self.d_loss = d_loss.data[0] if isinstance(d_loss, Variable) else d_loss
        d_loss.backward(retain_graph=retain_graph)
        self.d_optimization.step()
        
    def build_g_backward(self):
        g_loss = self.build_g_loss()
        self.g_loss = g_loss.data[0] if isinstance(g_loss, Variable) else g_loss
        g_loss.backward()
        self.g_optimization.step()
        
    def train(self):
        self.create_criterion()
        self.build_optim()
        if self.use_cuda:
            self.D.cuda()
            self.G.cuda()
        
        target_level = int(np.log2(self.config['target_r']))
        source_level = int(np.log2(self.config['source_r']))
        
        cur_level = source_level
        for r in range(source_level-1, target_level):
            batch_size = self.batchsize_map[2**(r+1)]
            stablize_imgs = int(self.config['stablize_kimgs']*1000)
            fade_in_imgs =  int(self.config['fade_in_kimgs']*1000)
            
            if r == target_level-1:
                fade_in_imgs = 0
            
            cur_imgs = 0
            iteration = (stablize_imgs+fade_in_imgs)//batch_size
            for ite in range(iteration):
                cur_level = r + float(max(cur_imgs-stablize_imgs,0))/fade_in_imgs
                cur_resolution = int(2**np.ceil(cur_level+1))
                
                train_type = 'stablizing' if cur_level==int(cur_level) else 'fade_in'
                
                z = np.random.normal(0,1,[batch_size, int(self.config['latent_size'])])
                input_x = self.data(batch_size, cur_resolution)
                
                self.z = Variable(torch.from_numpy(z), requires_grad=False).cuda().float()
                self.input_x = Variable(torch.from_numpy(input_x), requires_grad=False).cuda().float()
                
                # optimizing discriminator
                self.d_optimization.zero_grad()
                self.build_d_forward(cur_level, detach=True)
                self.build_d_backward()
                
                # optimizing generator
                self.g_optimization.zero_grad()
                self.build_g_forward(cur_level)
                self.build_g_backward()
                
                cur_imgs = cur_imgs + batch_size

                print('resolution[{}]:iteration[{}/{}]:train_type[{}],g_loss:{:.4f},d_loss:{:.4f}'.format(cur_resolution,ite,iteration,train_type,self.g_loss,self.d_loss))

                # sampling
                if np.mod(ite, self.config['sampling_ite']) == 0:
                    self.sample(os.path.join(self.config['sample_dir'], '{}_{}_{}_{}.png'.format(cur_resolution,cur_resolution, train_type, str(ite).zfill(6))))
                # save model
                if np.mod(ite, self.config['saving_ite']) == 0:
                    self.save_model(os.path.join(self.config['checkpoint_dir'], '{}_{}_{}_{}'.format(cur_resolution,cur_resolution, train_type, str(ite).zfill(6))))

    def save_model(self, filename):
        d_filename = filename + '_D.pth'
        g_filename = filename + '_G.pth'
        torch.save(self.D.state_dict(), d_filename)
        torch.save(self.G.state_dict(), g_filename)
        
    # obtain from pengge' codes
    def sample(self, filename):
        batch_size = self.z.size(0)
        n_row = self.rows_map[batch_size]
        n_col = int(np.ceil(batch_size / float(n_row)))
        white_space = np.ones((self.input_x.size(1), self.input_x.size(2), 3))
        samples = []
        i = j = 0
        for row in range(n_row):
            one_row = []
            # fake
            for col in range(n_col):
                one_row.append(self.fake[i].cpu().data.numpy())
                one_row.append(white_space)
                i += 1
            one_row.append(white_space)
            # real
            for col in range(n_col):
                one_row.append(self.input_x[j].cpu().data.numpy())
                if col < n_col-1:
                    one_row.append(white_space)
                j += 1
            samples += [np.concatenate(one_row, axis=2)]
        samples = np.concatenate(samples, axis=1).transpose([1, 2, 0])
        imsave(filename, samples)

    def load_model(self):
        pass
