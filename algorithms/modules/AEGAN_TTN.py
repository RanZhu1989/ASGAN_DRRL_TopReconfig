from modules import *

from utils.misc import get_time, data_logger, check_cuda

from tqdm import tqdm
'''
Adversarial Environment - GAN

This model is similar to VAE-GAN, but the latent space is learned by a pre-encoder.

'''


class AE_GAN:

    def __init__(
            self,
            nodes=123,
            features=24, 
            time_num=24, 
            logger_output_path='./log/',  
            model_saving_path='./model/', 
            embed_size=32,
            hidden_size=20,
            lambda_term=10.,
            rc_term=0.1,
            epoch_num=10000,
            d_iters=1,
            g_iters=3,
            rp_iters=10000,
            batch_size=32,
            lip_term=0.1,
            norm_term=0.1):
        self.epoch_num = epoch_num
        self.d_iters = d_iters
        self.g_iters = g_iters
        self.rp_iters = rp_iters
        self.batch_size = batch_size
        self.device = check_cuda()
        self.embed_size = embed_size
        self.logger_output_path = logger_output_path
        self.model_saving_path = model_saving_path
        self.features = features
        self.nodes = nodes

        # self.dim_latent = self.nodes * time_num * self.batch_size * embed_size  # B*N*T*E

        self.logger = data_logger(self.logger_output_path)
        
        # Generator
        self.G = TTransformer_G(
                                embed_size=embed_size,  
                                num_layers=2,  # TTN blocks number
                                heads=1,  # TTN heads
                                time_num=time_num,  
                                temporal_embedding=True,
                                device=self.device,
        )

        self.G.to(device=self.device)

        # Discriminator
        self.D = TTransformer_D(nodes=self.nodes,
                                features=features,
                                embed_size=embed_size,
                                num_layers=1,
                                heads=1,
                                time_num=time_num,
                                temporal_embedding=True,
                                device=self.device)

        self.D.to(device=self.device)
        # print('is paralleled ==', args.parallel)

        self.PE = Pre_Encoder(embed_size=embed_size,
                              num_layers=1,
                              heads=1,
                              time_num=time_num,
                              forward_expansion=4,
                              dropout=0,
                              temporal_embedding=True,
                              device=self.device)
        self.PE.to(device=self.device)
        # if args.parallel:
        # self.PE = nn.DataParallel(self.PE, device_ids=range(torch.cuda.device_count()))
        # self.PE = torch.load(args.save_path + 'pre_encoder.pkl')

        self.RP = Reward_Predictor(embed_size=self.embed_size,
                                  hidden_size=hidden_size)
        self.RP.to(device=self.device)
        # if args.parallel:
        # self.CP = nn.DataParallel(self.CP, device_ids=range(torch.cuda.device_count()))
        # self.CP = torch.load(args.save_path + 'cost_predictor.pkl')

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=0.0001)
        # train pre-encoder, cost predictor and generator in the g-training process
        self.g_optimizer = optim.Adam(
            [
            {'params': self.PE.parameters(),
            'lr': 0.0001}, 
            {'params': self.G.parameters(),
            'lr': 0.0001}
            ])
        self.rp_optimizer = optim.Adam(self.RP.parameters(), lr=0.001)

        # training process
        # Factors
        self.lambda_term = lambda_term  # w-constraint
        self.rc_term = rc_term  # g rec_loss
        self.lip_term = lip_term  # lip loss <- cp.get_lip()
        self.norm_term = norm_term  # norm loss <- torch.mean(batch_latent**2)

        # GAN reconstruction loss
        self.MSELoss = nn.MSELoss()

    def noise_generator(self, z_size):
        z = torch.randn(size=z_size)
        z = z.to(device=self.device)
        return z

    def train(self, x_data, y_label, save_model=False):
        '''
        x_data: [s, N, T]
        y_label: [s]
        '''
        r = len(x_data)
        
        log_name = 'Round_' + str(self.round) + '_GAN_Losses'
        self.logger.reset_user_label(log_name)
        self.logger.save_to_csv([
            "epoch", "d_loss", "g_loss", "rc_loss", "norm_loss"
        ])
        last_d_loss = 0.
        last_g_loss = 0.
        last_rc_loss = 0.
        last_lip_loss = 0.
        last_norm_loss = 0.
        last_rp_loss = 0.
        '''
        Algorithm 1 in the paper
        '''
        with tqdm(total=self.epoch_num, desc='Training GAN') as pbar:
            for epoch in range(self.epoch_num):
                # Requires grad, Generator requires_grad = False
                for p in self.D.parameters():
                    p.requires_grad = True
                for model in [self.PE, self.RP, self.G]:
                    for param in model.parameters():
                        param.requires_grad = False
                '''
                    #1. update discriminator 
                    
                    The goal of the discriminator is:
                    max(D(x)) - min(D(G(z))) + lambda*(||grad(D(x))|| - 1)^2
                    The first two terms are the same as the original GAN
                    The third term is the gradient penalty for the Lipschitz constraint of D. (please refer to WGAN-GP paper)
                '''
                d_loss = 0.
                for d_iter in range(self.d_iters):
                    # Train Discriminator
                    self.D.zero_grad()

                    # 1. sampling with batch
                    # -------------------------------------------
                    perm = np.random.permutation(range(r))[:self.batch_size]
                    x = x_data[perm]
                    # y = y_label[perm]
                    #------------------------------------------------
                    x = x.to(device=self.device)
                    # y = y.to(device=self.device)

                    # 2. feed into pre-encoder
                    x1 = self.PE.forward(x)

                    # 3. add noise
                    # z_size = [B, N, T, E]
                    z = self.noise_generator([
                        self.batch_size, self.nodes, self.features, self.embed_size
                    ])

                    # 4. generate samples
                    x_tilde = self.G.forward(x1 + z)

                    # backward loss
                    # loss of real samples
                    d_loss_real = self.D.forward(x)
                    d_loss_real = d_loss_real.mean()

                    # loss of fake samples
                    d_loss_fake = self.D.forward(x_tilde)
                    d_loss_fake = d_loss_fake.mean()

                    # gradient penalty
                    # with torch.backends.cudnn.flags(enabled=False):
                    gp = self.calculate_gradient_penalty(x, x_tilde)
                    gp = gp.mean()

                    d_loss = d_loss_fake - d_loss_real + gp
                    d_loss.backward()

                    self.d_optimizer.step()
                '''
                    #2. update generator
                    The goal of the generator is:
                    min{  -D(G(z)) 
                    + lambda1 * ||x - G(PE(x) + z)||^2 
                    + lambda2 * ||RP(PE(x)) - y||^2 
                    + lambda3 * Lipchitz constraint of Reward Predictor Network
                    + lambda4 * ||PE(x)||^2
                    }
                    The first term is the same as the original GAN
                    The second term is the reconstruction loss of the whole G (including PE and G)
                    The third term is for Lipschitz constraint of RP
                    The fourth term is the norm loss of PE     
                '''
                last_d_loss = d_loss.item()
                for model in [self.D, self.RP]:
                    for param in model.parameters():
                        param.requires_grad = False
                for model in [self.PE, self.G]:
                    for param in model.parameters():
                        param.requires_grad = True

                g_loss = 0.
                rc_loss = 0.
                for g_iter in range(self.g_iters):
                    # Train Generator
                    self.G.zero_grad()

                    # train generator
                    # 1. sampling with batch
                    perm = np.random.permutation(range(r))[:self.batch_size]
                    x = x_data[perm]
                    x = x.to(device=self.device)

                    # 2. feed into pre-encoder
                    x1 = self.PE.forward(x)

                    # 3. latent feed into cost predictor
                    # y1 = self.RP.forward(x1)

                    # 4. add noise
                    # noise term
                    z = self.noise_generator([
                        self.batch_size, self.nodes, self.features, self.embed_size
                    ])

                    # 5. generate samples
                    x_tilde = self.G.forward(x1 + z)  # the generated samples

                    # backward loss
                    gd_loss = self.D.forward(x_tilde)
                    gd_loss = gd_loss.mean()

                    # # MMD loss
                    # mmd_loss = self.calculate_maximum_mean_discrepancy(x, x_tilde)
                    # rc_loss = mmd_loss.mean()

                    # # MSE loss
                    rc_loss = self.MSELoss(x, x_tilde)
                    rc_loss = rc_loss.mean()  # mean of batch

                    # -------losses---------
                    grc_loss = self.rc_term * rc_loss - gd_loss
                    norm_loss = torch.mean(x1**2)
                    # total loss
                    g_loss = grc_loss + self.norm_term * norm_loss
                    g_loss.backward()

                    self.g_optimizer.step()

                pbar.set_postfix(
                    {'d_loss': d_loss.item(),
                        'g_loss': gd_loss.item(),  
                        'rc_loss': rc_loss.item(),
                        'norm_loss': norm_loss.item(),
                    })
                pbar.update(1)
                self.logger.save_to_csv([
                    epoch, d_loss.item(), gd_loss.item(), rc_loss.item(), norm_loss.item()
                ])
                last_g_loss = gd_loss.item()
                last_rc_loss = rc_loss.item()
                last_norm_loss = norm_loss.item()
            
        # Train predictor
        for p in self.RP.parameters():
            p.requires_grad = True
        for model in [self.D, self.PE, self.G]:
            for param in model.parameters():
                param.requires_grad = False
        
        log_name = 'Round_' + str(self.round) + '_RP_Losses'
        self.logger.reset_user_label(log_name)
        self.logger.save_to_csv([
            "epoch", "rp_loss", "lip_loss"
            ])
        
        with tqdm(total=self.rp_iters, desc='Training Reward Predictor') as pbar:
            for epoch in range(self.rp_iters):
                self.rp_optimizer.zero_grad()
                perm = np.random.permutation(range(r))[:self.batch_size]
                x = x_data[perm]
                y = y_label[perm]
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                x1 = self.PE.forward(x)
                y1 = self.RP.forward(x1)
                rp_loss = self.MSELoss(y1, y)
                rp_loss = rp_loss.mean()
                lip_loss = self.RP.get_lip()
                p_loss = rp_loss + self.lip_term * lip_loss
                p_loss.backward()
                self.rp_optimizer.step()  
                pbar.set_postfix(
                    {'rp_loss': rp_loss.item(),
                     'lip_loss': lip_loss
                    })
                pbar.update(1)
                self.logger.save_to_csv([
                    epoch, rp_loss.item(), lip_loss
                ])
        
        last_rp_loss = rp_loss.item() 
        last_lip_loss = lip_loss   
                
        # Obtain the latent
        x_data = x_data.to(device=self.device)
        latent_all = self.PE.forward(x_data).detach().cpu().numpy()
        
        # saving model
        if save_model:
            self.save_model()

        return latent_all, last_d_loss, last_g_loss, last_rc_loss, last_lip_loss, last_norm_loss, last_rp_loss

    def adversarial_gen(
            self,
            latent_list,  # latent s*[N, T, E] -> list
            target_drop,  # # The goal of gradient descent to change the proportion of quantities
            num_gen_per_latent=2,  # # How many times each new_latent is sent into G
            lambda_penalty=0.1,  # Penalty for d(z,z0)
            eta=0.1,  # step size
            max_num_iter = 20,
            steps=5,
            scaling_factor=1.20  # # Estimated correction factor for changing the proportion of quantities through gradient descent
    ):
        '''
        Generate adversarial samples via gradient descent  of latent variables
        
        new_latent = old_latent + eta * grad_new_latent[cost_pred(new_latent) - lambda*||old_latent - new_latent||^2]
		* here new_latent is a tensor and requires grad, old_latent is a static tensor (detach)
        
        1. Obtain new_latent through CP gradient descent
        2. Generate new samples by feeding new_latent to G (the number of samples generated per latent can be set)
        
        Output.shape = [s, N, T]
            
        '''
        #TODO Currently, only one sample can be generated from one latent at a time
        # self.RP.eval()
        new_latent = []
        l2 = torch.nn.MSELoss()
        # 1. Perform gradient descent for each latent
        for idx, latent in enumerate(tqdm(latent_list,desc='Adversarial Generation')):
            latent = torch.tensor(latent, dtype=torch.float32, device=self.device, requires_grad=True)
            latent_detach = latent.detach().unsqueeze(0)
            for idx1 in range(max_num_iter):
                eta_env = eta
                lambda_env = lambda_penalty
                latent_adv = latent.clone().unsqueeze(0) # add batch dim
                ini_pred_reward = self.RP.forward(latent_adv)
                                
                for _ in range(steps):
                    pred_reward = self.RP.forward(latent_adv)
                    #NOTE: Cost = -Reward
                    loss = -pred_reward - lambda_env * l2(latent_adv, latent_detach)
                    grad = torch.autograd.grad(loss, latent_adv)[0]
                    latent_adv += eta_env * grad
                    pass
                    
                if (ini_pred_reward-pred_reward) > target_drop * scaling_factor:
                    eta *= 0.9
                    lambda_penalty *= 1.25
                elif (ini_pred_reward-pred_reward) > target_drop:
                    break
                else:
                    eta *= 1.1
                    lambda_penalty *= 0.75
                    
            new_latent.append(latent_adv.detach().cpu().numpy().squeeze(0)) # remove batch dim
        
        new_latent = np.array(new_latent)    
        new_latent = torch.tensor(new_latent, dtype=torch.float32, device=self.device)
        
        # 2. The obtained new_latent is fed to G to generate new samples
        # self.G.eval()
        new_env_list = []
        for _ in range(num_gen_per_latent):
            z = self.noise_generator([
                        new_latent.shape[0], self.nodes, self.features, self.embed_size
                    ])
            new_env = self.G(new_latent+z)
            new_env = new_env.detach().cpu().numpy()
            new_env_list.append(new_env)
            
        concatenated_envs = np.concatenate(new_env_list, axis=0)
        return concatenated_envs
                    

    def calculate_gradient_penalty(self, x, x_tilde):
        # linear interpolation
        eps = torch.rand(size=[self.nodes, self.features])
        eps = eps.expand([self.batch_size, self.nodes, self.features])
        eps = eps.to(device=self.device)
        interpolated = eps * x + ((1 - eps) * x_tilde)

        # calculate probability of interpolated examples
        interpolated = Variable(interpolated, requires_grad=True)
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        ones = torch.ones(size=prob_interpolated.size()).to(device=self.device)
        gradients = autograd.grad(outputs=prob_interpolated,
                                  inputs=interpolated,
                                  grad_outputs=ones,
                                  create_graph=True,
                                  retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1)**2) * self.lambda_term
        return grad_penalty

    def save_model(self):
        time_str = get_time()
        torch.save(self.G.state_dict(), self.model_saving_path / ('Round_' + str(self.round) + '_generator_' + time_str + '.pth'))
        torch.save(self.D.state_dict(),
                   self.model_saving_path / ('Round_' + str(self.round) + '_discriminator_' + time_str + '.pth'))
        torch.save(self.PE.state_dict(),
                   self.model_saving_path / ('Round_' + str(self.round) + '_pre_encoder_'+ time_str + '.pth'))
        torch.save(self.RP.state_dict(),
                   self.model_saving_path / ('Round_' + str(self.round) + '_reward_predictor_' + time_str + '.pth'))
        
    def load_model(self, D_path, G_path, PE_path, RP_path):
        self.D.load_state_dict(torch.load(D_path))
        self.G.load_state_dict(torch.load(G_path))
        self.PE.load_state_dict(torch.load(PE_path))
        self.RP.load_state_dict(torch.load(RP_path))
        
        
class Reward_Predictor(nn.Module):

    def __init__(self, embed_size, hidden_size):
        #NOTE You need to calculate the dim_latent
        super(Reward_Predictor, self).__init__()
        self.layers = nn.Sequential(nn.Linear(embed_size, hidden_size),
                                    nn.Sigmoid(), nn.Linear(hidden_size, 1),
                                    nn.Sigmoid())

    def get_lip(self):
        return float(1 / 16 * (1/(self.N * self.T)) *
                     torch.linalg.norm(self.layers[0].weight.data, ord=2) *
                     torch.linalg.norm(self.layers[2].weight.data,
                                       ord=2))  # each sigmoid is 1/4

    def forward(self, x):
        x = self.layers(x)
        self.N = x.shape[1]
        self.T = x.shape[2]
        out = torch.mean(x, dim=(1, 2, 3))
        return out


'''
Modules for transformer-based networks


T self-attention--> T Transformer --> T-TransformerBlock --> Encoder --> T-Transformer_G/D
                                                                |
                                                                |--> Pre-encoder

'''


class TSelfAttention(nn.Module):

    def __init__(self, embed_size, heads):
        super(TSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size
               ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        B, N, T, C = query.shape

        # Split the embedding into self.heads different pieces
        values = values.reshape(B, N, T, self.heads,
                                self.head_dim)  # embed_size => heads×head_dim
        keys = keys.reshape(B, N, T, self.heads, self.head_dim)
        query = query.reshape(B, N, T, self.heads, self.head_dim)

        values = self.values(values)  # (N, T, heads, head_dim)
        keys = self.keys(keys)  # (N, T, heads, head_dim)
        queries = self.queries(query)  # (N, T, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        energy = torch.einsum("bnqhd, bnkhd -> bnqkh",
                              [queries, keys])  # self-attention
        # time在其他维度上对应元素相乘
        # queries shape: (B, N, T, heads, heads_dim),
        # keys shape: (B, N, T, heads, heads_dim)
        #NOTE energy: (B, N, T, T, heads)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size**(1 / 2)), dim=2)
        # attention shape: (N, query_len, key_len, heads)

        out = torch.einsum("bnqkh, bnkhd -> bnqhd",
                           [attention, values]).reshape(
                               B, N, T, self.heads * self.head_dim)
        # attention shape: (N, T, T, heads)
        # values shape: (N, T, heads, heads_dim)
        # out after matrix multiply: (N, T, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, T, embed_size)

        return out


class TTransformer(nn.Module):

    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion, temporal_embedding,
                 device):
        super(TTransformer, self).__init__()

        # Temporal embedding
        self.device = device
        self.time_num = time_num
        self.temporal_embedding_enable = temporal_embedding
        if self.temporal_embedding_enable:
            self.temporal_embedding = nn.Embedding(time_num,
                                                embed_size)  # temporal embedding
        self.attention = TSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        B, N, T, C = value.shape
        #NOTE [T, E]
        if self.temporal_embedding_enable:
            D_T = self.temporal_embedding(torch.arange(
                0, T, device=self.device))  # temporal embedding
            # D_T = self.temporal_embedding(torch.arange(0, T))
            # temporal embedding
            value = value + D_T  #NOTE broadcast: [B, N, T, E] + [T, E] => [B, N, T, E]
            
        attention = self.attention(value, value, value)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + value))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class TTransformerBlock(nn.Module):

    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion, temporal_embedding, 
                 device):
        super(TTransformerBlock, self).__init__()
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout,
                                         forward_expansion, temporal_embedding, device)
        self.merge = nn.Linear(2 * embed_size, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        # Add skip connection,run through normalization and finally dropout
        x1 = self.TTransformer(value)
        # x2 = self.TTransformer(value)

        out = self.merge(torch.cat(
            [x1, value], dim=3))  #NOTE torch.cat([x1, value] skip connection
        return out


class TT_Encoder(nn.Module):
    # STTransformer Block
    def __init__(self, embed_size, num_layers, heads, time_num,
                 forward_expansion, dropout, temporal_embedding, device):
        super(TT_Encoder, self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList([
            TTransformerBlock(embed_size,
                              heads,
                              time_num,
                              forward_expansion=forward_expansion,
                              dropout=dropout,
                              temporal_embedding=temporal_embedding,
                              device=device) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(x)

        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out)

        return out


class Pre_Encoder(nn.Module):
    '''
    input [B, N, T] (ENV feature)
    output [B, N, T, E] (latent Z)
    '''

    def __init__(self,
                 in_channels=1,
                 embed_size=32,
                 num_layers=2,
                 heads=4,
                 time_num=24,
                 forward_expansion=4,
                 dropout=0,
                 temporal_embedding=True,
                 device='cpu'):
        super(Pre_Encoder, self).__init__()
        self.encoder = TT_Encoder(embed_size,
                               num_layers,
                               heads,
                               time_num,
                               forward_expansion,
                               dropout,
                               temporal_embedding,
                               device=device)
        # linear
        self.linear1 = nn.Linear(in_channels, embed_size)
        # self.linear2 = nn.Linear(embed_size, in_channels)

    def forward(self, x):

        x = x.unsqueeze(dim=3)
        # [B, N, T, C]
        x = self.linear1(x)
        # [B, N, T, E]
        out = self.encoder(x)
        # out = self.linear2(enc)
        out = out.squeeze(dim=3)

        return out


class TTransformer_G(nn.Module):
    '''
    input [B, N, T, E] (latent Z + Noise)
    output [B, N, T ] (ENV feature)
    '''

    def __init__(self,
                 in_channels=1,
                 embed_size=32,
                 num_layers=3,
                 heads=4,
                 time_num=24,
                 forward_expansion=4,
                 dropout=0,
                 temporal_embedding=True,
                 device='cpu'):
        super(TTransformer_G, self).__init__()
        self.encoder = TT_Encoder(embed_size,
                               num_layers,
                               heads,
                               time_num,
                               forward_expansion,
                               dropout,
                               temporal_embedding,
                               device=device)

        # linear
        # self.linear1 = nn.Linear(in_channels, embed_size)
        self.linear2 = nn.Linear(embed_size, in_channels)

    def forward(self, x):

        # x = x.unsqueeze(dim=3)
        # [B, N, T, C]
        # x = self.linear1(x)
        # [B, N, T, E]
        enc = self.encoder(x)
        out = self.linear2(enc)
        out = out.squeeze(dim=3)
        out = torch.sigmoid(out)

        return out


class TTransformer_D(nn.Module):
    '''
    input [B, N, T]
    output [B, 1]
    '''

    def __init__(self,
                 in_channels=1,
                 nodes=123,
                 features=24,
                 embed_size=32,
                 num_layers=1,
                 heads=4,
                 time_num=24,
                 forward_expansion=4,
                 dropout=0,
                 temporal_embedding=True,
                 device='cpu'):
        super(TTransformer_D, self).__init__()

        self.encoder = TT_Encoder(embed_size,
                               num_layers,
                               heads,
                               time_num,
                               forward_expansion,
                               dropout,
                               temporal_embedding,
                               device=device)

        # 1x1 convolution
        self.linear1 = nn.Linear(in_channels, embed_size)
        self.critic = nn.Sequential(nn.Linear(embed_size,
                                              in_channels), nn.Flatten(),
                                    nn.Linear(nodes * features, 1),
                                    nn.Sigmoid())

    def forward(self, x):
        x = x.unsqueeze(dim=3)
        # [B, N, T, C]
        x = self.linear1(x)
        enc = self.encoder(x)
        out = self.critic(enc)

        return out