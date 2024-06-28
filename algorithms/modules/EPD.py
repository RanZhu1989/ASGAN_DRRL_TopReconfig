from modules import *
from utils.misc import get_time, data_logger, check_cuda
from tqdm import tqdm

class EPD():
    
    def __init__(self,
                 nodes=123,
                 features=24,
                 logger_output_path='./log/',
                 model_saving_path='./model/',
                latent_size=32,
                hidden_size=16,
                epoch_num=50000,
                batch_size=32,
                rp_term=0.1,
                lip_term=0.1,
                norm_term=0.1):
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.device = check_cuda()
        self.latent_size = latent_size
        self.logger_output_path = logger_output_path
        self.model_saving_path = model_saving_path
        self.features = features
        self.nodes = nodes

        # self.dim_latent = self.nodes * time_num * self.batch_size * embed_size  # B*N*T*E

        self.logger = data_logger(self.logger_output_path)
        
        # encoder
        self.encoder = Encoder(
            latent_size=self.latent_size,
            nodes=self.nodes,
            features=self.features
        )
        self.encoder.to(self.device)
        
        self.decoder = Decoder(
            latent_size=self.latent_size,
            nodes=self.nodes,
            features=self.features
        )
        self.decoder.to(self.device)
        self.RP = Reward_Predictor(
            latent_size=self.latent_size,
            hidden_size=hidden_size
        )
        self.RP.to(self.device)
        
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=0.001)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=0.001)
        self.predictor_optimizer = optim.Adam(self.RP.parameters(), lr=0.001)
        
        # Factors
        # self.lambda_term = lambda_term  # w-constraint
        self.rp_term = rp_term  # g rec_loss
        self.lip_term = lip_term  # lip loss <- cp.get_lip()
        self.norm_term = norm_term  # norm loss <- torch.mean(batch_latent**2)
        self.MSELoss = nn.MSELoss()
        
    def train(self, x_data, y_label, save_model=False):
        '''
        x_data: [s, N, T]
        y_label: [s]
        '''
        r = len(x_data)
        log_name = 'Round_' + str(self.round) + '_Losses'
        self.logger.reset_user_label(log_name)
        self.logger.save_to_csv([
            "epoch", "rc_loss", "rp_loss", "lip_loss", "norm_loss"
        ])
        last_rc_loss = 0.
        last_rp_loss = 0.
        last_lip_loss = 0.
        last_norm_loss = 0.
        
        with tqdm(total=self.epoch_num, desc='Training Encoder-Predictor-Decoder Model') as pbar:
            for epoch in range(self.epoch_num):
                for model in [self.encoder, self.decoder, self.RP]:
                    for param in model.parameters():
                        param.requires_grad = True
                
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                self.RP.zero_grad()
                
                # 1. sampling with batch
                # -------------------------------------------
                perm = np.random.permutation(range(r))[:self.batch_size]
                x = x_data[perm]
                y = y_label[perm]
                #------------------------------------------------
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                z = self.encoder.forward(x)
                y1 = self.RP.forward(z).squeeze()
                x_tilde = self.decoder.forward(z)
                
                rp_loss = self.MSELoss(y1, y)
                rp_loss = rp_loss.mean()
                rc_loss = self.MSELoss(x, x_tilde)
                rc_loss = rc_loss.mean()  # mean of batch
                lip_loss = self.RP.get_lip()
                norm_loss = torch.mean(z**2)
                loss = rc_loss + self.rp_term * rp_loss + self.lip_term * lip_loss + self.norm_term * norm_loss
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                self.predictor_optimizer.step()
                
                pbar.set_postfix(
                    {'rc_loss': rc_loss.item(),
                        'rp_loss': rp_loss.item(),  
                        'lip_loss': lip_loss,
                        'norm_loss': norm_loss.item(),
                    })
                pbar.update(1)
                self.logger.save_to_csv([
                    epoch, rc_loss.item(), rp_loss.item(), lip_loss, norm_loss.item()
                ])
                last_rc_loss = rc_loss.item()
                last_rp_loss = rp_loss.item()
                last_lip_loss = lip_loss
                last_norm_loss = norm_loss.item()
                
        x_data = x_data.to(device=self.device)
        latent_all = self.encoder.forward(x_data).detach().cpu().numpy()
        
        # saving model
        if save_model:
            self.save_model()

        return latent_all, last_rc_loss, last_rp_loss, last_lip_loss, last_norm_loss

    
    def adversarial_gen(
            self,
            latent_list,  
            target_drop,  
            num_gen_per_latent=2, 
            lambda_penalty=0.1,  
            eta=0.1,  
            max_num_iter = 20,
            steps=5,
            scaling_factor=1.20  
    ):
        # self.RP.eval()
        new_latent = []
        l2 = torch.nn.MSELoss()
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
        
        # self.G.eval()
        new_env_list = []
        for _ in range(num_gen_per_latent):
            new_env = self.decoder(new_latent)
            new_env = new_env.detach().cpu().numpy()
            new_env_list.append(new_env)
            
        concatenated_envs = np.concatenate(new_env_list, axis=0)
        
        return concatenated_envs
    
    def save_model(self):
        time_str = get_time()
        torch.save(self.encoder.state_dict(), self.model_saving_path / ('Round_' + str(self.round) + '_encoder_' + time_str + '.pth'))
        torch.save(self.decoder.state_dict(),
                   self.model_saving_path / ('Round_' + str(self.round) + '_decoder_' + time_str + '.pth'))
        torch.save(self.RP.state_dict(),
                   self.model_saving_path / ('Round_' + str(self.round) + '_predictor_'+ time_str + '.pth'))

        
    def load_model(self, encoder_path, decoder_path, predictor_path):
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.RP.load_state_dict(torch.load(predictor_path))

    
        
class Encoder(nn.Module):
    def __init__(self,
                 latent_size, 
                 nodes, 
                 features, 
                 hidden_size=32):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(nodes * features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
        self.N = nodes
        self.T = features

    def forward(self, x):
        # x: (B, N, T)
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.N * self.T)  # flatten to (B, N*T)
        latent = self.encoder(x)  # (B, N*T) -> (B, latent_dim)
        return latent

class Decoder(nn.Module):
    def __init__(self,
                 latent_size, 
                 nodes, 
                 features,
                 hidden_size=32):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, nodes * features)
        )
        self.N = nodes
        self.T = features

    def forward(self, z):
        # z: (B, latent_dim)
        batch_size = z.size(0)
        output = self.decoder(z)  # (B, latent_dim) -> (B, N*T)
        output = output.reshape(batch_size, self.N, self.T)  # reshape to (B, N, T)
        return output
    
class Reward_Predictor(nn.Module):

    def __init__(self, latent_size, hidden_size):
        #NOTE You need to calculate the dim_latent
        super(Reward_Predictor, self).__init__()
        self.layers = nn.Sequential(nn.Linear(latent_size, hidden_size),
                                    nn.Sigmoid(), nn.Linear(hidden_size, 1),
                                    nn.Sigmoid())

    def get_lip(self):
        return float(1 / 16 *
                     torch.linalg.norm(self.layers[0].weight.data, ord=2) *
                     torch.linalg.norm(self.layers[2].weight.data,
                                       ord=2))  # each sigmoid is 1/4

    def forward(self, x):
        out = self.layers(x)
        return out
    