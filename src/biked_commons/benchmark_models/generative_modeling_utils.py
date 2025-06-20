from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from torch.autograd import grad
from diffusers import DDPMScheduler
from torch.nn import MSELoss


from biked_commons.conditioning import conditioning
from biked_commons.design_evaluation.design_evaluation import *
from biked_commons.conditioning import conditioning
from biked_commons.design_evaluation import scoring

class TorchScaler:
    def __init__(self, data):
        self.data = data
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def scale(self, x):
        return (x - self.mean) / self.std

    def unscale(self, x):
        return x * self.std + self.mean

def sample_continuous(num_samples, split="test", randomize = False):
    emb = conditioning.sample_image_embedding(num_samples, split, randomize)
    rider = conditioning.sample_riders(num_samples, split, randomize)
    use_case = conditioning.sample_use_case(num_samples, split, randomize)
    all = torch.cat((emb, rider, use_case), dim=1)
    return all



def parse_continuous_condition(condition):
    image_embeddings = condition[:, :512]
    use_case_condition = condition[:, -3:]
    rider_condition = condition[:, 512:-3]
    condition = {"Rider": rider_condition, "Use Case": use_case_condition, "Embedding": image_embeddings}
    return condition

def piecewise_constraint_score(constraint_scores, constraint_falloff = 10):
    constraint_scores_safexp = torch.clamp(constraint_scores, max=0.0)
    piece1 = torch.exp(constraint_scores_safexp * constraint_falloff)/constraint_falloff
    piece2 = constraint_scores + 1/constraint_falloff
    mask = constraint_scores < 0.0
    mask = mask.float()
    result = piece1 * mask + piece2 * (1 - mask)
    return result

def get_composite_score_fn(scaler, columns, constrant_vs_objective_weight = 10.0, constraint_falloff=10.0, device="cpu"):
    evaluator, requirement_names, requirement_types = construct_tensor_evaluator(get_standard_evaluations(device), columns)

    isobjective = torch.tensor(requirement_types) == 1

    weights = scoring.get_ref_point(evaluator, requirement_names, requirement_names, reduction="meanabs", device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    assert weights.min() > 0, "Ref point should be greater than 0"

    def composite_score_fn(x, continuous_condition, evaluator = evaluator, scaler = scaler):
        #print if there are any NaN values in x
        if torch.isnan(x).any():
            print("NaN values in x")
        x = scaler.unscale(x)
        condition = parse_continuous_condition(continuous_condition)
        eval_scores = evaluator(x, condition)
        scaled_scores = eval_scores / weights
        objective_scores = scaled_scores[:, isobjective]
        constraint_scores_raw = scaled_scores[:, ~isobjective]
        constraint_scores = piecewise_constraint_score(constraint_scores_raw, constraint_falloff)
        total_scores = torch.sum(objective_scores, dim=1) + torch.sum(constraint_scores, dim=1) * constrant_vs_objective_weight
        composite_scores = total_scores / constrant_vs_objective_weight

        quality_scores = 1/composite_scores
        # print("Quality scores: ", quality_scores)
        if torch.isnan(quality_scores).any():
            print("NaN values in quality scores")

        mean_comp_scores = torch.mean(composite_scores)
        constraint_satisfaction_rate = torch.mean(torch.all(constraint_scores_raw <= 0, dim=1).float())
        report = {"CSR": constraint_satisfaction_rate, "MCS": mean_comp_scores}
        return quality_scores, report

    return composite_score_fn

def get_uneven_batch_sizes(total_data_points, batch_size):
    """
    Given the total number of data points and a target batch size, 
    returns a list of batch sizes that sum up to the total number of data points.
    The batch sizes are distributed as evenly as possible but may be uneven.

    :param total_data_points: Total number of data points (int).
    :param batch_size: Target batch size (int).
    :return: A list of batch sizes (list of int).
    """
    # Calculate the number of batches needed
    num_batches = total_data_points // batch_size
    remainder = total_data_points % batch_size

    # Initialize the batch sizes
    batch_sizes = [batch_size] * num_batches

    # Distribute the remainder across the batches
    for i in range(remainder):
        batch_sizes[i%num_batches] += 1

    return batch_sizes


def get_diversity_loss_fn(scaler:TorchScaler, columns, diversity_weight=0.1, score_weight=0.1, constraint_vs_objective_weight=10.0, constraint_falloff=10.0, dpp_batch=16, device="cpu"):
    composite_score_fn = get_composite_score_fn(scaler, columns, constrant_vs_objective_weight = constraint_vs_objective_weight, constraint_falloff = constraint_falloff, device=device)

    def diversity_loss_fn(x, condition, diversity_weight=diversity_weight, score_weight=score_weight):
        if diversity_weight == 0:
            return torch.tensor(0.0), {"DIV-OFF": 0.0}
        
        scores, report= composite_score_fn(x, condition)

        # Initialize the total loss
        total_loss = 0.0

        # Get uneven batch sizes based on the total number of data points
        batch_sizes = get_uneven_batch_sizes(x.size(0), dpp_batch)

        # Split the data into uneven batches
        start_idx = 0
        for batch_size in batch_sizes:
            # Get the current batch
            end_idx = start_idx + batch_size
            x_batch = x[start_idx:end_idx]
            scores_batch = scores[start_idx:end_idx]
            # Compute pairwise squared Euclidean distances for the batch
            r = torch.sum(x_batch ** 2, dim=1, keepdim=True)
            D = r - 2 * torch.matmul(x_batch, x_batch.T) + r.T
            D_norm = D / x_batch.size(1)  # Normalize by the number of features
            # Compute the similarity matrix using RBF for the batch
            S = torch.exp(-0.5 * D_norm ** 2) / 2

            # Compute the quality matrix for the batch
            Q = torch.matmul(scores_batch, scores_batch.T)
            Q = torch.pow(Q, score_weight)
            L = S * Q

            L = (L + L.T) / 2.0

            L_stable = L + 1e-6 * torch.eye(L.size(0), device=L.device)  

            # Compute the eigenvalues of the similarity matrix for the batch
            try:
                eig_val = torch.linalg.eigvalsh(L_stable)
            except:
                print(f"Eigenvalue computation failed for batch with size {batch_size}")
                eig_val = torch.ones(x_batch.size(0), device=x.device)
            if torch.isnan(eig_val).any():
                print("NaNs detected in eig_val")
            # if (eig_val <= 0).any():
            #     print("Nonpositive eigenvalues:", eig_val)
            # Compute the loss for the batch as the negative mean log of the eigenvalues
            if torch.isinf(torch.log(eig_val)).any():
                print("Log produced inf! Min/max eig_val:", eig_val.min().item(), eig_val.max().item())

            batch_loss = -torch.mean(torch.log(torch.clamp(eig_val, min=1e-6, max=1e6)))

            total_loss += batch_loss

            # Update the start index for the next batch
            start_idx = end_idx
        # Compute the final loss by averaging across batches
        loss = total_loss / len(batch_sizes)* diversity_weight
        return loss, report
    return diversity_loss_fn

class Down_Model(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=400, num_hidden_layers=2):
        super(Down_Model, self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.LeakyReLU()])
        
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

class Up_Model(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=400, num_hidden_layers=2):
        super(Up_Model, self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.LeakyReLU()])
        
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x



def GAN_step(D, G, D_opt, G_opt, data_batch, cond_batch, noise_batch, batch_size, device, auxiliary_loss_fn):
    criterion = nn.BCEWithLogitsLoss()
    D.zero_grad()
    real_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
    fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

    data_and_condition = torch.cat([data_batch, cond_batch], dim=1)
    noise_and_condition = torch.cat([noise_batch, cond_batch], dim=1)

    output = D(data_and_condition).view(-1)
    L_D_real = criterion(output, real_label)

    fake_data = G(noise_and_condition)
    fake_data_and_condition = torch.cat([fake_data, cond_batch], dim=1)
    output = D(fake_data_and_condition.detach()).view(-1)
    L_D_fake = criterion(output, fake_label)

    L_D_tot = L_D_real + L_D_fake
    L_D_tot.backward()
    D_opt.step()

    G.zero_grad()
    fake_data = G(noise_and_condition)
    fake_data_and_condition = torch.cat([fake_data, cond_batch], dim=1)
    output = D(fake_data_and_condition).view(-1)
    L_G = criterion(output, real_label)

    if auxiliary_loss_fn is not None:
        L_aux, rep= auxiliary_loss_fn(fake_data, cond_batch)
        L_G_tot = L_G + L_aux

        report = {"L_D_real": L_D_real.item(), "L_D_fake": L_D_fake.item(), "L_G": L_G.item(), "L_aux": L_aux.item()}
        report.update(rep)
    else:
        L_G_tot = L_G
        report = {"L_D_real": L_D_real.item(), "L_D_fake": L_D_fake.item(), "L_G": L_G.item()}

    L_G_tot.backward()

    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=20)

    G_opt.step()

    return report


def VAE_step(D, G, D_opt, G_opt, data_batch, cond_batch, noise_batch, batch_size, device, auxiliary_loss_fn):
    
    D.zero_grad()
    G.zero_grad()
    
    alpha = 0.2

    data_and_condition = torch.cat([data_batch, cond_batch], dim=1)

    encoded = D(data_and_condition)
    latent_dim = encoded.shape[1] // 2
    mu = encoded[:, :latent_dim]
    logvar = encoded[:, latent_dim:]
    
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std  # z = mu + sigma * epsilon
    
    # Forward pass through decoder (G)
    noise_and_condition = torch.cat([z, cond_batch], dim=1)

    reconstructed = G(noise_and_condition)
    
    # Compute losses
    L_R = nn.MSELoss()(reconstructed, data_batch)
    L_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data_batch.size(0)

    if auxiliary_loss_fn is not None:
        L_aux, rep = auxiliary_loss_fn(reconstructed, cond_batch)

        L_tot = alpha * L_KL + L_R + L_aux

        report = {"L_KL": L_KL.item(), "L_R": L_R.item(), "L_tot": L_tot.item(), "L_aux": L_aux.item()}
        report.update(rep)
    else:
        L_tot = alpha * L_KL + L_R

        report = {"L_KL": L_KL.item(), "L_R": L_R.item(), "L_tot": L_tot.item()}

    L_tot.backward()

    # total_norm_D = torch.sqrt(sum(p.grad.norm()**2 for p in D.parameters() if p.grad is not None))
    # total_norm_G = torch.sqrt(sum(p.grad.norm()**2 for p in G.parameters() if p.grad is not None))
    # print(f"Gradient norm for D: {total_norm_D.item():.4f}")
    # print(f"Gradient norm for G: {total_norm_G.item():.4f}")

    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=20)
    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=20)

    D_opt.step()
    G_opt.step()
    
    return report


def DDPM_step_wrapper(scheduler: DDPMScheduler):
    def DDPM_step(D, G, D_opt, G_opt, data_batch, cond_batch, noise_batch, batch_size, device, auxiliary_loss_fn):
        # sample random t
        t = torch.randint(0, scheduler.config.num_train_timesteps, (data_batch.size(0),), device=device).long()

        # compute x_t using q_sample
        noise = torch.randn_like(data_batch, device=device)
        x_t = scheduler.add_noise(data_batch, noise, t)

        # embed timestep and concat with cond
        t_embedded = t.unsqueeze(-1).float() / scheduler.config.num_train_timesteps
        x_input = torch.cat([x_t, t_embedded], dim=-1)

        # predict noise
        noise_pred = D(x_input)

        # MSE loss with optional weighting (scheduler does not expose beta_t directly)
        mse = MSELoss(reduction="none")(noise_pred, noise)
        base_loss = mse.mean()

        # reconstruct x0 and compute auxiliary loss
        alpha_cumprod = scheduler.alphas_cumprod.to(device)
        sqrt_alpha_cumprod_t = alpha_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod_t = (1 - alpha_cumprod[t]).sqrt().unsqueeze(-1)
        x0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t

        if auxiliary_loss_fn is not None:
            thresh = int(0.9 * scheduler.config.num_train_timesteps)
            valid = (t < thresh)

            total_loss = base_loss
            report = {
                "loss": base_loss.item(),
            }
        else:
            total_loss = base_loss
            report = {"loss": base_loss.item()}

        D.zero_grad()
        total_loss.backward()
        D_opt.step()

        return report

    return DDPM_step


def DDPM_step_cond_wrapper(scheduler: DDPMScheduler):
    def DDPM_step_cond(D, G, D_opt, G_opt, data_batch, cond_batch, noise_batch, batch_size, device, auxiliary_loss_fn):
        # sample random t
        t = torch.randint(0, scheduler.config.num_train_timesteps, (data_batch.size(0),), device=device).long()

        # compute x_t using q_sample
        noise = torch.randn_like(data_batch, device=device)
        x_t = scheduler.add_noise(data_batch, noise, t)

        # embed timestep and concat with cond
        t_embedded = t.unsqueeze(-1).float() / scheduler.config.num_train_timesteps
        x_input = torch.cat([x_t, cond_batch, t_embedded], dim=-1)

        # predict noise
        noise_pred = D(x_input)

        # MSE loss with optional weighting (scheduler does not expose beta_t directly)
        mse = MSELoss(reduction="none")(noise_pred, noise)
        base_loss = mse.mean()

        # reconstruct x0 and compute auxiliary loss
        alpha_cumprod = scheduler.alphas_cumprod.to(device)
        sqrt_alpha_cumprod_t = alpha_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod_t = (1 - alpha_cumprod[t]).sqrt().unsqueeze(-1)
        x0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t

        if auxiliary_loss_fn is not None:
            thresh = int(0.5 * scheduler.config.num_train_timesteps)
            valid = (t < thresh)

            if valid.any():
                x0_sub, cond_sub = x0_pred[valid], cond_batch[valid]
                L_aux, rep = auxiliary_loss_fn(x0_sub, cond_sub)
            else:
                L_aux = torch.tensor(0.0, device=device)
                rep = {}

            total_loss = base_loss + L_aux
            report = {
                "loss": base_loss.item(),
                "L_aux": L_aux.item(),
                **rep
            }
        else:
            total_loss = base_loss
            report = {"loss": base_loss.item()}

        D.zero_grad()
        total_loss.backward()
        D_opt.step()

        return report

    return DDPM_step_cond


class ReusableDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))
        self.previous_indices = []

    def _shuffle_indices(self):
        self.indices = torch.randperm(len(self.dataset)).tolist()

    def get_batch(self):
        queued = self.previous_indices
        while len(queued) < self.batch_size:
            if self.shuffle:
                self._shuffle_indices()
            queued.extend(self.indices)  # Add individual elements to queued list
        
        self.previous_indices = queued[self.batch_size:]  # Store remaining indices for the next batch
        batch_indices = queued[:self.batch_size]  # Get the batch of the correct size
        return torch.stack([self.dataset[i][0] for i in batch_indices])

def train(D, G, D_opt, G_opt, loader, num_steps, batch_size, noise_dim, train_step_fn, device, auxiliary_loss_fn, cond_idx=None):
    # Loss function
    D.train()
    G.train()
    steps_range = trange(num_steps, position=0, leave=True)

    for step in steps_range:
        data_batch = loader.get_batch().to(device)
        noise_batch = torch.randn(batch_size, noise_dim).to(device)
        if cond_idx is None:
            cond_batch = sample_continuous(batch_size, split="train", randomize=True).to(device)
        else:
            cond_batch = sample_continuous(10, split="test", randomize=False).to(device)[cond_idx]
            cond_batch = cond_batch.repeat(batch_size, 1).to(device)

        effective_auxiliary_loss_fn = auxiliary_loss_fn
        report = train_step_fn(D, G, D_opt, G_opt, data_batch, cond_batch, noise_batch, batch_size, device, effective_auxiliary_loss_fn)
        postfix = {key: "{:.4f}".format(value) for key, value in report.items()}
        steps_range.set_postfix(postfix)
    return D, G


def get_DDPM_generate_cond(scheduler: DDPMScheduler, data_dim, batch_size=64):
    def DDPM_generate_cond(D, G, cond_batch, latent_dim, device, batch_size=batch_size):
        with torch.no_grad():
            results = []
            numgen = cond_batch.shape[0]
            for start_idx in range(0, numgen, batch_size):
                end_idx = min(start_idx + batch_size, numgen)
                current_batch_size = end_idx - start_idx

                x = torch.randn(current_batch_size, data_dim).to(device)

                for t in reversed(range(scheduler.config.num_train_timesteps)):
                    t_tensor = torch.full((current_batch_size,), t, dtype=torch.long, device=device)
                    t_embedded = t_tensor.unsqueeze(-1).float() / scheduler.config.num_train_timesteps
                    x_input = torch.cat([x, cond_batch, t_embedded], dim=-1)

                    with torch.no_grad():
                        noise_pred = D(x_input)

                    x = scheduler.step(model_output=noise_pred, timestep=t, sample=x).prev_sample
                results.append(x)

            return torch.cat(results, dim=0)
    return DDPM_generate_cond

def get_DDPM_generate_guided(scheduler: DDPMScheduler, data_dim, auxiliary_loss_fn, batch_size=64):
    def DDPM_generate_guided(D, G, cond_batch, latent_dim, device, auxiliary_loss_fn=auxiliary_loss_fn, batch_size=batch_size):
        results = []
        numgen = cond_batch.shape[0]

        num_guided_timesteps = int(0.5 * scheduler.config.num_train_timesteps)

        for start_idx in range(0, numgen, batch_size):
            end_idx = min(start_idx + batch_size, numgen)
            current_batch_size = end_idx - start_idx

            # Start with pure noise
            x = torch.randn(current_batch_size, data_dim, device=device)

            for t in tqdm(reversed(range(scheduler.config.num_train_timesteps))):
                t_tensor = torch.full((current_batch_size,), t, dtype=torch.long, device=device)
                t_embedded = t_tensor.unsqueeze(-1).float() / scheduler.config.num_train_timesteps
                x_input = torch.cat([x, t_embedded], dim=-1)

                # Model prediction
                noise_pred = D(x_input)

                # Apply auxiliary loss only if t < threshold (later timesteps)
                threshold = int(0.5 * scheduler.config.num_train_timesteps)
                if t < threshold:
                    alpha_cumprod = scheduler.alphas_cumprod.to(device)
                    sqrt_alpha_cumprod_t = alpha_cumprod[t].sqrt().unsqueeze(-1)
                    sqrt_one_minus_alpha_cumprod_t = (1 - alpha_cumprod[t]).sqrt().unsqueeze(-1)
                    x0_pred = (x - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t

                    aux_loss, _ = auxiliary_loss_fn(x0_pred, cond_batch)
                    aux_loss.backward(retain_graph=True)
                    grad = x.grad / num_guided_timesteps

                    # Update x based on this local gradient only
                    x = (x - grad).detach().requires_grad_(True)
                    x.retain_grad()

                # Always apply scheduler step
                x = scheduler.step(model_output=noise_pred, timestep=t, sample=x).prev_sample
                x.retain_grad()

            results.append(x)

        return torch.cat(results, dim=0)

    return DDPM_generate_guided






def VAE_generate(D, G, cond_batch, latent_dim, device):
    with torch.no_grad():
        numgen = cond_batch.shape[0]
        z = torch.randn(numgen, latent_dim).to(device)
        z_and_condition = torch.cat([z, cond_batch], dim=1)
        generated_data = G(z_and_condition)
        return generated_data

# def VAE_generate_cond(D, G, cond_batch, latent_dim, device):
#     numgen = cond_batch.shape[0]
#     z = torch.randn(numgen, latent_dim).to(device)
#     labels = torch.ones(numgen, 1).to(device)
#     z = torch.cat([z, labels], dim=1)
#     generated_data = G(z)
#     return generated_data
    
def GAN_generate(D, G, cond_batch, noise_dim, device):
    with torch.no_grad():
        numgen = cond_batch.shape[0]
        noise = torch.randn(numgen, noise_dim).to(device)
        noise_and_condition = torch.cat([noise, cond_batch], dim=1)
        labels = torch.ones(numgen, 1).to(device)
        generated_data = G(noise_and_condition)
        return generated_data

def train_model(data, model_type, train_params, auxiliary_loss_fn, cond_idx, device):
    batch_size, disc_lr, gen_lr, noise_dim, num_epochs, n_hidden, layer_size= train_params

    

    data = torch.tensor(data).float()
    sample_condition = sample_continuous(data.shape[0], randomize=True).to(device)

    loader = ReusableDataLoader(TensorDataset(data), batch_size)

    data_dim = data.shape[1]
    cond_dim = sample_condition.shape[1]

    if model_type in ["GAN"]:
        train_step = GAN_step
        generate_fn = GAN_generate
        D_in = data_dim + cond_dim
        D_out = 1
        G_in = noise_dim +cond_dim
        G_out = data_dim
    elif model_type in ["VAE"]:
        train_step = VAE_step
        generate_fn = VAE_generate
        D_in = data_dim + cond_dim
        D_out = 2*noise_dim
        G_in = noise_dim + cond_dim
        G_out = data_dim
    elif model_type in ["DDPM_guided"]:
        scheduler = DDPMScheduler(num_train_timesteps=100)
        train_step = DDPM_step_wrapper(scheduler)
        generate_fn = get_DDPM_generate_guided(scheduler, data_dim, auxiliary_loss_fn, batch_size=batch_size)
        D_in = data_dim + 1
        D_out = data_dim
        G_in = 1 #unused
        G_out = 1 #unused
    elif model_type in ["DDPM_conditional"]:
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        train_step = DDPM_step_cond_wrapper(scheduler)
        generate_fn = get_DDPM_generate_cond(scheduler, data_dim, batch_size=batch_size)
        D_in = data_dim + cond_dim + 1
        D_out = data_dim
        G_in = 1 #unused
        G_out = 1 #unused
    # else:
    #     raise ValueError("Invalid model_type")


    D = Down_Model(D_in, D_out, layer_size, n_hidden)
    G = Up_Model(G_in, G_out, layer_size, n_hidden)

    
    D.to(device)
    G.to(device)
    D_opt = torch.optim.Adam(D.parameters(), lr=disc_lr, betas=(0.5,0.999))
    G_opt = torch.optim.Adam(G.parameters(), lr=gen_lr, betas=(0.5,0.999))

    

    if num_epochs>0:
        num_steps = num_epochs*len(data)//batch_size
    else:
        num_steps = -num_epochs #hacky way to specify fixed number of steps rather than epochs
    

    train(D, G, D_opt, G_opt, loader, num_steps, batch_size, noise_dim, train_step, device, auxiliary_loss_fn, cond_idx)

    return D, G, generate_fn
