from Dataset_qgis import CycleGANDataset_train_qgis, CycleGANDataset_val_qgis
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import PolynomialLR, LinearLR
from models import cycle_G, cycle_D, cycle_D_SN, cycle_G_SN
from losses import lsgan, adversarial, cycleloss, reconloss, ploss
import os
from tqdm import tqdm
from torch.autograd import Variable
from utils import plot_figures, plot_figure2, add_noise, compute_msv, split_into_patches, merge_into_tile, plot_mask
import logging
import glob
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from KAIR.models.network_dncnn import DnCNN

## hyperparameters
params = dict()
params["lr"] = 0.0002
params["beta1"] = 0.5
params["dropout"] = 0.0
params["weight_decay_G"] = 0.0 # l2-regularization
params["add_noise_G"] = False # add noise to the generated image to tackle the steganography
params["sigma"] = 0.01
params["sn_G"] = False # add spectral normalization to generators
params["sn_D"] = False # add spectral normalization to discrimators
params["K"] = 1 # the number of pauses to update the discriminators
params["niter"] = 480
params["save_epoch_freq"] = 20
params["save_last"] = True
params["eval_epoch_freq"] = 10
params["prev_iter"] = 0
params["save_dir"] = "model_weights/cyclegan/epure2_align_blur_mu3_mask_full_l3_bias_BN_GN2_denoise"
params["use_lsgan"] = True
params["use_upsampling"] = True
params["align_data"] = True ## False if original data is misaligned
params["hidden_dim"] = 64
params["nlayers"] = 3 # in discriminatives network; default 3 
params["slope_D"] = 0.2
params["batch_size"] = 32
params["patch_size"] = 128
params["gamma"] = 10
params["use_align"] = True
params["use_ploss"] = False # use perceptual loss
params["use_closs"] = True
params["coef_ploss"] = 1.0
params["mu"] = 3.0
params["plot_figure"] = True
params["figure_savedir"] = "fig_results/cyclegan/epure2_align_blur_mu3_mask_full_l3_bias_BN_GN2_denoise"
params["text_mask"] = True # for discriminative loss
params["gloss_use_mask"] = True # for cycle loss; can be set to false if mask the input
params["align_use_mask"] = True # for recon loss; can be set to false if mask the input
params["log_dir"] = "logs/cyclegan/epure2_align_blur_mu3_mask_full_l3_bias_BN_GN2_denoise.txt"
params["do_tensorboard"] = False
params["tb_dir"] = "tb/epure2_align_blur_mu3_mask_full_l3_bias_BN_GN2_denoise"

params["dncnn_model_path"] = "KAIR/model_zoo/dncnn_color_blind.pth"
### Gaussian blurry
params["Gaussian_sigma"] = 4 # 4 # None if blurry is not applicable
params["Gaussian_k"] = 7
### examine the steganography
params["vis_stegano"] = True

params["device_id"] = 1

## define dataloader
style1 = "Cassini"
style2 = "IGN_epure2"

train_path1 = f"patchdata2/{style1}/train"
train_path2 = f"patchdata2/{style2}/train"
train_mask_path1 = f"patchdata2/{style1}_mask/train"

val_path1 = f"patchdata2/{style1}/val"
val_path2 = f"patchdata2/{style2}/val"
val_mask_path1 = f"patchdata2/{style1}_mask/val"

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Pad(padding=16), # still, unstable training for l4
    # transforms.Resize(size=(params["patch_size"], params["patch_size"])),
])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Pad(padding=16),
    # transforms.Resize(size=(params["patch_size"], params["patch_size"])),
])

train_set = CycleGANDataset_train_qgis(train_path1, train_path2, train_mask_path1, patch_size=params["patch_size"], transform=transform)
val_set = CycleGANDataset_val_qgis(val_path1, val_path2, val_mask_path1, transform=transform_eval)

train_dataloader = DataLoader(dataset=train_set, batch_size=params["batch_size"], shuffle=True, drop_last=True)
val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, drop_last=False)

writer = SummaryWriter(params["tb_dir"])
def train(params, train_dataloader, val_dataloader, eval_only=False):
    device = f"cuda:{params['device_id']}" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(params["save_dir"]):
        os.mkdir(params["save_dir"])

    ## define model & optimizer & loss func
    if params["sn_G"]:
        print("Using spectral normalization in Generator")
        model_G12 = cycle_G_SN(in_channel=3, out_channel=3, hidden_dim=params["hidden_dim"], use_mask=params["text_mask"], upsampling=params["use_upsampling"]).to(device)
        model_G21 = cycle_G_SN(in_channel=3, out_channel=3, hidden_dim=params["hidden_dim"], use_mask=params["text_mask"], upsampling=params["use_upsampling"]).to(device)
    else:
        model_G12 = cycle_G(in_channel=3, out_channel=3, hidden_dim=params["hidden_dim"], use_mask=params["text_mask"], norm_type = "batch", upsampling=params["use_upsampling"]).to(device)
        model_G21 = cycle_G(in_channel=3, out_channel=3, hidden_dim=params["hidden_dim"], use_mask=params["text_mask"], norm_type = "batch", upsampling=params["use_upsampling"]).to(device)
    if params["sn_D"]:
        print("Using spectral normalization in Discriminator")
        model_D1 = cycle_D_SN(in_channel=3, hidden_dim=params["hidden_dim"], n_layers=params["nlayers"], use_sigmoid=not params["use_lsgan"], 
                                use_mask = params["text_mask"], dropout = params["dropout"]).to(device)
        model_D2 = cycle_D_SN(in_channel=3, hidden_dim=params["hidden_dim"], n_layers=params["nlayers"], use_sigmoid=not params["use_lsgan"], 
                                use_mask = params["text_mask"], dropout = params["dropout"]).to(device)  

    else:
        model_D1 = cycle_D(in_channel=3, hidden_dim=params["hidden_dim"], n_layers=params["nlayers"], use_sigmoid=not params["use_lsgan"], 
                        use_mask = params["text_mask"], norm_type = "group", slope = params["slope_D"], dropout = params["dropout"]).to(device)
        model_D2 = cycle_D(in_channel=3, hidden_dim=params["hidden_dim"], n_layers=params["nlayers"], use_sigmoid=not params["use_lsgan"], 
                        use_mask = False, norm_type = "group", slope = params["slope_D"], dropout = params["dropout"]).to(device)

    # DNCNN
    denoiser = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
    denoiser.load_state_dict(torch.load(params["dncnn_model_path"]), strict=True)
    denoiser.eval()
    for k, v in denoiser.named_parameters():
        v.requires_grad = False
    denoiser = denoiser.to(device)

    Closs = cycleloss(params["gamma"])
    if params["use_lsgan"]:
        Adloss = lsgan()
    else:
        Adloss = adversarial()
    Reconloss = reconloss(params["mu"], sigma = params["Gaussian_sigma"], kernel = params["Gaussian_k"])
    Ploss = ploss().to(device)

    optimizer_G = Adam([{"params": model_G12.parameters(), "lr": params["lr"], "betas":(params["beta1"], 0.999), "weight_decay":0.0},
                      {"params": model_G21.parameters(), "lr": params["lr"], "betas":(params["beta1"], 0.999), "weight_decay":params["weight_decay_G"]}])
    optimizer_D = Adam([{"params": model_D1.parameters(), "lr": params["lr"], "betas":(params["beta1"], 0.999)},
                      {"params": model_D2.parameters(), "lr": params["lr"], "betas":(params["beta1"], 0.999)},])
    schedule_G = PolynomialLR(optimizer_G, total_iters=params["niter"])
    schedule_D = PolynomialLR(optimizer_D, total_iters=params["niter"])
    # schedule_G = LinearLR(optimizer_G, start_factor=1.0, end_factor=1e-2, total_iters=params["niter"])
    # schedule_D = LinearLR(optimizer_D, start_factor=1.0, end_factor=1e-2, total_iters=params["niter"])

    # if params["continue_train"]:
    #     print("Use pretrain weights")
    #     model = torch.load(params["pretrain_dir"])
    #     model_G12.load_state_dict(model["G12_state_dict"])
    #     model_G21.load_state_dict(model["G21_state_dict"])
    #     model_D1.load_state_dict(model["D1_state_dict"])
    #     model_D2.load_state_dict(model["D2_state_dict"])
    #     optimizer_G.load_state_dict(model["optimizer_G"])
    #     optimizer_D.load_state_dict(model["optimizer_D"])
    #     schedule_G.load_state_dict(model["schedule_G"])
    #     schedule_D.load_state_dict(model["schedule_D"])

    if eval_only:
        print("Start testing")
        evaluate(params, test_dataloader, model_G12, model_G21, model_D1, model_D2, params["plot_figure"], 0)
        return
    ## logger
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(params["log_dir"])
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    ## train
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    logger.info("start training...")
    n_update = 0
    for n in tqdm(range(params["prev_iter"], params["prev_iter"]+params["niter"])):
        # # Getting all memory using os.popen()
        # total_memory, used_memory, free_memory = map(
        #     int, os.popen('free -t -m').readlines()[-1].split()[1:])
        # # Memory usage
        # print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))

        model_G12.train()
        model_G21.train()
        model_D1.train()
        model_D2.train()
        for images1, images2, masks1 in tqdm(train_dataloader):
            images1, images2 = images1.float().to(device), images2.float().to(device)
            masks1 = masks1.squeeze().float().to(device)

            if params["sn_D"]:
                model_D1.normalize_weight()
                model_D2.normalize_weight()

            gen12 = model_G12(images1)
            gen21 = model_G21(images2)

            netD_output_size = model_D1(gen21.detach(), masks1)[0].size()
            valid = Variable(Tensor(netD_output_size).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(Tensor(netD_output_size).fill_(0.0), requires_grad=False).to(device)

            # update discriminator
            if n_update % params["K"] == 0:
                optimizer_D.zero_grad()
                dloss = Adloss(model_D1(gen21.detach(), masks1)[0], fake)/4 + Adloss(model_D1(images1, masks1)[0], valid)/4 \
                        + Adloss(model_D2(gen12.detach())[0], fake)/4 + Adloss(model_D2(images2)[0], valid)/4 

                dloss.backward()
                optimizer_D.step()

            # update generator
            gen12 = model_G12(images1)
            gen21 = model_G21(images2)
            if params["add_noise_G"]:
                gen1 = model_G21(add_noise(gen12, params["sigma"]), 0)
                gen2 = model_G12(add_noise(gen21, params["sigma"]), 0)
            else:
                # gen1 = model_G21(gen12)
                gen1 = model_G21(denoiser(gen12))
                gen2 = model_G12(gen21)


            optimizer_G.zero_grad()
            gloss = Adloss(model_D1(model_G21(images2), masks1)[0], valid)/2 + Adloss(model_D2(model_G12(images1))[0], valid)/2 
                    
            if params["use_ploss"]:
                gloss += params["coef_ploss"] * (Ploss(gen12, images2) + Ploss(gen21, images1))

            if params["use_closs"]:
                if params["gloss_use_mask"]:
                    gloss += Closs(images1, gen1, masks1)/2 + Closs(images2, gen2)/2
                    # gloss += Closs(images2, gen2)/2 + 0.2 * Closs(images1, gen1, masks1)/2
                    # gloss += Adloss(model_D1(gen1, masks1)[0], valid)
                else:
                    gloss += Closs(images1, gen1)/2 + Closs(images2, gen2)/2
            
            if params["use_align"]:                    
                if params["align_use_mask"]:
                    gloss += Reconloss(gen12, images2, 0)/2 + Reconloss(gen21, images1, masks1)/2
                else:
                    gloss += Reconloss(gen12, images2)/2 + Reconloss(gen21, images1)/2
            
            gloss.backward()
            optimizer_G.step()
            n_update = (n_update + 1) % params["K"]

        if (n+1) % params["save_epoch_freq"] == 0:
            state_dict = {
                "G12_state_dict": model_G12.state_dict(),
                "G21_state_dict": model_G21.state_dict(),
                "D1_state_dict": model_D1.state_dict(),
                "D2_state_dict": model_D2.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "schedule_G": schedule_G.state_dict(),
                "schedule_D": schedule_D.state_dict(),
            }
            torch.save(state_dict, os.path.join(params["save_dir"], f"{n+1}.ckpt"))
        if (n+1) % params["eval_epoch_freq"] == 0:
            # total_g_loss_train, total_d_loss_train, total_align_loss_train, total_p_loss_train = evaluate(params, train_dataloader, model_G12, model_G21, model_D1, model_D2, False, n)
            # print(
            #     "train" + " [Epoch %d/%d] [D loss: %f] [G loss: %f] [Align loss:%f] [P loss: %f]"
            #     % (n+1, params['niter'], total_d_loss_train, total_g_loss_train, total_align_loss_train, total_p_loss_train)
            # )
            # logger.info("train" + " [Epoch %d/%d] [D loss: %f] [G loss: %f] [Align loss:%f] [P loss: %f]"
            #     % (n+1, params['niter'], total_d_loss_train, total_g_loss_train, total_align_loss_train, total_p_loss_train))
            
            total_g_loss, total_d_loss, total_align_loss, total_p_loss, total_closs = evaluate(params, val_dataloader, model_G12, model_G21, model_D1, model_D2, denoiser, params["plot_figure"], n)
            print(
                "Val" + " [Epoch %d/%d] [D loss: %f] [G loss: %f] [C loss: %f] [Align loss:%f] [P loss: %f]"
                % (n+1, params['niter'], total_d_loss, total_g_loss, total_closs, total_align_loss, total_p_loss)
            )
            logger.info("Val" + " [Epoch %d/%d] [D loss: %f] [G loss: %f] [C loss: %f] [Align loss:%f] [P loss: %f]"
                % (n+1, params['niter'], total_d_loss, total_g_loss, total_closs, total_align_loss, total_p_loss))
        
        schedule_G.step()
        schedule_D.step()

        if (n+1) % params["eval_epoch_freq"] == 0 and params["do_tensorboard"]:
            num = 0
            for m in model_D2.modules():
                if isinstance(m, nn.BatchNorm2d):
                    num += 1
                    writer.add_histogram(f"weight distribution BN{num} in D2", m.weight.cpu(), n)
                    writer.add_histogram(f"bias distribution BN{num} in D2", m.bias.cpu(), n)
                    mean_estm = m.running_mean.cpu().numpy()
                    var_estm = m.running_var.cpu().numpy()
                    writer.add_scalars(f"mean BN{num}", {f"{j}": mean_estm[j] for j in range(len(mean_estm))}, n)
                    writer.add_scalars(f"var BN{num}", {f"{j}": var_estm[j] for j in range(len(var_estm))}, n)
    logger.info("End of training")
    writer.close()

def evaluate(params, dataloader, model_G12, model_G21, model_D1, model_D2, denoiser, is_plot, epoch = 0):
    model_G12.eval()
    model_G21.eval()
    model_D1.eval()
    model_D2.eval()
    device = f"cuda:{params['device_id']}" if torch.cuda.is_available() else "cpu"

    Closs = cycleloss(params["gamma"])
    if params["use_lsgan"]:
        Adloss = lsgan()
    else:
        Adloss = adversarial()
    Reconloss = reconloss(params["mu"], sigma = params["Gaussian_sigma"], kernel = params["Gaussian_k"])
    Ploss = ploss().to(device)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    total_g_loss = 0
    total_d_loss = 0
    total_p_loss = 0
    total_align_loss = 0
    total_closs = 0

    count = 0
    with torch.no_grad():
        for images1, images2, masks1 in dataloader:
            images1, images2 = images1.float().to(device), images2.float().to(device)
            masks1 = masks1.float().squeeze().to(device)

            gen12 = model_G12(images1)
            gen21 = model_G21(images2)
            # gen1 = model_G21(gen12)
            gen1 = model_G21(denoiser(gen12))
            gen2 = model_G12(gen21)

            netD_output_size = model_D1(gen21.detach(), masks1)[0].size()
            valid = Variable(Tensor(netD_output_size).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(Tensor(netD_output_size).fill_(0.0), requires_grad=False).to(device)

            dloss = Adloss(model_D1(gen21.detach(), masks1)[0], fake)/4 + Adloss(model_D1(images1, masks1)[0], valid)/4 \
                    + Adloss(model_D2(gen12.detach())[0], fake)/4 + Adloss(model_D2(images2)[0], valid)/4
            # dloss = Adloss(model_D1(gen21.detach(), masks2), fake)/4 + Adloss(model_D1(images1, masks1), valid)/4 \
            #         + Adloss(model_D2(gen12.detach(), masks1), fake)/4 + Adloss(model_D2(images2, masks2), valid)/4

            gloss = Adloss(model_D1(model_G21(images2), masks1)[0], valid)/2 + Adloss(model_D2(model_G12(images1))[0], valid)/2
            # gloss = Adloss(model_D1(model_G21(images2, masks2), masks2), valid)/2 + Adloss(model_D2(model_G12(images1, masks1), masks1), valid)/2
            align_loss = torch.tensor(0, device=device, dtype=torch.float32)
            closs = torch.tensor(0, device=device, dtype=torch.float32)

            if params["use_closs"]:
                if params["gloss_use_mask"]:
                    closs += Closs(images1, gen1, masks1)/2 + Closs(images2, gen2)/2
                else:
                    closs += Closs(images1, gen1)/2 + Closs(images2, gen2)/2
            
            if params["use_align"]:
                if params["align_use_mask"]:
                    align_loss += Reconloss(gen12, images2, 0)/2 + Reconloss(gen21, images1, masks1)/2
                else:
                    align_loss += Reconloss(gen12, images2)/2 + Reconloss(gen21, images1)/2

            perceptual_loss = (Ploss(gen12, images2) + Ploss(gen21, images1)) / 2

            total_d_loss += dloss
            total_g_loss += gloss
            total_p_loss += perceptual_loss
            total_align_loss += align_loss
            total_closs += closs

            if is_plot and count < 100:
                if len(masks1.shape) == 2:
                    masks1 = masks1[None]
                mask_gen1 = model_D1(gen21.detach(), masks1)[1]
                mask_gen2 = model_D2(gen12.detach())[1]
                for i in range(len(images1)):
                    if not os.path.exists(params["figure_savedir"]):
                        os.mkdir(params["figure_savedir"])
                    image_list = [images1[i], images2[i], gen12[i], gen21[i], gen1[i], gen2[i], denoiser(gen12[i])]
                    mask_list = [masks1[i]]
                    # if params["vis_stegano"]:
                    #     gen12_patches128 = split_into_patches(images2[i], patch_size=(128, 128))
                    #     # gen12_patches64 = split_into_patches(gen12[i], patch_size=(64, 64))
                    #     gen1_patches128 = model_G21(gen12_patches128, 0)
                    #     # gen1_patches64 = model_G21(gen12_patches64, 0)

                    #     gen21_patches128 = split_into_patches(images1[i], patch_size=(128, 128))
                    #     # gen21_patches64 = split_into_patches(gen21[i], patch_size=(64, 64))
                    #     gen2_patches128 = model_G12(gen21_patches128, 0)
                    #     # gen2_patches64 = model_G12(gen21_patches64, 0)

                    #     image_list.append(merge_into_tile(gen1_patches128, size=(2,2)))
                    #     image_list.append(merge_into_tile(gen2_patches128, size=(2,2)))
                    #     # image_list.append(merge_into_tile(gen1_patches64, size=(4,4)))
                    #     # image_list.append(merge_into_tile(gen2_patches64, size=(4,4)))

                    if epoch >= 400:
                        image_list.append(mask_gen1[i])
                        for j in range(len(mask_list)):
                            plot_mask(mask_list[j], os.path.join(params["figure_savedir"], f"{epoch}_{count}_{j+len(image_list)}.jpg"))
                    
                    for j in range(len(image_list)):
                        plot_figure2(image_list[j], os.path.join(params["figure_savedir"], f"{epoch}_{count}_{j}.jpg"))
                    count += 1
                    # plot_figures([images1[0], images2[0], gen12[0], gen21[0]], os.path.join(params["figure_savedir"], f"{epoch}.jpg"))
                    # is_plot = False
        total_g_loss /= len(dataloader)
        total_d_loss /= len(dataloader)
        total_p_loss /= len(dataloader)
        total_align_loss /= len(dataloader)
        total_closs /= len(dataloader)
    return total_g_loss.item(), total_d_loss.item(), total_align_loss.item(), total_p_loss.item(), total_closs.item()
if __name__ == "__main__":
    train(params, train_dataloader, val_dataloader, eval_only=False)
