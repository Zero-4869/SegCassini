import argparse
from models_seg import unet, cycle_D
import torch
from torch.utils.data import DataLoader
from Dataset_qgis import SegDataset_Train, SegDataset_Val
import os
from torchvision import transforms
from losses_seg import CELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.tensorboard import SummaryWriter
import logging
import matplotlib.pyplot as plt
from PIL import Image
from utils import plot_figure2, plot_binary
from tqdm import tqdm


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ign", type=str, default="patchdata2/epure_multilayers_rgb2_seg", help="directory to IGN dataset")
    parser.add_argument("--gt", type=str, default="patchdata2/epure_multilayers_seg_wbg", help="directory to Ground truth dataset")
    parser.add_argument("--model_savedir", type=str, default="model_weights/seg/seg_rgb2")
    parser.add_argument("--fig_savedir", type=str, default="fig_results/seg/seg_rgb2")
    parser.add_argument("--log_dir", type=str, default="logs/seg/seg_rgb2.txt")
    parser.add_argument("--tb_dir", type=str, default="tb/seg/seg_rgb2")

    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--norm_type_G", type=str, default="batch")
    parser.add_argument("--norm_type_D", type=str, default="group")
    parser.add_argument("--nlayers", type=int, default=3)
    parser.add_argument("--no_use_lsgan", action="store_false")
    parser.add_argument("--no_text_mask", action="store_false")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--N_epochs", type=int, default=100)
    parser.add_argument("--eval_epoch_freq", type=int, default=1)
    parser.add_argument("--save_epoch_freq", type=int, default=5)
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    return args

def train(args, train_dataloader, G12, G21, D1, D2, optimizer, scheduler):
    loss = BCELoss()
    G12.train(); G21.train(); D1.train(); D2.train()
    
    pass
    

def evaluate(args, val_dataloader, model, device, epoch):
    model.eval()
    total_CEloss = 0
    count = 0
    criterion = CELoss().to(device)
    with torch.no_grad():
        for image, gt in tqdm(val_dataloader):
            image, gt = image.float().to(device), gt.long().to(device)
            pred = model(image)            
            loss = criterion(pred, gt)
            total_CEloss += loss.detach()

            # if count < 100:
            #     if not os.path.exists(args.fig_savedir):
            #         os.mkdir(args.fig_savedir)
            #     plot_figure2(image[0], os.path.join(args.fig_savedir, f"{epoch%5}_{count}_0.jpg"))
            #     image_list = [gt[0].cpu().detach().numpy().transpose(1,2,0), pred[0].cpu().detach().numpy().transpose(1,2,0)]
            #     for j in range(len(image_list)):
            #         plot_binary(image_list[j], os.path.join(args.fig_savedir, f"{epoch%5}_{count}_{j+1}.jpg"))
            #     count += 1
        total_CEloss /= len(val_dataloader)
    
    return total_CEloss

def main(args):
    # define the dataloader
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = SegDataset_Train(
        os.path.join(args.ign, "train"), os.path.join(args.gt, "train"),
        patch_size = args.patch_size, transform = transform
    )
    val_set = SegDataset_Val(
        os.path.join(args.ign, "val"), os.path.join(args.gt, "val"),
        transform = transform
    )
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, drop_last=False)

    # define the model
    model = unet(in_channel=3, out_channel=5, hidden_dim=args.hidden_dim, norm_type=args.norm_type_G).to(device) # From Cassini to IGN
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    scheduler = PolynomialLR(optimizer, total_iters = args.N_epochs)
    writer = SummaryWriter(args.tb_dir)

    criterion = CELoss().to(device)

    if not os.path.exists(args.model_savedir):
        os.mkdir(args.model_savedir)

    # logger
    logger = logging.getLogger("Seg Model")
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Using the classifier trained with gt derived from transferred data")
    logger.info("start training")
    
    tolerance = 10
    prev_loss = 100
    for epoch in range(args.N_epochs):
        model.train()
        
        total_loss = 0
        for image, gt in tqdm(train_dataloader):
            image, gt = image.float().to(device), gt.long().to(device)
            pred = model(image)

            optimizer.zero_grad()
            loss = criterion(pred, gt)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # save model
        if (epoch+1) % args.save_epoch_freq == 0:
            state_dict = {
                "G12_state_dict": model.state_dict(),
                "Optimizer": optimizer.state_dict(),
                "Scheduler": scheduler.state_dict()
            }
            torch.save(state_dict, os.path.join(args.model_savedir, f"{(epoch+1)%5}.ckpt"))
        state_dict = {
                "G12_state_dict": model.state_dict(),
                "Optimizer": optimizer.state_dict(),
                "Scheduler": scheduler.state_dict()
            }
        torch.save(state_dict, os.path.join(args.model_savedir, f"final.ckpt"))
        
        # log losses
        total_loss /= len(train_dataloader)
        logger.info("Train" + " [Epoch %d/%d] [CE loss: %f]"
                    % (epoch+1, args.N_epochs, total_loss))
        
        writer.add_scalar("train_loss", total_loss, epoch)

        if (epoch+1) % args.eval_epoch_freq == 0:
            loss_eval = evaluate(args, val_dataloader, model, device, epoch+1)
            logger.info("====================================")
            logger.info("Val" + " [Epoch %d/%d] [CE loss: %f]"
                    % (epoch+1, args.N_epochs, loss_eval))
            writer.add_scalar("val_loss", loss_eval, epoch)
            # early stopping
            if prev_loss < loss_eval:
                tolerance -= 1
            if tolerance == 0:
                logger.info("Early stopping")
                break
            prev_loss = loss_eval


        



if __name__ == "__main__":
    args = argparser()
    main(args)