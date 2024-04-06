import argparse
from models_seg import unet, cycle_D
import torch
from torch.utils.data import DataLoader
from Dataset_qgis import SegDataset_Test, SegDataset_Val
import os
from torchvision import transforms
from losses_seg import CELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.tensorboard import SummaryWriter
import logging
import matplotlib.pyplot as plt
from PIL import Image
from utils import plot_figure2, plot_binary, save_mid_result
from tqdm import tqdm
from metric import dilatedIoU
import numpy as np


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ign", type=str, default="patchdata2/epure_multilayers_rgb_seg", help="directory to ign dataset")
    parser.add_argument("--gt", type=str, default="patchdata2/epure_multilayers_seg_wbg", help="directory to Ground truth dataset")
    parser.add_argument("--model_dir", type=str, default="model_weights/seg/seg_rgb/final.ckpt")
    parser.add_argument("--log_dir", type=str, default="logs/seg/seg_rgb_test.txt")
    # parser.add_argument("--result_savedir", type=str, default="data_result/seg/30IGN_v2")
    # parser.add_argument("--fig_savedir", type=str, default="fig_results_test/seg/30IGN_v2")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--norm_type_G", type=str, default="batch")
    parser.add_argument("--norm_type_D", type=str, default="group")
    parser.add_argument("--nlayers", type=int, default=3)
    parser.add_argument("--no_use_lsgan", action="store_false")
    parser.add_argument("--no_text_mask", action="store_false")
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--is_plot", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--mode", type=str, default="test")
    args = parser.parse_args()
    return args
 

def evaluate(args, test_dataloader, model, device):
    model.eval()
    criterion = CELoss().to(device)

    # logger
    logger = logging.getLogger("Seg Model")
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Using the classifier trained with gt derived from transferred data")
    logger.info("start testing")

    # if not os.path.exists(args.fig_savedir):
    #     os.mkdir(args.fig_savedir)
    # if not os.path.exists(args.result_savedir):
    #     os.mkdir(args.result_savedir)
    # if args.split:
    #     fig_outpath = os.path.join(args.fig_savedir, args.mode)
    #     result_outpath = os.path.join(args.result_savedir, args.mode)
    #     if not os.path.exists(fig_outpath):
    #         os.mkdir(fig_outpath)
    #     if not os.path.exists(result_outpath):
    #         os.mkdir(result_outpath)
    # else:
    #     fig_outpath = args.fig_savedir
    #     result_outpath = args.result_savedir

    miou = [0]*5
    total_CEloss = 0
    with torch.no_grad():
        for image, gt in tqdm(test_dataloader):
            image, gt = image.float().to(device), gt.long().to(device)
            pred = model(image)
            
            total_CEloss += criterion(pred, gt).detach()
            pred_label = np.argmax(pred.permute(0, 2, 3, 1).cpu().numpy(), axis=-1)
            for l in range(5):
                miou[l] += dilatedIoU(pred_label[0], gt.cpu().numpy()[0], l, window=3)

        total_CEloss /= len(test_dataloader)
        miou = [miou[i]/len(test_dataloader) for i in range(5)]   
        logger.info("[Total CEloss:%f], [background miou: %f], [forest miou: %f], [hydro miou: %f], [road miou: %f], [town miou: %f]"
                    % (total_CEloss, miou[0], miou[1], miou[2], miou[3], miou[4]))
            # dirname = os.path.join(result_outpath, city_name[0])
            # if not os.path.exists(dirname):
            #     os.mkdir(dirname)
            # save_mid_result(pred[0], os.path.join(dirname, basename[0]))
            
            # if args.is_plot:
            #     p = os.path.join(fig_outpath, city_name[0])
            #     name = basename[0].split(".")[0]
            #     if not os.path.exists(p):
            #         os.mkdir(p)
            #     plot_figure2(image[0], os.path.join(p, f"{name}_0.jpg"))
            #     image_list = [pred[0].cpu().detach().numpy().transpose(1,2,0)]
            #     for j in range(len(image_list)):
            #         plot_binary(image_list[j], os.path.join(p, f"{name}_{j+1}.jpg"))


def main(args):
    # define the dataloader
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_set = SegDataset_Val(
            os.path.join(args.ign, args.mode), os.path.join(args.gt, args.mode), transform=transform)
    # else:
    #     test_set = SegDataset_Test(
    #         os.path.join(args.ign, args.mode), transform = transform)
    test_dataloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, drop_last=False)

    # define the model
    model = unet(in_channel=3, out_channel=5, hidden_dim=args.hidden_dim, norm_type=args.norm_type_G).to(device) # From Cassini to IGN
    
    # load pretrain model
    state = torch.load(args.model_dir)
    model.load_state_dict(state["G12_state_dict"])

    evaluate(args, test_dataloader, model, device)
        

if __name__ == "__main__":
    args = argparser()
    main(args)