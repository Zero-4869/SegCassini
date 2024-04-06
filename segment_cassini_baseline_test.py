from Dataset_qgis import Dataset_aggregation
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import cycle_G
from models_seg import unet
import os
from tqdm import tqdm
from utils import plot_figures, plot_figure2, add_noise, compute_msv, split_into_patches, merge_into_tile, plot_mask
import argparse
from metric import IoU, dilatedIoU, f1, precision, recall
from utils import plot_figure2, save_mid_result
import numpy as np
from PIL import Image
import logging


forest_names = ["Clermont"]
hydro_names = ["Agen", "Amiens", "Argentan", "Arques - Aumale - Forges - Neufchatel Â– Yvetot", "Auch", "Blois", "Cahors", "Cambrai", "Castillonez", "Cherbourg", "Clermont", "Compiegne", "Coutances",  "Fontainebleau", "Laon Â– Noyon", "Lille", "Meziere Â– Sedan", "Montauban", "Paris", "Rocroi", "Saint Omer"]
def argparser():
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument("--cassini", type=str, default="data/30cassini_cropped", help="directory to Cassini dataset")
    parser.add_argument("--gt_forest", type=str, default="data/cassini_gt_sheet/land_use")
    parser.add_argument("--gt_hydro", type=str, default="data/cassini_gt_sheet/hydro")
    parser.add_argument("--gt_road", type=str, default="data/cassini_gt_sheet/roads")
    parser.add_argument("--gt_town", type=str, default="data/cassini_gt_sheet/cities_and_domains")

    # segment params
    parser.add_argument("--model_dir", type=str, default="model_weights/seg/seg_cassini/final.ckpt")
    # parser.add_argument("--model_dir", type=str, default="model_weights/seg/baseline_v2/100.ckpt")
    # parser.add_argument("--model_dir", type=str, default="model_weights/seg/dis_v2_retrain/100.ckpt")
    parser.add_argument("--hidden_dim_seg", type=int, default=64)
    parser.add_argument("--norm_type_G", type=str, default="batch")
    parser.add_argument("--device_id", type=int, default=1)
    # plot 
    parser.add_argument("--fig_savedir", type=str, default="fig_results_test/seg/evaluation_baseline")
    parser.add_argument("--log_dir", type=str, default="logs/seg/Cassini_seg_baseline.txt")
    args = parser.parse_args()
    return args

def main(args):
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if not os.path.exists(args.fig_savedir):
        os.mkdir(args.fig_savedir)
    # dataset
    dataset = Dataset_aggregation(
        args.cassini, args.gt_forest, args.gt_hydro, args.gt_road, args.gt_town, transform=transform
    )
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    # model
    G12_seg = unet(in_channel=3, out_channel=5, hidden_dim=args.hidden_dim_seg, \
        norm_type=args.norm_type_G).to(device) # from IGN to labels
    
    print("loading pretrained models")    
    model_seg = torch.load(args.model_dir)
    G12_seg.load_state_dict(model_seg["G12_state_dict"])
    # loss function

    G12_seg.eval()

    loss_forest = []
    loss_hydro = []
    loss_road = []
    loss_town = []
    
    # logger
    logger = logging.getLogger("Cassini Seg Model")
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Using the classifier trained with gt derived from transferred data")
    logger.info("start testing")

    with torch.no_grad():
        for cassini, gt_forest, gt_hydro, gt_road, gt_town, cityname, basename in tqdm(dataloader):
            cassini = cassini.float().to(device)
            
            gt_forest = gt_forest.float().to(device).squeeze()
            gt_hydro = gt_hydro.float().to(device).squeeze()
            gt_road = gt_road.float().to(device).squeeze()
            gt_town = gt_town.float().to(device).squeeze()

            pred = G12_seg(cassini)[0].permute(1,2,0).cpu().numpy()
            label = np.argmax(pred, axis=-1)

            # save results
            savepath = os.path.join(args.fig_savedir, cityname[0])
            if not os.path.exists(savepath):
                os.mkdir(savepath)
            plot_figure2(cassini[0], os.path.join(savepath, f"cassini.png"))
            forest = np.zeros_like(label); hydro = np.zeros_like(label); road = np.zeros_like(label); town = np.zeros_like(label)
            forest[label==1]=255; hydro[label==2]=255; road[label==3]=255; town[label==4]=255
            Image.fromarray((forest).astype(np.uint8)).save(os.path.join(savepath, f"forest_pred.png"))
            Image.fromarray((hydro).astype(np.uint8)).save(os.path.join(savepath, f"hydro_pred.png"))
            Image.fromarray((road).astype(np.uint8)).save(os.path.join(savepath, f"road_pred.png"))
            Image.fromarray((town).astype(np.uint8)).save(os.path.join(savepath, f"town_pred.png"))

            if cityname[0] in forest_names:
                loss_forest.append(dilatedIoU(label, gt_forest.cpu().numpy(), label=1, window=5)) 
                Image.fromarray((255 *gt_forest.cpu().numpy()).astype(np.uint8)).save(os.path.join(savepath, f"forest_gt.png"))
            if cityname[0] in hydro_names:
                loss_hydro.append(dilatedIoU(label, 2*gt_hydro.cpu().numpy(), label=2, window=5)) 
                Image.fromarray((255 * gt_hydro.cpu().numpy()).astype(np.uint8)).save(os.path.join(savepath, f"hydro_gt.png"))
            loss_road.append(dilatedIoU(label, 3*gt_road.cpu().numpy(), label=3, window=5)) 
            Image.fromarray((255 *gt_road.cpu().numpy()).astype(np.uint8)).save(os.path.join(savepath, f"road_gt.png"))   
            loss_town.append(dilatedIoU(label, 4*gt_town.cpu().numpy(), label=4, window=5))
            Image.fromarray((255 * gt_town.cpu().numpy()).astype(np.uint8)).save(os.path.join(savepath, f"town_gt.png"))
        
    print("Mean forest dilated IoU = ", np.mean(loss_forest))
    print("Mean hydro dilated IoU = ", np.mean(loss_hydro))
    print("Mean road dilated IoU = ", np.mean(loss_road))
    print("Mean town dilated IoU = ", np.mean(loss_town))
    logger.info("window=5")
    logger.info("[forest miou: %f], [hydro miou: %f], [road miou: %f], [town miou: %f]"
                    % (np.mean(loss_forest), np.mean(loss_hydro), np.mean(loss_road), np.mean(loss_town)))
if __name__ == "__main__":
    args = argparser()
    main(args)
    ## Aggregation of hydro_areas and hydro_network
    # from PIL import Image
    # import glob
    # import numpy as np
    # p1 = "data/cassini_gt_sheet/hydro_areas"
    # p2 = "data/cassini_gt_sheet/hydro_network"
    # out = "data/cassini_gt_sheet/hydro"
    # if not os.path.exists(out):
    #     os.mkdir(out)

    # files1 = sorted(os.listdir(p1))
    # files2 = sorted(os.listdir(p2))
    # files_pair = list(zip(files1, files2))
    # for file1, file2 in tqdm(files_pair):
    #     assert file1 == file2, print(file1, file2)
    #     image1 = np.array(Image.open(glob.glob(os.path.join(p1, file1, "*.tif"))[0]))
    #     image2 = np.array(Image.open(glob.glob(os.path.join(p2, file2, "*.tif"))[0]))
    #     image1[np.where(image2 == 255)] = 255
    #     basename = file1
    #     out_dir = os.path.join(out, file1)
    #     if not os.path.exists(out_dir):
    #         os.mkdir(out_dir)
    #     Image.fromarray(image1).save(os.path.join(out_dir, basename+".tif"))

