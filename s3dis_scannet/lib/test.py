import logging
import os
import shutil
import tempfile
import warnings
import platform

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from lib.utils import Timer, AverageMeter, precision_at_one, fast_hist, per_class_iu, \
    get_prediction, get_torch_device, save_predictions, visualize_results, \
    permute_pointcloud, save_rotation_pred

from MinkowskiEngine import SparseTensor

from tqdm import tqdm

def print_info(iteration,
               max_iteration,
              #  data_time,
              #  iter_time,
               has_gt=False,
               losses=None,
               scores=None,
               ious=None,
               hist=None,
               ap_class=None,
               class_names=None):
  debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
  # debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

  if has_gt:
    acc = hist.diagonal() / hist.sum(1) * 100
    debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
        "Score {top1.val:.3f} (AVG: {top1.avg:.3f})\t" \
        "mIOU {mIOU:.3f} mAP {mAP:.3f} mAcc {mAcc:.3f}\n".format(
            loss=losses, top1=scores, mIOU=np.nanmean(ious),
            mAP=np.nanmean(ap_class), mAcc=np.nanmean(acc))
    if class_names is not None:
      debug_str += "\nClasses: " + " ".join(class_names) + '\n'
    debug_str += 'IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n'
    debug_str += 'mAP: ' + ' '.join('{:.03f}'.format(i) for i in ap_class) + '\n'
    debug_str += 'mAcc: ' + ' '.join('{:.03f}'.format(i) for i in acc) + '\n'

  logging.info(debug_str)


def average_precision(prob_np, target_np):
  num_class = prob_np.shape[1]
  label = label_binarize(target_np, classes=list(range(num_class)))
  with np.errstate(divide='ignore', invalid='ignore'):
    return average_precision_score(label, prob_np, average=None)

import time
def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

def test(model, train_data_loader, data_loader, config, transform_data_fn=None, has_gt=True):
  if config.bitcount_export:
    data_loader = train_data_loader
  device = get_torch_device(config.is_cuda)
  dataset = data_loader.dataset
  num_labels = dataset.NUM_LABELS
  # global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
  criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)
  losses, scores, ious = AverageMeter(), AverageMeter(), 0
  aps = np.zeros((0, num_labels))
  hist = np.zeros((num_labels, num_labels))

  logging.info('===> Start testing')

  # global_timer.tic()
  data_iter = data_loader.__iter__()
  max_iter = len(data_loader)
  max_iter_unique = max_iter

  # Fix batch normalization running mean and std
  model.eval()

  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()

  if config.save_prediction or config.test_original_pointcloud:
    if config.save_prediction:
      save_pred_dir = config.save_pred_dir
      os.makedirs(save_pred_dir, exist_ok=True)
    else:
      save_pred_dir = tempfile.mkdtemp()
    if os.listdir(save_pred_dir):
      raise ValueError(f'Directory {save_pred_dir} not empty. '
                       'Please remove the existing prediction.')

  if config.fp16:
    print("[NANM] FP16 enabled")

  print("\033[104m[NANM] Backend is set as", config.backend, "\033[0m")

  
  warmup = 30 if 'Stanford' in config.dataset else 10

  times = []
  with torch.no_grad():
    repeat = 1 if 'Stanford' in config.dataset else 1
    print("[NANM] Repeating for \33[105m", repeat, "\033[0m times")
    for i in range(repeat):
      data_iter = data_loader.__iter__()
      print("{}th iteration...".format(i))
      for iteration in tqdm(range(max_iter)):
        # data_timer.tic()
        if config.return_transformation:
          coords, input, target, transformation = data_iter.next()
        else:
          coords, input, target = data_iter.next()
          transformation = None
          
        # data_time = data_timer.toc(False)
  
        # Preprocess input
        # iter_timer.tic()
  
        if config.wrapper_type != 'None':
          color = input[:, :3].int()
        if config.normalize_color:
          input[:, :3] = input[:, :3] / 255. - 0.5
  
        if config.backend == 'spconv':
          # Spconv
          import spconv.pytorch as spconv
          sparse_shape = (torch.max(coords[:, 1:], dim=0).values + 1).cpu().numpy()
          sinput = spconv.SparseConvTensor(input.to(device), coords.to(device), sparse_shape, torch.max(coords[:, 0]).item() + 1)
        elif config.backend == 'mink':
          # Minkowski
          sinput = SparseTensor(input, coords, device=device)
        elif config.backend == 'torchsparse':
          # Torchsparse
          import torchsparse
          coords = coords.index_select(1, torch.LongTensor([1, 2, 3, 0])).int()
          sinput = torchsparse.SparseTensor(input, coords).to(device)
        else:
          raise Exception(f"{config.backend} backend not supported")

        # Feed forward
        inputs = (sinput,) if config.wrapper_type == 'None' else (sinput, coords, color)
        
        with torch.cuda.amp.autocast(enabled=config.fp16):
          # Warm up 10 iter
          if iteration == 0:
            for _ in range(warmup):
              _ = model(*inputs)
          
          start_time = cuda_time()
          soutput = model(*inputs)
          times.append(cuda_time()-start_time)

          # Only calculate miou 1 time
          if i != 0 or config.bitcount_export:
            continue

          output = soutput.features if config.backend != 'torchsparse' else soutput.F
    
          pred = get_prediction(dataset, output, target).int()
          # iter_time = iter_timer.toc(False)
    
          if config.save_prediction or config.test_original_pointcloud:
            save_predictions(coords, pred, transformation, dataset, config, iteration, save_pred_dir)
  
          ######################## Comment out for w/o gemm ###########################
          if has_gt:
            if config.evaluate_original_pointcloud:
              raise NotImplementedError('pointcloud')
              output, pred, target = permute_pointcloud(coords, pointcloud, transformation,
                                                        dataset.label_map, output, pred)
    
            target_np = target.numpy()
    
            num_sample = target_np.shape[0]
    
            target = target.to(device)
    
            cross_ent = criterion(output, target.long())
            losses.update(float(cross_ent), num_sample)
            scores.update(precision_at_one(pred, target), num_sample)
            hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels)
            ious = per_class_iu(hist) * 100
    
            prob = torch.nn.functional.softmax(output, dim=1)
            ap = average_precision(prob.cpu().detach().numpy(), target_np)
            aps = np.vstack((aps, ap))
            # Due to heavy bias in class, there exists class with no test label at all
            with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=RuntimeWarning)
              ap_class = np.nanmean(aps, 0) * 100.
    
          if iteration % config.test_stat_freq == 0 and iteration > 0:
            reordered_ious = dataset.reorder_result(ious)
            reordered_ap_class = dataset.reorder_result(ap_class)
            class_names = dataset.get_classnames()
            print_info(
                iteration,
                max_iter_unique,
                # data_time,
                # iter_time,
                has_gt,
                losses,
                scores,
                reordered_ious,
                hist,
                reordered_ap_class,
                class_names=class_names)
          ###############################################################################
    
          # if iteration % config.empty_cache_freq == 0:
          #   # Clear cache
          #   torch.cuda.empty_cache()
      
      if config.bitcount_export:
        if config.fp16==False:
          def generate_weight_masks(bitcount_l, key, is_prune, average_weight_sum):
            bitcount = torch.zeros(bitcount_l[0]['bitcount'].shape[1], device=bitcount_l[0]['bitcount'].device)
            for bc in bitcount_l:
              bitcount += torch.sum(bc['bitcount'], dim=0)

            # Load weight value
            weights = torch.load(config.weights)
            weight = weights['state_dict'][key]
            if weight.shape[1:4]==torch.Size((3,3,3)):
                a, b, c, d, e = weight.shape
                temp = torch.reshape(weight, (a, 27, e))
                weight_sum = torch.sum(torch.abs(temp), (2, 0))
            else:
                weight_sum = torch.zeros(8)

            indices = list(np.argsort(bitcount.cpu().numpy()))
            indices_weight_sum = list(np.argsort(weight_sum.cpu().numpy()))
            indices_average_weight_sum = list(np.argsort(average_weight_sum.cpu().numpy()))
            
            weight_masks = {}
            weight_masks_magnitude = {}
            weight_masks_exp = {}
            weight_masks_exp_weight_magnitude = {}
            
            if key == 'conv0p1s1.weight':
              number = 0
            elif key == 'block1.0.conv1.weight':
              number = 1
            elif key == 'block2.0.conv1.weight':
              number = 2
            elif key == 'block3.0.conv1.weight':
              number = 3
            elif key == 'block4.0.conv1.weight':
              number = 4
            else:
              number = -1   # Do not need to be saved

            if len(indices) == 27 and is_prune:
              mask = 134217727  # 111111111 111111111 111111111 (27 bits)
              for i, ind in enumerate(indices):
                  weight_masks_exp[i] = mask & ~(1<<ind)
            else:
              for i, ind in enumerate(indices):
                  weight_masks_exp[i] = -1

            weight_masks_exp[27] = -1  # Mask for not pruning

            if len(indices_weight_sum) == 27 and is_prune:
              mask = 134217727  # 111111111 111111111 111111111 (27 bits)
              for i, ind in enumerate(indices_weight_sum):
                  weight_masks_exp_weight_magnitude[i] = mask & ~(1<<ind)
            else:
              for i, ind in enumerate(indices_weight_sum):
                  weight_masks_exp_weight_magnitude[i] = -1
            
            weight_masks_exp_weight_magnitude[27] = -1  # Mask for not pruning

            if len(indices) == 27:
              mask = 0
              for i, ind in enumerate(reversed(indices)):
                  mask |= (1<<ind)
                  if i==4:
                      weight_masks[5] = mask
                  elif i==6:
                      weight_masks[4] = mask
                  elif i==10:
                      weight_masks[3] = mask
                  elif i==14:
                      weight_masks[2] = mask
                  elif i==18:
                      weight_masks[1] = mask
                  elif i==26:
                      weight_masks[0] = -1
            else:
              weight_masks[5] = 255
              weight_masks[4] = 255
              weight_masks[3] = 255
              weight_masks[2] = 255
              weight_masks[1] = 255
              weight_masks[0] = -1

            if len(indices_weight_sum) == 27:
              mask = 0
              for i, ind in enumerate(reversed(indices_weight_sum)):
                  mask |= (1<<ind)
                  if i==4:
                      weight_masks_magnitude[5] = mask
                  elif i==6:
                      weight_masks_magnitude[4] = mask
                  elif i==10:
                      weight_masks_magnitude[3] = mask
                  elif i==14:
                      weight_masks_magnitude[2] = mask
                  elif i==18:
                      weight_masks_magnitude[1] = mask
                  elif i==26:
                      weight_masks_magnitude[0] = -1
            else:
              weight_masks_magnitude[5] = 255
              weight_masks_magnitude[4] = 255
              weight_masks_magnitude[3] = 255
              weight_masks_magnitude[2] = 255
              weight_masks_magnitude[1] = 255
              weight_masks_magnitude[0] = -1

            print()

            return weight_masks, weight_masks_magnitude, weight_masks_exp, weight_masks_exp_weight_magnitude

          weight_masks = {}
          weight_masks_magnitude = {}
          weight_masks_exp = {}
          weight_masks_exp_weight_magnitude = {}
          # weight_indices = {}
          weight_masks['conv0p1s1'], weight_masks_magnitude['conv0p1s1'], weight_masks_exp['conv0p1s1'], weight_masks_exp_weight_magnitude['conv0p1s1'] = generate_weight_masks(model.conv0p1s1.yj_bitcount_export, 'conv0p1s1.weight', False, torch.zeros(27))
          weight_masks['conv1p1s2'], weight_masks_magnitude['conv1p1s2'], weight_masks_exp['conv1p1s2'], weight_masks_exp_weight_magnitude['conv1p1s2'] = generate_weight_masks(model.conv1p1s2.yj_bitcount_export, 'conv1p1s2.weight', False, torch.zeros(8))
          weight_masks['conv2p2s2'], weight_masks_magnitude['conv2p2s2'], weight_masks_exp['conv2p2s2'], weight_masks_exp_weight_magnitude['conv2p2s2'] = generate_weight_masks(model.conv2p2s2.yj_bitcount_export, 'conv2p2s2.weight', False, torch.zeros(8))
          weight_masks['conv3p4s2'], weight_masks_magnitude['conv3p4s2'], weight_masks_exp['conv3p4s2'], weight_masks_exp_weight_magnitude['conv3p4s2'] = generate_weight_masks(model.conv3p4s2.yj_bitcount_export, 'conv3p4s2.weight', False, torch.zeros(8))
          weight_masks['conv4p8s2'], weight_masks_magnitude['conv4p8s2'], weight_masks_exp['conv4p8s2'], weight_masks_exp_weight_magnitude['conv4p8s2'] = generate_weight_masks(model.conv4p8s2.yj_bitcount_export, 'conv4p8s2.weight', False, torch.zeros(8))

          assert(len(model.LAYERS)==8)

          # Calculate average weight_sum for each prune level
          average_weight_sum = [0, 0, 0, 0, 0]
          for i in range(len(model.LAYERS)):
            if i == 0 or i == 6:
              level = 1
            elif i == 1 or i == 5:
              level = 2
            elif i == 2 or i == 4:
              level = 3
            elif i == 3:
              level = 4
            elif i == 7:
              level = 0

            weights = torch.load(config.weights)
            for j in range(model.LAYERS[i]):
                weight1 = weights['state_dict']['block'+str(i+1)+'.'+str(j)+'.'+'conv1.weight']
                weight2 = weights['state_dict']['block'+str(i+1)+'.'+str(j)+'.'+'conv2.weight']
                a, _, _, _, b = weight1.shape
                c, _, _, _, d = weight2.shape
                temp1 = torch.reshape(weight1, (a, 27, b))
                temp2 = torch.reshape(weight2, (c, 27, d))
                average_weight_sum[level] += torch.sum(torch.abs(temp1), (2, 0))
                average_weight_sum[level] += torch.sum(torch.abs(temp2), (2, 0))

          for i in range(len(model.LAYERS)):
            if i == 0:
              block = model.block1
              level = 1
            elif i == 1:
              block = model.block2
              level = 2
            elif i == 2:
              block = model.block3
              level = 3
            elif i == 3:
              block = model.block4
              level = 4
            elif i == 4:
              block = model.block5
              level = 3
            elif i == 5:
              block = model.block6
              level = 2
            elif i == 6:
              block = model.block7
              level = 1
            elif i == 7:
              block = model.block8
              level = 0 

            weight_masks['block'+str(i+1)] = dict()
            weight_masks_magnitude['block'+str(i+1)] = dict()
            weight_masks_exp['block'+str(i+1)] = dict()
            weight_masks_exp_weight_magnitude['block'+str(i+1)] = dict()
            # weight_indices['block'+str(i+1)] = dict()
            for j in range(model.LAYERS[i]):
              weight_masks['block'+str(i+1)][str(j)] = dict()
              weight_masks_magnitude['block'+str(i+1)][str(j)] = dict()
              weight_masks_exp['block'+str(i+1)][str(j)] = dict()
              weight_masks_exp_weight_magnitude['block'+str(i+1)][str(j)] = dict()
              weight_masks['block'+str(i+1)][str(j)]['conv1'], weight_masks_magnitude['block'+str(i+1)][str(j)]['conv1'], weight_masks_exp['block'+str(i+1)][str(j)]['conv1'], weight_masks_exp_weight_magnitude['block'+str(i+1)][str(j)]['conv1'] = generate_weight_masks(block[j].conv1.yj_bitcount_export, 'block'+str(i+1)+'.'+str(j)+'.'+'conv1.weight', True, average_weight_sum[level])
              weight_masks['block'+str(i+1)][str(j)]['conv2'], weight_masks_magnitude['block'+str(i+1)][str(j)]['conv2'], weight_masks_exp['block'+str(i+1)][str(j)]['conv2'], weight_masks_exp_weight_magnitude['block'+str(i+1)][str(j)]['conv2'] = generate_weight_masks(block[j].conv2.yj_bitcount_export, 'block'+str(i+1)+'.'+str(j)+'.'+'conv2.weight', True, average_weight_sum[level])
              # weight_indices['block'+str(i+1)][str(j)] = dict()
              # weight_indices['block'+str(i+1)][str(j)]['conv1'] = list(np.argsort(block[j].conv1.yj_bitcount_export.cpu().numpy()))
              # weight_indices['block'+str(i+1)][str(j)]['conv2'] = list(np.argsort(block[j].conv2.yj_bitcount_export.cpu().numpy()))
          
        import pickle
        #  with open("/arc-share/pc_yejin/weight_masks/"+config.dataset+'_'+config.model+'_weight_mask_trainset.pickle', 'wb') as handle:
        #      pickle.dump(weight_masks, handle)
        #      print("weight_masks: ", weight_masks)
#          with open("/arc-share/pc_yejin/weight_masks/"+config.dataset+'_'+config.model+'_weight_mask_trainset_magnitude.pickle', 'wb') as handle:
#              pickle.dump(weight_masks_magnitude, handle)
#              print("weight_masks_magnitude: ", weight_masks_magnitude)
#          with open("/arc-share/pc_yejin/weight_masks/"+config.dataset+'_'+config.model+'_weight_mask_trainset_exp.pickle', 'wb') as handle:
#              pickle.dump(weight_masks_exp, handle)
#              print("weight_masks_exp: ", weight_masks_exp)
#          with open("/arc-share/pc_yejin/weight_masks/"+config.dataset+'_'+config.model+'_weight_mask_trainset_exp_weight_magnitude.pickle', 'wb') as handle:
#              pickle.dump(weight_masks_exp_weight_magnitude, handle)
#              print("weight_masks_exp_weight_magnitude: ", weight_masks_exp_weight_magnitude)

          # with open("/arc-share/pc_yejin/weight_masks/"+config.dataset+'_'+config.model+'_weight_indices_trainset.pickle', 'wb') as handle:
          #     pickle.dump(weight_indices, handle)
          #     print("weight_indices: ", weight_indices)

        print("[NANM] length of bitcount_export: ", len(model.conv0p1s1.yj_bitcount_export))
        bitcount_export = {}
        bitcount_export['conv0p1s1'] = model.conv0p1s1.yj_bitcount_export
        bitcount_export['conv1p1s2'] = model.conv1p1s2.yj_bitcount_export
        bitcount_export['conv2p2s2'] = model.conv2p2s2.yj_bitcount_export
        bitcount_export['conv3p4s2'] = model.conv3p4s2.yj_bitcount_export
        bitcount_export['conv4p8s2'] = model.conv4p8s2.yj_bitcount_export

        bitcount_export['convtr4p16s2'] = model.convtr4p16s2.yj_bitcount_export
        bitcount_export['convtr5p8s2']  = model.convtr5p8s2.yj_bitcount_export
        bitcount_export['convtr6p4s2']  = model.convtr6p4s2.yj_bitcount_export
        bitcount_export['convtr7p2s2']  = model.convtr7p2s2.yj_bitcount_export

        for i in range(len(model.LAYERS)):
          if i == 0:
            block = model.block1
          elif i == 1:
            block = model.block2
          elif i == 2:
            block = model.block3
          elif i == 3:
            block = model.block4
          elif i == 4:
            block = model.block5
          elif i == 5:
            block = model.block6
          elif i == 6:
            block = model.block7
          elif i == 7:
            block = model.block8

          bitcount_export['block'+str(i+1)] = dict()
          for j in range(model.LAYERS[i]):
            bitcount_export['block'+str(i+1)][str(j)] = dict()
            bitcount_export['block'+str(i+1)][str(j)]['conv1'] = block[j].conv1.yj_bitcount_export
            bitcount_export['block'+str(i+1)][str(j)]['conv2'] = block[j].conv2.yj_bitcount_export

        import pickle
        if config.fp16:
          with open('/arc-share/pc_yejin/saved_tune_res/'+os.uname()[1]+'/'+config.dataset+'_'+config.model+'_bitcount_per_kv_'+os.uname()[1]+'_train_set_fp16.pickle', 'wb') as handle:
              pickle.dump(bitcount_export, handle)
              print("Bitcount per chunk: ", bitcount_export)
        else:
          with open('/arc-share/pc_yejin/saved_tune_res/'+os.uname()[1]+'/'+config.dataset+'_'+config.model+'_bitcount_per_kv_'+os.uname()[1]+'_train_set.pickle', 'wb') as handle:
              pickle.dump(bitcount_export, handle)
              print("Bitcount per chunk: ", bitcount_export)
        exit(0)

  print(f"\033[93m[NANM] FPS: {1/np.mean(times):2f}\033[0m")
  print(f"\033[93m[NANM] Time (ms): {np.sum(times)*1000:.2f}\033[0m")

  # global_time = global_timer.toc(False)

  #################### Comment out for w/o gemm ###########################
  reordered_ious = dataset.reorder_result(ious)
  reordered_ap_class = dataset.reorder_result(ap_class)
  class_names = dataset.get_classnames()
  print_info(
      iteration,
      max_iter_unique,
      # data_time,
      # iter_time,
      has_gt,
      losses,
      scores,
      reordered_ious,
      hist,
      reordered_ap_class,
      class_names=class_names)

  if config.test_original_pointcloud:
    logging.info('===> Start testing on original pointcloud space.')
    dataset.test_pointcloud(save_pred_dir)
  ##########################################################################

  # logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

  return losses.avg, scores.avg, np.nanmean(ap_class), np.nanmean(per_class_iu(hist)) * 100
