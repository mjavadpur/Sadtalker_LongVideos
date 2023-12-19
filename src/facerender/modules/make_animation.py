from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
from src.utils.paste_pic import paste_pic_stream
import cv2
import os
from skimage import img_as_ubyte
from pathlib import Path

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, 1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical['value']    # (bs, k, 3) 
    yaw, pitch, roll= he['yaw'], he['pitch'], he['roll']      
    yaw = headpose_pred_to_degree(yaw) 
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he:
        yaw = he['yaw_in']
    if 'pitch_in' in he:
        pitch = he['pitch_in']
    if 'roll_in' in he:
        roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    if wo_exp:
        exp =  exp*0  
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0]*0
    t[:, 2] = t[:, 2]*0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {'value': kp_transformed}

def remove_directory_contents(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Iterate over the files in the directory and remove them
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f"Removed file: {file_path}")
            elif os.path.isdir(file_path):
                # Recursively remove subdirectories
                remove_directory_contents(file_path)
                # Remove the empty directory
                os.rmdir(file_path)
                print(f"Removed directory: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

def make_animation(args, audio_path, save_dir, video_name, img_size, crop_info, source_image, source_semantics, target_semantics,
                            generator, kp_detector, he_estimator, mapping, 
                            yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                            use_exp=True, use_half=False):
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_dir = Path(temp_dir)/'sad'
    # temp_dir = Path.cwd()/'sad'
    print('temp dir',temp_dir)
    frame_dir = temp_dir/'frames'
    remove_directory_contents(str(temp_dir))
    frame_dir.mkdir(exist_ok=True, parents=True)
    print(f'tempdir: {temp_dir}\nframedir: {frame_dir}')
    with torch.no_grad():
        kp_canonical = kp_detector(source_image)
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source)
    
        for frame_idx in tqdm(range(target_semantics.shape[1]), 'Face Renderer:'):
            # still check the dimension
            target_semantics_frame = target_semantics[:, frame_idx]
            he_driving = mapping(target_semantics_frame)
            if yaw_c_seq is not None:
                he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
            if pitch_c_seq is not None:
                he_driving['pitch_in'] = pitch_c_seq[:, frame_idx] 
            if roll_c_seq is not None:
                he_driving['roll_in'] = roll_c_seq[:, frame_idx] 
            
            kp_driving = keypoint_transformation(kp_canonical, he_driving)
                
            kp_norm = kp_driving
            out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
            video=[]
            for img in out['prediction']:
                image = np.transpose(img.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
                video.append(image)
            result = img_as_ubyte(video)
            original_size = crop_info[0]
            if original_size:
                # print(f'original size: {original_size}. resizing...')
                result = [ cv2.resize(result_i,(img_size, int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]
            for i, frame in enumerate(result):
                cv2.imwrite(str(frame_dir/f'{i}_{frame_idx:04d}.png'), frame[:,:,::-1])

        # write png to mp4
        size1, size2= frame.shape[:2]
        path = os.path.join(str(save_dir), 'temp_' + video_name + '.avi')
        print(f'video size {size1, size2}')
        openVideo = cv2.VideoWriter(path, 
                                cv2.VideoWriter_fourcc(*'DIVX'), 25, (size2, size1))
        # openVideo = cv2.VideoWriter(path, -1, 25, (size2, size1))
        for pngFile in frame_dir.iterdir():
            if pngFile.suffix!=".png": continue
            f = cv2.imread(str(pngFile))
            # print(f)
            openVideo.write(f)
        openVideo.release()
        print(f'succesfully wrote png to mp4 at {path}')
        # video_name_full = video_name + '_full.mp4'
        # full_video_path = temp_dir/video_name_full
        # print(f'full_video_path final video: {full_video_path}')
        # new_audio_path = audio_path
        # return_path = full_video_path
        print('Pasting faces back into frame (SeamlessClone)')
        full_video_path = paste_pic_stream(temp_dir, path, args.source_image, crop_info, extended_crop= True if 'ext' in args.preprocess.lower() else False)
        print(f'full video path {full_video_path}')
        full_video_path_final = temp_dir/f'{video_name}_full_audio.mp4'
        #     predictions.append(out['prediction'])
        # predictions_ts = torch.stack(predictions, dim=1)
        import subprocess
        import platform
        print(f'final video {full_video_path_final}')
        command = 'ffmpeg -y -init_hw_device cuda -hwaccel nvdec -hwaccel_output_format cuda -i {} -i {} -c:v h264_nvenc -preset:v p1 {}'.format(audio_path, str(full_video_path), str(full_video_path_final))
        subprocess.call(command, shell=platform.system() != 'Windows')
    return full_video_path_final, temp_dir

class AnimateModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        
        source_image = x['source_image']
        source_semantics = x['source_semantics']
        target_semantics = x['target_semantics']
        yaw_c_seq = x['yaw_c_seq']
        pitch_c_seq = x['pitch_c_seq']
        roll_c_seq = x['roll_c_seq']

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor,
                                        self.mapping, use_exp = True,
                                        yaw_c_seq=yaw_c_seq, pitch_c_seq=pitch_c_seq, roll_c_seq=roll_c_seq)
        
        return predictions_video