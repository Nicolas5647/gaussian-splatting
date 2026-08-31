import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import threading
import argparse
import warnings
from pathlib import Path
import cv2
import numpy as np
import torch
import gc
from tqdm import tqdm
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Utilisation du GPU : {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("Utilisation du CPU (CUDA non disponible)")


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)

ESCAPE_KEY = 27
ENTER_KEY = 13

def clean_dir(path):
    if not os.path.exists(path):
        return
    
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def x1y1x2y2_to_xywh(box, w, h):
    return [box[0] / w, box[1] / h, (box[2] - box[0]) / w, (box[3] - box[1]) / h]


def scale_point(pt, w, h):
    return [pt[0] / w, pt[1] / h]


def get_inference_state(predictor, session_id):
    if hasattr(predictor, 'sessions') and session_id in predictor.sessions:
        session_data = predictor.sessions[session_id]
        if isinstance(session_data, dict):
            return session_data.get("state")
        elif hasattr(session_data, "state"):
            return getattr(session_data, "state")
    return None


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_cpu(x) for x in obj)
    return obj


def make_empty_frame_state(active_id = 1):
    return {
        'object': {active_id: {'points': [], 'labels': [], 'boxes': []}},
        'current_masks': {},
        'ix': -1, 'iy': -1, 
        'drawing': False
    }


def send_prompt_to_sam3(predictor, session_id, state, active_id, frame_idx):
    obj_data = state['object'][active_id]
    response = None
    request_args = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": frame_idx,
        "obj_id": active_id
    }

    has_prompt = False
    if len(obj_data['boxes']) > 0:
        boxes_tensor = torch.tensor(obj_data['boxes'], dtype=torch.float32)
        labels_tensor = torch.ones(len(obj_data['boxes']), dtype=torch.int32)
        
        request_args["bounding_boxes"] = boxes_tensor
        request_args["bounding_box_labels"] = labels_tensor
        has_prompt = True

    if len(obj_data['points']) > 0:
        pts_tensor = torch.tensor(obj_data['points'], dtype=torch.float32)
        labels_tensor = torch.tensor(obj_data['labels'], dtype=torch.int32)
        
        request_args["points"] = pts_tensor
        request_args["point_labels"] = labels_tensor
        has_prompt = True

    if not has_prompt:
        return

    try:
        with torch.inference_mode():
            response = predictor.handle_request(request=request_args)
    except Exception as e:
        print(f"\n[ERROR] Échec lors de l'envoi du prompt à SAM 3 (Frame {frame_idx}) : {e}")
        import traceback
        traceback.print_exc()
        return

    if response:
        outputs = response.get("outputs", {})
        if "out_binary_masks" in outputs and "out_obj_ids" in outputs:
            out_masks_np = to_cpu(outputs["out_binary_masks"])
            out_ids = to_cpu(outputs["out_obj_ids"])

            state['current_masks'] = {}
            if out_masks_np is not None and out_ids is not None:
                for i, o_id in enumerate(out_ids):
                    state['current_masks'][int(o_id)] = out_masks_np[i]

def draw_callback(event, x, y, flags, param):
    if param.get('is_propagating', [False])[0]:
        return
    
    state = param['state']
    active_id = param['active_id']
    width = param['width']
    height = param['height']
    scale = param['scale']
    predictor = param['predictor']
    session_id = param['session_id']
    frame_idx = param['frame_idx']
    disp_w = int(width * scale)
    disp_h = int(height * scale)

    if frame_idx not in state:
        state[frame_idx] = make_empty_frame_state(active_id)

    is_point = False
    for state_frame in state.values():
        for active_frame in state_frame["object"].values():
            if len(active_frame['points']) > 0:
                is_point = True
                break


    frame_state = state[frame_idx]
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_state['drawing'] = True
        frame_state['ix'] = max(0, min(x, disp_w - 1))
        frame_state['iy'] = max(0, min(y, disp_h - 1))

    elif event == cv2.EVENT_MOUSEMOVE and frame_state['drawing']:
        img_copy = param['base_img'].copy()
        cx = max(0, min(x, disp_w - 1))
        cy = max(0, min(y, disp_h - 1))
        cv2.rectangle(img_copy, (frame_state['ix'], frame_state['iy']), (cx, cy), (0, 255, 0), 2)
        cv2.imshow("SAM 3 - Annotation", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        frame_state['drawing'] = False
        cx = max(0, min(x, disp_w - 1))
        cy = max(0, min(y, disp_h - 1))
        
        dx = abs(cx - frame_state['ix'])
        dy = abs(cy - frame_state['iy'])

        if is_point or dx < 15 and dy < 15:
            px = int(cx / scale)
            py = int(cy / scale)
            px = max(0, min(px, width - 1))
            py = max(0, min(py, height - 1))
            scaled_pt = scale_point((px, py), width, height)
            frame_state['object'][active_id]['points'].append(scaled_pt)
            frame_state['object'][active_id]['labels'].append(1)
        elif dx > 15 and dy > 15:
            x1 = int(min(frame_state['ix'], cx) / scale)
            y1 = int(min(frame_state['iy'], cy) / scale)
            x2 = int(max(frame_state['ix'], cx) / scale)
            y2 = int(max(frame_state['iy'], cy) / scale)
            
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                rel_box = x1y1x2y2_to_xywh([x1, y1, x2, y2], width, height)
                frame_state['object'][active_id]['boxes'].append(rel_box)
        send_prompt_to_sam3(predictor, session_id, frame_state, active_id, frame_idx)
        param['update_display']()

    elif event == cv2.EVENT_RBUTTONDOWN:
        cx = max(0, min(x, disp_w - 1))
        cy = max(0, min(y, disp_h - 1))
        px = int(cx / scale)
        py = int(cy / scale)
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))
        scaled_pt = scale_point((px, py), width, height)
        frame_state['object'][active_id]['points'].append(scaled_pt)
        frame_state['object'][active_id]['labels'].append(0)

        send_prompt_to_sam3(predictor, session_id, frame_state, active_id, frame_idx)
        param['update_display']()

def update_display_fn(img, state, active_id, disp_w, disp_h, callback_params, frame_idx, is_propagating=False):
    disp_img = cv2.resize(img, (disp_w, disp_h))
    mask_overlay = np.zeros_like(disp_img)
    active_mask_overlay = np.zeros_like(disp_img)

    for obj_id, mask in state['current_masks'].items():
        if mask is not None:
            mask_resized = cv2.resize(mask.astype(np.uint8), (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            if obj_id != active_id:
                mask_overlay[mask_resized > 0] = YELLOW
            else:
                active_mask_overlay[mask_resized > 0] = GREEN

    has_inactive = np.any(mask_overlay > 0, axis=-1)
    if np.any(has_inactive):
        blended_inactive = cv2.addWeighted(disp_img, 0.7, mask_overlay, 0.3, 0)
        disp_img[has_inactive] = blended_inactive[has_inactive]

    has_active = np.any(active_mask_overlay > 0, axis=-1)
    if np.any(has_active):
        blended_active = cv2.addWeighted(disp_img, 0.7, active_mask_overlay, 0.3, 0)
        disp_img[has_active] = blended_active[has_active]

    obj_data = state['object'][active_id]
    
    for box in obj_data['boxes']:
        rx, ry, rw, rh = box
        bx1 = int(rx * disp_w)
        by1 = int(ry * disp_h)
        bw = int(rw * disp_w)
        bh = int(rh * disp_h)
        cv2.rectangle(disp_img, (bx1, by1), (bx1 + bw, by1 + bh), GREEN, 2)

    for pt, label in zip(obj_data['points'], obj_data['labels']):
        px, py = int(pt[0] * disp_w), int(pt[1] * disp_h)
        pt_color = GREEN if label == 1 else RED 
        cv2.circle(disp_img, (px, py), 5, pt_color, -1)

    cv2.putText(
        disp_img, f"Obj {active_id} | Boites: {len(obj_data['boxes'])} | Points: {len(obj_data['points'])}", 
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    cv2.putText(
        disp_img, f"Frame: {frame_idx}",
        (disp_w - 125, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    if is_propagating:
        propagated_count = callback_params.get('current_propagation_frame', [0])[0]
        total_frames = callback_params.get('total_frames', 0)
        cv2.putText(
            disp_img, "Propagation en cours, seulement la navigation est possible",
            (15, disp_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)
        cv2.putText(
            disp_img, f"{propagated_count}/{total_frames}",
            (15, disp_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)
    else:   
        cv2.putText(
            disp_img, "Clic G: + | Clic D: - | Glisser: Tracer | Entree: Propager | Q/D: Naviguer",
            (15, disp_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.putText(
            disp_img, "n / p: Changer de selection | C: Supprimer | R: Reset | ESC: Quitter",
            (15, disp_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    cv2.imshow("SAM 3 - Annotation", disp_img)
    callback_params['base_img'] = disp_img


def create_inverted_offset_mask(mask_path, offset_pixels=20):
    mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_cv is None:
        raise ValueError(f"Impossible de lire le masque : {mask_path}")

    if offset_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset_pixels * 2 + 1, offset_pixels * 2 + 1))
        mask_cv = cv2.dilate(mask_cv, kernel, iterations=1)

    return Image.fromarray(mask_cv)


def run_diffusion_video_inpainting(input_dir, mask_dir, output_dir, prompt, neg_prompt, offset=20, device=DEVICE):
    from diffusers import StableDiffusionInpaintPipeline
    

    print("\nChargement du modèle de diffusion par Diffusers")
    model_id = "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    print(f"\nSuppressions des objets et remplissage vidéo (Offset: {offset}px)")

    generator = torch.Generator(device=device).manual_seed(42)
    prev_frame_gray = None
    prev_inpainted_np = None

    for img_name in tqdm(image_files):
        img_path = os.path.join(input_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        init_img_pil = Image.open(img_path).convert("RGB")
        
        mask_img_pil = create_inverted_offset_mask(mask_path, offset_pixels=offset)

        w, h = init_img_pil.size
        curr_frame_np = np.array(init_img_pil)
        curr_frame_gray = cv2.cvtColor(curr_frame_np, cv2.COLOR_RGB2GRAY)

        max_dim = 1024
        scale = min(1.0, max_dim / max(w, h))
        w_sd = int((w * scale) // 8) * 8
        h_sd = int((h * scale) // 8) * 8

        init_sd = init_img_pil.resize((w_sd, h_sd), Image.LANCZOS)
        mask_sd = mask_img_pil.resize((w_sd, h_sd), Image.NEAREST)

        with torch.inference_mode():
            output_sd = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=init_sd,
                mask_image=mask_sd,
                height=h_sd,
                width=w_sd,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=generator
            ).images[0]

        res_high = output_sd.resize((w, h), Image.LANCZOS)
        res_high_np = np.array(res_high)

        Image.fromarray(res_high_np).save(os.path.join(output_dir, img_name))

    print("\nSuppression et reconstitution vidéo terminées !")


def process_images(source_dir, text_prompt = None, visual_param = None, result_dir = None, max_height = None, max_width = None, invert_mask = False, diffusion_prompt = None, neg_prompt = None):
    input_dir = Path(source_dir) / "input"
    mask_dir = Path(source_dir) / "masks"
    
    os.makedirs(mask_dir, exist_ok=True)
    clean_dir(mask_dir)

    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        clean_dir(result_dir)

    images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    if not images:
        raise ValueError(f"Erreur : Aucune image trouvée dans {input_dir}")

    first_img_path = os.path.join(input_dir, images[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        raise ValueError(f"Erreur lors du chargement de {first_img_path}")

    height, width, _ = first_img.shape

    if height > max_height:
        scale_calc = max_height / height
        work_height = max_height
        work_width = int(width * scale_calc)
    elif width > max_width:
        scale_calc = max_width / width
        work_width = max_width
        work_height = int(height * scale_calc)
    else:
        scale_calc = 1.0
        work_height = height
        work_width = width

    temp_input_dir = Path(source_dir) / "temp_input_resized"
    temp_input_dir.mkdir(parents=True, exist_ok=True)
    clean_dir(temp_input_dir)

    print(f"Préparation des images de calcul SAM 3 (rescaled to {work_width}x{work_height})")
    for img_name in tqdm(images):
        img = cv2.imread(os.path.join(input_dir, img_name))
        if scale_calc != 1.0:
            img_resized = cv2.resize(img, (work_width, work_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(temp_input_dir, img_name), img_resized)
        else:
            cv2.imwrite(os.path.join(temp_input_dir, img_name), img)


    print("Chargement de SAM 3")
    predictor = build_sam3_video_predictor(gpus_to_use=[0])
    if hasattr(predictor, "model"):
        predictor.model.hotstart_delay = 0
        print("Hotstart delay configuré à 0")

    print("Démarrage de la session vidéo")
    response = predictor.handle_request(request={"type": "start_session", "resource_path": str(temp_input_dir)})
    session_id = response["session_id"]
    print(f"Session créée avec ID : {session_id}")

    with torch.inference_mode():
        predictor.handle_request(request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text_prompt if text_prompt else "visual",
        })
    
    tmp_outputs_per_frame = {}
    if visual_param != "interactive":
        print("\nPropagation automatique SAM 3...")
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
        with torch.inference_mode():
            with torch.autocast(device_type=device_type, dtype=dtype):
                for resp in predictor.handle_stream_request(request={"type": "propagate_in_video", "session_id": session_id}):
                    f_idx = resp["frame_index"] 
                    tmp_outputs_per_frame[f_idx] = to_cpu(resp["outputs"])
    else:        
        active_id = 1
        max_disp_dim = 1080
        scale = min(1.0, max_disp_dim / max(width, height))
        disp_w, disp_h = int(width * scale), int(height * scale)

        cv2.namedWindow("SAM 3 - Annotation", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

        frame_idx = 0
        is_propagating = [False]

        interaction_finished = False

        state = {}
        current_propagation_frame = [0]

        while not interaction_finished:
            img_name = images[frame_idx]
            current_img_path = os.path.join(temp_input_dir, img_name)
            current_img = cv2.imread(current_img_path)
            if current_img is None:
                continue

            callback_params = {
                'base_img': cv2.resize(current_img, (disp_w, disp_h)),
                'scale': scale,
                'state': state,
                'active_id': active_id,
                'width': width,
                'height': height,
                'predictor': predictor,
                'session_id': session_id,
                'frame_idx': frame_idx,
                'is_propagating': is_propagating,
                'current_propagation_frame': current_propagation_frame,
                'total_frames': len(images)
            }

            def update_display():
                current_state = state.get(frame_idx)
                if current_state is None:
                    current_state = make_empty_frame_state(active_id)
                    state[frame_idx] = current_state
                if active_id not in current_state["object"]:
                    current_state["object"][active_id] = {'points': [], 'labels': [], 'boxes': []}
                update_display_fn(current_img, current_state, active_id, disp_w, disp_h, callback_params, frame_idx, is_propagating=is_propagating[0])

            callback_params['update_display'] = update_display
            cv2.setMouseCallback("SAM 3 - Annotation", draw_callback, callback_params)
            update_display()

            frame_navigation = False
            while not frame_navigation and not interaction_finished:
                key = cv2.waitKey(30) & 0xFF
                if is_propagating[0]:
                    update_display()

                if not is_propagating[0] and key == ENTER_KEY:
                    has_prompts = False
                    for frame_data in state.values():
                        for id in frame_data["object"]:
                            if len(frame_data['object'][id]['boxes']) > 0 or len(frame_data['object'][id]['points']) > 0:
                                has_prompts = True
                                break
                    
                    if has_prompts:
                        print(f"\nPropagation des selections")
                        is_propagating[0] = True
                        update_display()

                        def _run_propagation():
                            try:
                                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                                dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
                                
                                current_propagation_frame[0] = 0
                                with torch.inference_mode():
                                    with torch.autocast(device_type=device_type, dtype=dtype):
                                        for resp in predictor.handle_stream_request(request={"type": "propagate_in_video", "session_id": session_id}):
                                            f_idx = resp["frame_index"]
                                            tmp_outputs_per_frame[f_idx] = to_cpu(resp["outputs"])
                                            current_propagation_frame[0] = f_idx

                                            out_masks_np = to_cpu(resp["outputs"]["out_binary_masks"])
                                            out_ids = to_cpu(resp["outputs"]["out_obj_ids"])

                                            default = make_empty_frame_state(active_id)
                                            default['current_masks'] = {}
                                            for i, o_id in enumerate(out_ids):
                                                default['current_masks'][int(o_id)] = out_masks_np[i]
                                            
                                            state[f_idx] = default
                                            
                            except Exception as e:
                                print(f"\nÉchec de la propagation : {e}")
                            finally:
                                is_propagating[0] = False
                                print("Propagation terminée !")

                        t = threading.Thread(target=_run_propagation, daemon=True)
                        t.start()
                            
                    else:
                        print("Veuillez d'abord ajouter au moins un prompt (boîte ou point) pour valider.")

                elif key == ord('d'):
                    frame_idx = (frame_idx + 1) % len(images)
                    frame_navigation = True
                
                elif key == ord('q'):
                    frame_idx = (frame_idx - 1 + len(images)) % len(images)
                    frame_navigation = True

                elif not is_propagating[0] and key == ord('p'):
                    active_id = active_id - 1 if active_id > 1 else 1
                    if not active_id in state[frame_idx]["object"]:
                        state[frame_idx]["object"][active_id] = {'points': [], 'labels': [], 'boxes': []}
                    callback_params["state"] = state
                    callback_params["active_id"] = active_id
                    update_display()

                elif not is_propagating[0] and key == ord('n'):
                    active_id = active_id + 1
                    if not active_id in state[frame_idx]["object"]:
                        state[frame_idx]["object"][active_id] = {'points': [], 'labels': [], 'boxes': []}
                    callback_params["state"] = state
                    callback_params["active_id"] = active_id
                    update_display()

                elif not is_propagating[0] and key == ESCAPE_KEY:
                    interaction_finished = True
                
                elif not is_propagating[0] and key == ord('c'):
                    if frame_idx in state.keys():
                        del state[frame_idx]
                    
                    inf_state = get_inference_state(predictor, session_id)
                    if inf_state is not None:
                        with torch.inference_mode():
                            predictor.clear_all_prompts_in_frame(inf_state, frame_idx, active_id)
                        print(f"Tous les prompts de la frame {frame_idx} ont été supprimés.")
                    callback_params["state"] = state
                    update_display()

                elif not is_propagating[0] and key == ord('r'):
                    state.clear()
                    active_id = 1
                    
                    inf_state = get_inference_state(predictor, session_id)
                    if inf_state is not None:
                        with torch.inference_mode():
                            predictor.reset_state(inf_state)
                        print("Session et objets réinitialisés")
                    callback_params["state"] = state
                    update_display()

        cv2.destroyAllWindows()
        if len(tmp_outputs_per_frame) == 0:
            predictor.shutdown()
            return

    outputs_per_frame = tmp_outputs_per_frame.copy()

    gc.collect()
    torch.cuda.empty_cache()

    print("\nExportation des masques")
    for frame_idx, img_name in tqdm(enumerate(images), total=len(images)):
        frame_out = outputs_per_frame.get(frame_idx, {})
        out_masks = frame_out.get("out_binary_masks")
        out_ids = frame_out.get("out_obj_ids")

        combined_mask_low = np.zeros((work_height, work_width), dtype=np.uint8)
        if out_masks is not None and len(out_masks) > 0 and out_ids is not None:
            if hasattr(out_masks, "cpu"):
                out_masks = out_masks.cpu().numpy()

            for i, o_id in enumerate(out_ids):
                mask_np = out_masks[i]
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                    
                combined_mask_low[mask_np > 0] = 255
        if scale_calc != 1.0:
            combined_mask = cv2.resize(
                combined_mask_low, 
                (width, height), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            combined_mask = combined_mask_low

        kernel_size = 20
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(os.path.join(mask_dir, f"{img_name}"), closed_mask)
        if result_dir:
            img = cv2.imread(os.path.join(input_dir, img_name))
            if img is not None:

                overlay = img.copy()
                red_overlay = np.zeros_like(img)
                red_overlay[closed_mask > 0] = (0, 0, 255)
                mask_indices = closed_mask > 0
                overlay[mask_indices] = cv2.addWeighted(
                    img, 0.7, red_overlay, 0.3, 0
                )[mask_indices]
                
                cv2.imwrite(os.path.join(result_dir, f"{img_name}"), overlay)

    print("\nTraitement terminé avec succès")
    predictor.shutdown()

    print("\nNettoyage des fichiers temporaires")
    clean_dir(temp_input_dir)
    os.rmdir(temp_input_dir)
    
    gc.collect()
    torch.cuda.empty_cache()

    if invert_mask:
        inverted_mask_dir = Path(source_dir) / "inverted_masks"
        os.makedirs(inverted_mask_dir, exist_ok=True)
        clean_dir(inverted_mask_dir)
        run_diffusion_video_inpainting(
            input_dir=input_dir,
            mask_dir=mask_dir,
            output_dir = inverted_mask_dir,
            prompt=diffusion_prompt,
            neg_prompt=neg_prompt,
            device=DEVICE
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 3 Video Suite - Version Optimisée")
    parser.add_argument("-s", "--source", required=True, help="Dossier source contenant le sous-dossier 'input'")
    parser.add_argument("-sp", "--sprompt", help="Prompt textuel SAM 3")
    parser.add_argument("-dp", "--dprompt", help="Prompt textuel pour la diffusion (si --inv est utilisé)")
    parser.add_argument("-np", "--nprompt", help="Prompt textuel pour la diffusion (si --inv est utilisé)")
    parser.add_argument("-v", "--visual", nargs="?", const="interactive", help="Prompt visuel")
    parser.add_argument("--result", help="Dossier optionnel de sortie des images de rendu")
    parser.add_argument("--max-height", type=int, default=1080, help="Hauteur maximale de l'image (par défaut: 1080)")
    parser.add_argument("--max-width", type=int, default=1920, help="Largeur maximale de l'image (par défaut: 1920)")
    parser.add_argument("--inv", action="store_true", help="Inverse le masque")

    args = parser.parse_args()

    if not args.sprompt and not args.visual:
        parser.error("Veuillez spécifier au moins un type de prompt (-p ou -v)")

    process_images(
        source_dir=args.source,
        text_prompt=args.sprompt,
        visual_param=args.visual,
        result_dir=args.result,
        max_height=args.max_height,
        max_width=args.max_width,
        invert_mask=args.inv,
        diffusion_prompt=args.dprompt
    )