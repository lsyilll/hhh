"""Local-only inference shared by the HTTP service and the executable smoke check."""
import gc
import os
from pathlib import Path
import numpy as np
from model_service.providers import ENCODERS, MODEL_REPOS, box_pixels, normalized_box, weights_ready

class ResearchRuntime:
    def __init__(self, root, device='cpu'):
        self.root = Path(root)
        self.device = device
        self.active = {}

    def load(self, name):
        if name not in MODEL_REPOS:
            raise ValueError('Unknown model')
        if self.active.get('name') == name:
            return self.active['model'], self.active['processor']
        if not weights_ready(self.root, name):
            raise ValueError('Model files missing or incomplete. Run scripts/download_models.py ' + name)
        if self.device.startswith('cuda') and os.getenv('GUANYE_EXCLUSIVE_GPU') != '1':
            raise ValueError('Stop AgriChat, then explicitly enable exclusive GPU mode')
        self.clear()
        import torch
        folder = self.root / name
        if name in ENCODERS:
            from transformers import AutoImageProcessor, AutoModel
            processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True)
            model = AutoModel.from_pretrained(folder, local_files_only=True, use_safetensors=True).eval().to(self.device)
        elif name == 'sam2':
            from transformers import Sam2Model, Sam2Processor
            processor = Sam2Processor.from_pretrained(folder, local_files_only=True)
            model = Sam2Model.from_pretrained(folder, local_files_only=True, use_safetensors=True).eval().to(self.device)
        elif name == 'sam3':
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            model = build_sam3_image_model(checkpoint_path=str(folder / 'sam3.pt'), load_from_HF=False, device=self.device)
            processor = Sam3Processor(model, device=self.device)
        else:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
            model = AutoModelForImageTextToText.from_pretrained(folder, local_files_only=True, use_safetensors=True,
                                                               torch_dtype=torch.bfloat16).eval().to(self.device)
        self.active.update(name=name, model=model, processor=processor)
        return model, processor

    def clear(self):
        self.active.clear()
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def embedding(self, images, name):
        import torch
        if name not in ENCODERS:
            raise ValueError('Unknown visual encoder')
        model, processor = self.load(name)
        rows = []
        with torch.inference_mode():
            for image in images:
                inputs = processor(images=image, return_tensors='pt').to(self.device)
                vector = model(**inputs).last_hidden_state[:, 0].float().cpu().numpy()[0]
                vector /= max(np.linalg.norm(vector), 1e-8)
                rows.append(vector)
        return np.asarray(rows, dtype=np.float32)

    def segmentation(self, image, name='sam2', box=None, prompt='leaf'):
        import torch
        if name not in ('sam2', 'sam3'):
            raise ValueError('Unknown segmentation model')
        # SAM2 does not interpret text. It must receive an explicit selected region.
        if name == 'sam2':
            pixel_box = box_pixels(box, image.width, image.height)
        elif box is not None:
            box = normalized_box(box)
        model, processor = self.load(name)
        with torch.inference_mode():
            if name == 'sam2':
                inputs = processor(images=image, input_boxes=[[pixel_box]], return_tensors='pt').to(self.device)
                output = model(**inputs, multimask_output=True)
                masks = processor.post_process_masks(output.pred_masks.cpu(), inputs['original_sizes'].cpu())[0]
                masks = masks.reshape(-1, image.height, image.width).numpy().astype(bool)
                scores = output.iou_scores.float().cpu().numpy().reshape(-1)
            else:
                state = processor.set_image(image)
                state = processor.set_text_prompt(prompt=str(prompt)[:100], state=state)
                if box is not None:
                    state = processor.add_geometric_prompt(box=box, label=True, state=state)
                masks = state['masks'].detach().cpu().numpy().reshape(-1, image.height, image.width).astype(bool)
                scores = state['scores'].float().detach().cpu().numpy().reshape(-1)
        return masks, scores
