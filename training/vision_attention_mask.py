import torch
from PIL import Image
import imageio.v3 as iio
import open_clip
from modeling_xgenmm import XGenMMVisionEncoder,  XGenMMVisionTokenizer
from config_xgenmm import XGenMMVisionEncoderConfig, XGenMMVisionTokenizerConfig

def predict(image_path: str):
    encoder_config = XGenMMVisionEncoderConfig(model_name="ViT-B-32", force_image_size=224)
    encoder_model = XGenMMVisionEncoder(encoder_config).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_model = encoder_model.to(device)

    _, _, preprocess = open_clip.create_model_and_transforms(
        model_name=encoder_config.model_name,
        pretrained="laion2b_e16",
        force_image_size=encoder_config.force_image_size,
    )
    
    img_np = iio.imread(image_path)
    img_pil = Image.fromarray(img_np)
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = encoder_model(img_tensor)  # (B, 1, T, D)
        print(image_features)

        # Assume shape is (B, 1, T, D) → create attention mask of shape (B, T)
        B, _, T, _ = image_features.shape
        vision_attn_masks = torch.ones((B, T), dtype=torch.bool).to(device)

        # 🔍 Print the vision attention mask
        print("Vision Attention Mask Shape:", vision_attn_masks.shape)
        print("Vision Attention Mask:", vision_attn_masks)

if __name__ == "__main__":
    predict("/home/adminpc/xgen-mm-phi3-mini-instruct-interleave-r-v1.5-vn/test/test-sample/image-2.jpeg")
