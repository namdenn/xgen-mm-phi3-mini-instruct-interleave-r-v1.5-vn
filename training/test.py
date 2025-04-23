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
        image_features = encoder_model(img_tensor)
        print(image_features)
        print(image_features.shape)
        image_features = image_features.unsqueeze(1).unsqueeze(2).unsqueeze(2)  
        print(image_features.shape)


    tokenizer_config = XGenMMVisionTokenizerConfig(
        vis_feature_dim=image_features.shape[-1],
        lang_embedding_dim=3072,
        num_vis_tokens=128
    )
    tokenizer_model =  XGenMMVisionTokenizer(tokenizer_config).eval().to(device)

    with torch.no_grad():
        image_tokens = tokenizer_model(image_features, vision_attn_masks=None)

        print("Visual Tokens (first 5):", image_tokens[0, :5])
        # print("Vision Attention Mask shape:", vision_attn_mask.shape)
        # print("Vision Attention Mask sample:", vision_attn_mask[0, :10])

if __name__ == "__main__":
    predict("/home/adminpc/xgen-mm-phi3-mini-instruct-interleave-r-v1.5-vn/training/test-sample/image-2.jpeg")
