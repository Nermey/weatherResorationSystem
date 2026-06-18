from PIL import Image
import torch
import torchvision.transforms as T
import timm
from udrs2former_inference import UDRS2FormerInference
from ConvIR.convir_infer import ConvIRInference
from DehazeFormer.dehazeformer_infer import DehazeFormerInference


class WeatherRestorationSystem:
    def __init__(
            self,
            device: str = 'cpu',
            classificator_path: str = './models/E08_ConvNextV2T_best.pth',
            rain_model_path: str = './models/udrs2former_demo.pth',
            snow_model_path: str = './ConvIR/Image_desnowing/models/snow100k-base.pkl',
            haze_model_path: str = './models/dehazeformer-b-2.pth',

    ):
        self.__classes = ['fogsmog', 'rain', 'snow']
        self.device = device

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.__load_classificator(classificator_path=classificator_path)
        self.__load_rain_model(rain_model_path=rain_model_path)
        self.__load_snow_model(snow_model_path=snow_model_path)
        self.__load_haze_model(haze_model_path=haze_model_path)

    def __load_classificator(self, classificator_path: str):
        self.classificator = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in22k_in1k',
            pretrained=False,
            num_classes=len(self.__classes),
        )

        self.classificator.load_state_dict(
            torch.load(classificator_path, map_location=self.device)
        )

        self.classificator.to(self.device)
        self.classificator.eval()

    @torch.no_grad()
    def classify(self, image: Image.Image) -> tuple[str, float]:
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.classificator(x)
        probs = torch.softmax(logits, dim=1)[0]

        # best class
        class_id = probs.argmax().item()
        confidence = probs[class_id].item()
        class_name = self.__classes[class_id]

        return class_name, confidence

    def __load_rain_model(self, rain_model_path: str):
        self.rain_model = UDRS2FormerInference(
            weights_path=rain_model_path,
            arch_path='./UDR-S2Former_deraining/UDR_S2Former.py',
            img_size=(320, 320),
            tile_size=320,
            tile_overlap=0,
            device=self.device,
        )

    def __load_snow_model(self, snow_model_path: str):
        self.snow_model = ConvIRInference(
            task='desnow',
            weights_path=snow_model_path,
            convir_root='./ConvIR',
            version='base',
            device=self.device,
        )

    def __load_haze_model(self, haze_model_path: str):
        self.haze_model = DehazeFormerInference(
            variant='b',
            weights_path=haze_model_path,
            dehazeformer_root='./DehazeFormer',
            device=self.device,
        )

    def restore(self, image: Image.Image) -> Image.Image:
        weather_type, confidence = self.classify(image=image)
        print(weather_type, confidence)

        if confidence <= 0.8:
            return image

        if weather_type == 'rain':
            return self.rain_model(image)

        elif weather_type == 'snow':
            return self.snow_model(image)

        elif weather_type == 'fogsmog':
            return self.haze_model(image)

        return image

import os

#usage
w = WeatherRestorationSystem(device='mps')

img = Image.open('snow.jpg').convert('RGB')

result = w.restore(img)
result.save('result_snow.jpg')

