from torchvision.transforms import transforms
from model import *
from PIL import Image
class DetectBold:
    def __init__(self):
        self.model = Model(input_shape=1,
                      hidden_units=10,
                      output_shape=2).to(device)
        self.model.load_state_dict(torch.load('./weights/model_weights_4.pth'))
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
    def predict(self, image_path):
        image = Image.open(image_path)
        data_transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        image = data_transform(image)
        #print(image.shape)
        image = image.unsqueeze(dim=0)
        with torch.inference_mode():
            pred = self.model(image.to(self.device))
        print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
        if pred[0][0] > pred[0][1] and pred[0][0] > 0.7:
            print('chữ thường')
        elif pred[0][1] > pred[0][0] and pred[0][1] > 0.7:
            print('chữ đậm')
        else:
            print('không thể phát hiện kiểu chữ')


if __name__ == "__main__":
    model = DetectBold()
    model.predict('/home/dungdinh/Documents/Prj2/data/data_train/test/0/0d206fb281b74cfd8640499d9f4e98d0_0_.jpg')
    
    