from torchvision.models import vgg16

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.vgg = vgg16()
        self.vgg = nn.Sequential(*list(self.vgg.features.children()))

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.vgg(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
    
    
    
    
import cv2
image = cv2.imread("C:\\Users\\alina\\jupiter\\Wider_face_dataset_2\\wider_face_train\\0.jpg")
dsize = (50, 50)
image = cv2.resize(image, dsize)
image = torchvision.transforms.ToTensor()(image)
image = image.view(1, 3, 50, 50)
model = SiameseNetwork()
model(image, image)
