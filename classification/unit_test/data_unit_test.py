import unittest
import torch
import torchvision
import torchvision.transforms as transforms

class DataTestCase(unittest.TestCase):
 
    @classmethod
    def setUpClass(cls) -> None:

        batch_size = 1
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        cls.samples = next(iter(train_ds_loader))

    def test_image_tensor_dimensions(self):

        image_tensor_shape = DataTestCase.samples[0].shape
        self.assertEqual(image_tensor_shape[0], 1)
        self.assertEqual(image_tensor_shape[1], 3)
        self.assertEqual(image_tensor_shape[2], 32)
        self.assertEqual(image_tensor_shape[3], 32)


if __name__ == '__main__':
    unittest.main()