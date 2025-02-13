from .client import FLClient
from models.cnn import CNN

class MaliciousClient(FLClient):
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for images, labels in self.train_loader:
                labels[labels == 1] = 7  # Flip labels for attack
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters({}), len(self.train_loader.dataset), {}

fl.client.start_numpy_client("127.0.0.1:8080", client=MaliciousClient(CNN()))
