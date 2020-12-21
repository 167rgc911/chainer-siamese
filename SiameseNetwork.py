import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, Chain
from contrastive import contrastive


class SiameseNetwork(Chain):

    def __init__(self):
        super(SiameseNetwork, self).__init__(
            conv1=L.Convolution2D(1, 20, ksize=5, stride=1),
            conv2=L.Convolution2D(20, 50, ksize=5, stride=1),
            fc3=L.Linear(800, 500),
            fc4=L.Linear(500, 10),
            fc5=L.Linear(10, 2),
        )

    def forward_once(self, x_data, train=True):
        chainer.config.train = train

        x = Variable(x_data)

        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv2(h), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        y = self.fc5(h)

        return y

    def forward(self, x0, x1, label, train=True):
        chainer.config.train = train

        y0 = self.forward_once(x0, train)
        y1 = self.forward_once(x1, train)
        label = Variable(label)

        return contrastive(y0, y1, label)
