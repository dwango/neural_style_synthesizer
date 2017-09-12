from __future__ import print_function
import chainer
import chainer.functions
import chainer.links.caffe
class VGG(object):
    def __init__(self, caffemodelpath, alpha=[0,0,1,1], beta=[0.000244140625, 6.103515625e-05, 1.52587890625e-05, 3.814697265625e-06], no_padding=False): ### beta is decided by experiments
        self.model = chainer.links.caffe.CaffeFunction(caffemodelpath)
        self.alpha = alpha
        self.beta = beta
        if no_padding:
            for layer in self.model.children():
                if not layer.name.find("conv") == -1:
                    layer.pad = 0

    def forward_layers(self, x, average_pooling=False):
        if average_pooling:
            pooling = lambda x: chainer.functions.average_pooling_2d(chainer.functions.relu(x), 2, stride=2)
        else:
            pooling = lambda x: chainer.functions.max_pooling_2d(chainer.functions.relu(x), 2, stride=2)

        y1 = self.model.conv1_2(chainer.functions.relu(self.model.conv1_1(x)))
        # x1 = chainer.functions.average_pooling_2d(chainer.functions.relu(y1), 2, stride=2)
        # x1 = chainer.functions.max_pooling_2d(chainer.functions.relu(y1), 2, stride=2)
        x1 = pooling(y1)

        y2 = self.model.conv2_2(chainer.functions.relu(self.model.conv2_1(x1)))
        # x2 = chainer.functions.average_pooling_2d(chainer.functions.relu(y2), 2, stride=2)
        # x2 = chainer.functions.max_pooling_2d(chainer.functions.relu(y2), 2, stride=2)
        x2 = pooling(y2)

        y3 = self.model.conv3_3(chainer.functions.relu(self.model.conv3_2(chainer.functions.relu(self.model.conv3_1(x2)))))
        # x3 = chainer.functions.average_pooling_2d(chainer.functions.relu(y3), 2, stride=2)
        # x3 = chainer.functions.max_pooling_2d(chainer.functions.relu(y3), 2, stride=2)
        x3 = pooling(y3)

        y4 = self.model.conv4_3(chainer.functions.relu(self.model.conv4_2(chainer.functions.relu(self.model.conv4_1(x3)))))
        #x4 = chainer.functions.average_pooling_2d(chainer.functions.relu(y4), 2, stride=2)
        #y5 = self.model.conv5_3(chainer.functions.relu(self.model.conv5_2(chainer.functions.relu(self.model.conv5_1(x4)))))
        return [y1,y2,y3,y4]

class NIN:
    def __init__(self, caffemodelpath, alpha=[0,0,1,1], beta=[0.000244140625, 6.103515625e-05, 1.52587890625e-05, 3.814697265625e-06]):
        self.model = chainer.functions.caffe.CaffeFunction(caffemodelpath)
        self.alpha = alpha
        self.beta = beta

    def forward_layers(self, x, average_pooling=False):
        if average_pooling:
            pooling = lambda x: chainer.functions.average_pooling_2d(chainer.functions.relu(x), 3, stride=2)
        else:
            pooling = lambda x: chainer.functions.max_pooling_2d(chainer.functions.relu(x), 3, stride=2)
        y0 = chainer.functions.relu(self.model.conv1(x))
        y1 = self.model.cccp2(chainer.functions.relu(self.model.cccp1(y0)))
        x1 = chainer.functions.relu(self.model.conv2(pooling(chainer.functions.relu(y1))))
        y2 = self.model.cccp4(chainer.functions.relu(self.model.cccp3(x1)))
        x2 = chainer.functions.relu(self.model.conv3(pooling(chainer.functions.relu(y2))))
        y3 = self.model.cccp6(chainer.functions.relu(self.model.cccp5(x2)))
        x3 = chainer.functions.relu(getattr(self.model,"conv4-1024")(chainer.functions.dropout(pooling(chainer.functions.relu(y3)), train=False)))
        return [y0,x1,x2,x3]
