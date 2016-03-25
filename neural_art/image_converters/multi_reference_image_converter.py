import chainer
import chainer.links
import chainer.cuda
import chainer.optimizers
import chainer.functions
import neural_art
import numpy
import scipy.optimize
import image_converter
import openopt

class MultiReferenceImageConverter(image_converter.BaseImageConverter):
    def __init__(self, texture_imgs, gpu=-1, optimizer=None, model=None, content_weight=1, texture_weight=1, average_pooling=False):
        super(MultiReferenceImageConverter, self).__init__(gpu=gpu, optimizer=optimizer, model=model, content_weight=content_weight, texture_weight=texture_weight, average_pooling=average_pooling)
        self.texture_features = []
        for texture_img in texture_imgs:
            texture_array = self.xp.array(neural_art.utility.img2array(texture_img))
            layers = self.model.forward_layers(chainer.Variable(texture_array), average_pooling=self.average_pooling)
            for layer in layers:
                print(layer.data.shape)
            # self.texture_features.append(self._to_texture_feature(layers))
            # self.texture_features.append(chainer.Variable(self._to_texture_feature(layers).data.get()))
            self.texture_features.append(chainer.Variable(self._to_texture_feature(layers).data))

    def _constructed_feature(self, ratio):
        constructed_feature = None
        for texture_feature_index in xrange(len(self.texture_features)):
            if constructed_feature is None:
                constructed_feature = ratio[texture_feature_index] * self.texture_features[texture_feature_index]
            else:
                constructed_feature += ratio[texture_feature_index] * self.texture_features[texture_feature_index]
        # return constructed_feature
        return chainer.Variable(self.xp.array(constructed_feature.data))

    def convert_debug(self, content_img, init_img, output_directory, max_iteration=1000, debug_span=100, optimize=True, random_init=False):
        initial_array = self.xp.array(neural_art.utility.img2array(content_img))
        initial_feature = self._to_texture_feature(self.model.forward_layers(chainer.Variable(initial_array), average_pooling=self.average_pooling))
        if optimize:
            self.texture_ratio = self.optimize_texture_feature(initial_feature)
        else:
            self.texture_ratio = numpy.ones(len(self.texture_features)) / len(self.texture_features)
        self.constructed_feature = self._constructed_feature(self.texture_ratio)
        # START=100000
        # END=400000
        # print(self.texture_ratio)
        # print(self.texture_ratio.shape)
        # print(initial_feature.data.shape)
        # print("initial feature", initial_feature.data[START:END].max())
        # print("initial feature", initial_feature.data[START:END].min())
        # print("initial feature", initial_feature.data[START:END].sum())
        # print("contructed feature", self.constructed_feature.data[START:END].max())
        # print("contructed feature", self.constructed_feature.data[START:END].min())
        # print("contructed feature", self.constructed_feature.data[START:END].sum())
        for i in xrange(0, initial_feature.data.shape[0], 10000):
            print(i, ":", self.constructed_feature.data[i:i+10000].sum()/initial_feature.data[i:i+10000].sum())
        return super(MultiReferenceImageConverter, self).convert_debug(content_img=content_img, init_img=init_img, output_directory=output_directory, max_iteration=max_iteration, debug_span=debug_span, random_init=random_init)

    def _texture_loss(self, layers):
        now_feature = self._to_texture_feature(layers)
        debug = True
        if debug:
            constructed_feature = self._constructed_feature(numpy.ones(len(self.texture_features))/len(self.texture_features))
            loss_texture = self.squared_error(
                now_feature,
                constructed_feature
            )
            print("loss_texture_before", loss_texture.data)

        loss_texture = self.squared_error(
            now_feature,
            self.constructed_feature
        )
        print("loss_texture", loss_texture.data)
        return loss_texture

    def optimize_texture_feature(self, target_feature):
        """
        minimize (target - k1s1 - k2s2 + ...) ^ 2

        ->
        -2 * target * (k1s1 + k2s2) + (k1s1 + k2s2 + ...) ^ 2

        ->
        (k1 k2)(s1^2 s1s2, s2s1, s2^2)(k1 k2) + (-2Ts1, -2Ts2)(k1, k2)
        """
        num_textures = len(self.texture_features)
        H = numpy.zeros((num_textures, num_textures))
        for x in xrange(num_textures):
            for y in xrange(num_textures):
                H[x, y] = 2*self.texture_features[x].data.dot(self.texture_features[y].data)
        print("H:", H)
        f = numpy.zeros(num_textures)
        for x in xrange(num_textures):
            f[x] = -2 * target_feature.data.dot(self.texture_features[x].data)
        lower_bound = numpy.zeros(num_textures) # non negative
        aeq, beq = numpy.ones(num_textures), 1 # w1+w2+... = 1
        p = openopt.QP(H, f, Aeq=aeq, beq=beq, lb=lower_bound)
        r = p.solve("cvxopt_qp")
        ratio = r.xf
        print "Style ratios: ", ratio
        print "sum", self.xp.sum(ratio)
        return ratio

    def optimize_texture_feature_scipy(self, target_feature):
        """
        Find appropriate weights which minimize loss between target_faeture and constructed feature
        """
        num_textures = len(self.texture_features)
        # feature_matrix = numpy.array([texture_feature.data.get() for texture_feature in self.texture_features])
        def loss_function(weights):
            # constructed_feature = weights * feature_matrix
            constructed_feature = self._constructed_feature(weights)
            return self.squared_error(target_feature, constructed_feature).data.get()

        def jacobian_function(weights):
            jacobian = []
            for texture_feature_index in xrange(num_textures):
                # v1 = weights[texture_feature_index] * self.xp.dot(self.texture_features[texture_feature_index].data, self.texture_features[texture_feature_index].data)
                X = target_feature
                for texture_feature_index_another in xrange(num_textures):
                    X += -weights[texture_feature_index_another]*self.texture_features[texture_feature_index_another]
                jacobian.append(-2*float(self.texture_features[texture_feature_index].data.dot(X.data)))
                # jacobian.append(float(v1+X.data.dot(X.data)))
            print("Ts1:", target_feature.data.dot(self.texture_features[0].data))
            for i in xrange(num_textures):
                print("s{}s1:".format(i), self.texture_features[i].data.dot(self.texture_features[0].data))
            print("w:", weights)
            print("jac:", jacobian)
            return jacobian

        def ineq_i(i):
            def ineq(x):
                print(x)
                # print(dir(x))
                # print("i:", i)
                # print(x[i])
                # val = numpy.array([float(x[i])])
                # print(val > 0)
                return numpy.array(x[i])
            return ineq

        constraints = [
                # # {'type': 'ineq',
                # # 'fun' : lambda x: float(x[i])}
                # {'type': 'ineq',
                # 'fun' : ineq_i(i),
                # # 'fun' : lambda x: float(x[i])}
                # 'jac' : lambda x: numpy.identity(num_textures)[i]}
                # for i in range(num_textures)
            ] + [
                {'type': 'eq',
                 'fun' : lambda x: numpy.array([numpy.sum(x) - 1.0]),
                 'jac' : lambda x: numpy.ones(num_textures)}
            ] + [
                {'type': 'ineq',
                'fun' : lambda x: x,
                'jac' : lambda x: numpy.identity(num_textures)}
            ]
        # constraints = [
        #         {'type': 'eq',
        #          'fun' : lambda x: numpy.array([numpy.sum(x) - 1.0]),
        #          'jac' : lambda x: numpy.ones(num_textures)}
        #     ]
        # bounds = [(-1000, 100) for _ in xrange(num_textures)]
        print constraints
        x0 = numpy.ones(num_textures)/(num_textures+1.0) #sum(x0) != 1, so that it should be optimized
        x0 = numpy.random.random(num_textures)

        optimized_result = scipy.optimize.minimize(fun=loss_function, x0=x0, constraints=constraints, jac=jacobian_function, method='SLSQP', options={'disp': True, 'maxiter': 100, 'iprint': 2})
        ratio = optimized_result.x
        ratio = self.xp.array(ratio, dtype=self.xp.float32)
        ratio /= self.xp.sum(ratio) #sum==1

        before = loss_function(numpy.ones(num_textures)/num_textures)
        after = loss_function(ratio)
        neural_art.utility.print_ltsv({
            "before": before,
            "after": after,
            "optimization": after/before
        })
        print "Style ratios: ", ratio
        print "sum", self.xp.sum(ratio)
        return ratio

