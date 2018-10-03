import numpy as np
import tensorflow as tf

import semisup


batch_size = 100
emb_size = 128
num_classes = 10
noise_factor = .1

class LossesTest(tf.test.TestCase):
    def mockModel(self):
        def dummy(*args, **kwargs):
            return np.zeros((100, 128))

        return semisup.SemisupModel(dummy, 10, [1])

    def setupTransformationLossTest(self):
        embeddings = []
        logits = []

        class_embs = np.identity(emb_size)
        np.random.shuffle(class_embs)
        logit_embs = np.identity(num_classes)

        for b in range(batch_size):
            c = np.random.randint(0, num_classes)
            emb = class_embs[c] + np.random.uniform(-noise_factor, noise_factor, emb_size)
            embeddings.append(emb)
            logit = logit_embs[c] + np.random.uniform(-noise_factor, noise_factor, num_classes)
            logits.append(logit)

        embeddings = np.array(embeddings, dtype=np.float32)
        logits = np.array(logits, dtype=np.float32)

        t_embs = tf.Variable(embeddings, dtype=np.float32)
        t_logits = tf.Variable(logits, dtype=np.float32)

        t_embs_bad = tf.random_normal((batch_size, emb_size), dtype=np.float32)
        t_logits_bad = tf.random_normal((batch_size, num_classes), dtype=np.float32)

#        print(t_embs_bad.shape, t_logits_bad.shape, batch_size)
        model = self.mockModel()

        t_loss_good = model.add_transformation_loss_sparse(t_embs, t_embs,
                                                    t_logits,
                                                    t_logits, batch_size, batch_size)
        t_loss_bad_1 = model.add_transformation_loss_sparse(t_embs, t_embs_bad,
                                                     t_logits,
                                                     t_logits, batch_size, batch_size)
        t_loss_bad_2 = model.add_transformation_loss_sparse(t_embs, t_embs_bad,
                                                     t_logits,
                                                     t_logits_bad, batch_size, batch_size)
        return [t_loss_good, t_loss_bad_1, t_loss_bad_2, t_embs]

    def testTransformationLoss(self):
        t_loss_good, t_loss_bad_1, t_loss_bad_2, t_embs = self.setupTransformationLossTest()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            [np_loss_good, np_loss_bad_1, np_loss_bad_2, np_embs] = sess.run(
                    [t_loss_good, t_loss_bad_1, t_loss_bad_2, t_embs])
            print('good loss', np_loss_good)
            print('bad loss 1', np_loss_bad_1)
            print('bad loss 2', np_loss_bad_2)
            # print('embs', np_embs)
            self.assertGreater(np_loss_bad_1, np_loss_good)
            self.assertGreater(np_loss_bad_2, np_loss_good)


    def testLogitEntropy(self):
        model = self.mockModel()
        input = tf.placeholder(np.float32, shape=(10, 10), name='test_in')
        loss = model.add_logit_entropy(input)
        with self.test_session() as sess:

            clean_res = sess.run(loss, {input: np.identity(10, np.float64)})
            normal_res = sess.run(loss, {input: np.random.normal(size=[10,10])})
            unif_res = sess.run(loss, {input: np.random.uniform(0, 1, size=[10,10])})

            self.assertAlmostEqual(clean_res, 0, places=4)
            self.assertGreater(normal_res, 0.4)
            self.assertGreater(unif_res, 0.4)

    def testClusterHardening(self):
        model = self.mockModel()
        input = tf.placeholder(np.float32, shape=(10, 10), name='test_in')
        loss = model.add_cluster_hardening_loss(input)
        with self.test_session() as sess:

            # logits are softmaxed afterwards, so don't have to be in [-1,1]
            clean_res = sess.run(loss, {input: np.ones((10,10))*-10+np.identity(10, np.float64)*10})
            normal_res = sess.run(loss, {input: np.random.normal(size=[10, 10])})
            unif_res = sess.run(loss, {input: np.random.uniform(-1, 1, size=[10,10])})

            self.assertAlmostEqual(clean_res, 0, places=2)
            self.assertGreater(normal_res, 0.1)
            self.assertGreater(unif_res, 0.1)

    def testSATLoss(self):
        model = self.mockModel()
        unsup_input = tf.placeholder(np.float32, shape=(10, 10), name='test_in')
        unsup_aug_input = tf.placeholder(np.float32, shape=(10, 10), name='test_in')
        loss = model.add_sat_loss(unsup_input, unsup_aug_input)
        with self.test_session() as sess:

            # logits are softmaxed afterwards, so don't have to be in [-1,1]
            clean_res = sess.run(loss, {
                unsup_input: np.ones((10,10))*-10+np.identity(10, np.float64)*10,
                unsup_aug_input: np.ones((10,10))*-10+np.identity(10, np.float64)*10,
            })
            normal_res = sess.run(loss, {
                unsup_input: np.ones((10,10))*-10+np.identity(10, np.float64)*10,
                unsup_aug_input: np.random.normal(size=[10, 10]),
            })
            unif_res = sess.run(loss, {
                unsup_input: np.ones((10,10))*-10+np.identity(10, np.float64)*10,
                unsup_aug_input: np.random.uniform(-1, 1, size=[10,10]),
            })

            print(clean_res, normal_res, unif_res)

            self.assertAlmostEqual(clean_res, 0, places=2)
            self.assertGreater(normal_res, 0.1)
            self.assertGreater(unif_res, 0.1)

if __name__ == '__main__':
    tf.test.main()
