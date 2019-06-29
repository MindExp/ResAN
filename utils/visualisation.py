import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def conputeTSNE(self, step, source_images, target_images, source_labels, target_labels, sess):
    target_images = target_images[:2000]
    target_labels = target_labels[:2000]
    source_images = source_images[:2000]
    source_labels = source_labels[:2000]

    target_labels = one_hot(target_labels.astype(int), 10)
    print(source_labels.shape)

    assert len(target_labels) == len(source_labels)

    n_slices = int(2000 / 128)

    fx_src = np.empty((0, 64))
    fx_trg = np.empty((0, 64))

    for src_im, trg_im in zip(np.array_split(source_images, n_slices),
                              np.array_split(target_images, n_slices),
                              ):
        feed_dict = {self.source_image: src_im, self.target_image: trg_im, self.Training_flag: False}

        fx_src_, fx_trg_ = sess.run([self.source_model.fc4, self.target_model.fc4], feed_dict)

        fx_src = np.vstack((fx_src, np.squeeze(fx_src_)))
        fx_trg = np.vstack((fx_trg, np.squeeze(fx_trg_)))

    src_labels = np.argmax(source_labels, 1)
    trg_labels = np.argmax(target_labels, 1)

    assert len(src_labels) == len(fx_src)
    assert len(trg_labels) == len(fx_trg)

    print('Computing T-SNE.')

    model = TSNE(n_components=2, random_state=0)
    print(plt.style.available)
    plt.style.use('ggplot')

    TSNE_hA = model.fit_transform(np.vstack((fx_src, fx_trg)))
    plt.figure(1, facecolor="white")
    plt.cla()
    plt.scatter(TSNE_hA[:, 0], TSNE_hA[:, 1], c=np.hstack((src_labels, trg_labels,)), s=10, cmap=mpl.cm.jet)
    plt.savefig('img01/%d.png' % step)
