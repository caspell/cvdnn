import numpy as np
import lmdb
import caffe
import gzip

IMAGE_SIZE = 28
NUM_CHANNELS = 1

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.uint8)
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def create_lmdb(datas , labels , lmdb_path) :

    env = lmdb.open(lmdb_path, map_size=int(1e12))

    txn = env.begin(write=True)

    datum = caffe.proto.caffe_pb2.Datum()

    datum.height = datas.shape[1]
    datum.width = datas.shape[2]
    datum.channels = datas.shape[3]

    for i in range(len(datas)) :
        label_num = labels[i]
        image_src = datas[i]

        datum.data = image_src.tobytes()
        datum.label = label_num

        str_id = '{:08}'.format(i)

        txn.put(str_id, datum.SerializeToString())

        if (i + 1) % 1000 == 0 :
            txn.commit()
            txn = env.begin(write=True)

    if (i + 1) % 1000 != 0 :
        txn.commit()

    env.close()

if __name__ == '__main__' :
    mode = 'test'

    if mode == 'train' :
        IMAGE_PATH = '/home/mhkim/data/mnist/train-images-idx3-ubyte.gz'
        LABEL_PATH = '/home/mhkim/data/mnist/train-labels-idx1-ubyte.gz'
        LMDB_PATH = '/home/mhkim/data/fonts_data/fonts_28_train_lmdb'
        size = 60000
    else :
        IMAGE_PATH = '/home/mhkim/data/mnist/t10k-images-idx3-ubyte.gz'
        LABEL_PATH = '/home/mhkim/data/mnist/t10k-labels-idx1-ubyte.gz'
        LMDB_PATH = '/home/mhkim/data/fonts_data/fonts_28_test_lmdb'
        size = 10000

    X = extract_data(IMAGE_PATH, size)
    labels = extract_labels(LABEL_PATH, size)

    create_lmdb(X , labels, LMDB_PATH)

    # arr = np.array((28, 28), dtype=np.uint8)
    # print ( arr )

