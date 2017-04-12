import numpy as np
from collections import namedtuple
import tensorflow as tf
from tensorflow.contrib import rnn

SECOND_PER_FRAME = 30

SAMPLE_DATA_INDEX = [
            [[60,  80] , [120, 137]],
            [[145, 165] , [190, 212]],
            [[218, 240] , [298, 317]],
            [[328, 342] , [410, 442]],
            [[453, 467] , [610, 582]],
            [[588, 608] , [635, 658]],
            [[663, 683] , [725, 746]],
            [[757, 771] , [801, 879]],
            [[885, 904] , [940, 960]],
            [[967, 985] , [1030, 1138]],
            [[1147, 1162] , [1195, 1228]],
            [[1230, 1250] , [1280, 1318]],
            [[1330, 1343] , [1385, 1405]],
            [[1533, 1547] , [1572, 1588]],
            [[1881, 1902] , [1951, 1976]],
            [[1987, 2001] , [2028, 2047]],
            [[2053, 2071] , [2100, 2152]],
            [[2271, 2291] , [2314, 2334]],
            [[2343, 2364] , [2386, 2410]],
            [[2420, 2435] , [2467, 2502]],
            [[2780, 2794] , [2827, 2844]]
        ]

def load_avg_data () :
    with open('/home/mhkim/data/numpy/sampyo/sampyo_6_data_average', 'r') as fd :
        arr = np.load(fd)
    return arr

def read_data ():
    """
        [0, 1, 2, 3]
        others, in , shuffling , out
    """
    datas = load_avg_data()

    Lines = namedtuple('Lines', 'input_begin input_end output_begin output_end')

    Data = namedtuple('Data', 'data labels next_batch')

    Dataset = namedtuple('Dataset', 'train test')

    _index = np.array(SAMPLE_DATA_INDEX)

    max_range = np.max([ e - s for s , e in _index[:,0,:]])

    last_index = np.max(_index) + 36

    index_count , _ , _ = np.shape(_index)

    _values = np.zeros((last_index, SECOND_PER_FRAME, 2))
    _labels = np.zeros((last_index, 4))

    others_map = np.zeros((last_index))

    value_index = 0
    for i in range(index_count) :
        (_a, _b), (_c, _d) = np.array(_index[i])
        l = Lines(_a, _b , _c, _d)

        step, local_begin = l.input_end - l.input_begin, l.input_begin
        for t in range(step):
            begin = (local_begin + t) * SECOND_PER_FRAME
            _values[value_index] = datas[begin:begin + SECOND_PER_FRAME, 0::3]
            _labels[value_index] = [0,1,0,0]
            value_index = value_index + 1

        step , local_begin = l.output_begin - l.input_end , l.input_end
        for t in range(step):
            begin = (local_begin + t) * SECOND_PER_FRAME
            _values[value_index] = datas[begin:begin+SECOND_PER_FRAME,0::3]
            _labels[value_index] = [0,0,1,0]
            value_index = value_index + 1

        step, local_begin = l.output_end - l.output_begin, l.output_begin
        for t in range(step):
            begin = (local_begin + t) * SECOND_PER_FRAME
            _values[value_index] = datas[begin:begin + SECOND_PER_FRAME, 0::3]
            _labels[value_index] = [0,0,0,1]
            value_index = value_index + 1

        others_map[l.input_begin:l.output_end] = 1

    for i in range(last_index - SECOND_PER_FRAME):
        if others_map[i] == 0 :
            begin = i * SECOND_PER_FRAME
            _values[value_index] = datas[begin:begin + SECOND_PER_FRAME, 0::3]
            # _labels[value_index] = 0
            value_index = value_index + 1

    next_batch = (lambda index : Data( _values[:index] , _labels[:index] , next_batch))

    train = Data(_values , _labels , next_batch)

    test = Data(_values[:5] , _labels[:5], next_batch)

    return Dataset(train, test)

def get_scenes (batch_size=1 , time_limit=None):

    datas = load_avg_data()

    if time_limit :
        datas = datas[:time_limit * SECOND_PER_FRAME]

    total_frame , _ = np.shape(datas)

    seconds = total_frame // SECOND_PER_FRAME

    size = seconds // batch_size

    max_frame_number = size * batch_size * SECOND_PER_FRAME

    target_frames = datas[:max_frame_number]

    f, t = np.shape(target_frames)

    target_frames = target_frames.reshape(f * t)

    result = np.reshape(target_frames, (size , batch_size, SECOND_PER_FRAME, 4))

    return result[:,:,:,0::3]

if __name__ == '__main__' :

    # result = get_scenes(64 , time_limit=60 * 60 * 1)
    #
    # for r in result :
    #     print (np.shape(r))
    #
    # print (np.shape(result))

    a = [[9,0]]

    b = np.append(a, [[7, 8]],axis=1)
    b = np.append(b, [[7, 8]],axis=1)
    b = np.append(b, [[7, 8]],axis=1)

    print ( b)

    # dataset = read_data()
    #
    # d , l , _ = dataset.train.next_batch(1000)
    #
    # print ( l )