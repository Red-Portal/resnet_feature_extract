
import numpy as np
import cv2
import mxnet as mx
import argparse

def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs

def main():
    ctx = mx.gpu(args.gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.modeel_path, args.epoch)

    train = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "cifar10_train.rec") if args.data_type == 'cifar10' else
                              os.path.join(args.data_dir, "train_256_q90.rec") if args.aug_level == 1
                              else os.path.join(args.data_dir, "train_480_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 32, 32) if args.data_type=="cifar10" else (3, 224, 224),
        batch_size          = args.batch_size,
        pad                 = 4 if args.data_type == "cifar10" else 0,
        fill_value          = 127,  # only used when pad is valid
        rand_crop           = True,
        max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
        min_random_scale    = 1.0 if args.data_type == "cifar10" else 1.0 if args.aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio    = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 0.25,
        random_h            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 36,  # 0.4*90
        random_s            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        random_l            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle    = 0 if args.aug_level <= 2 else 10,
        max_shear_ratio     = 0 if args.aug_level <= 2 else 0.1,
        rand_mirror         = True,
        shuffle             = True,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
    val = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "cifar10_val.rec") if args.data_type == 'cifar10' else
                              os.path.join(args.data_dir, "val_256_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = (3, 32, 32) if args.data_type=="cifar10" else (3, 224, 224),
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
    model = mx.model.FeedForward(
        ctx                 = devs,
        symbol              = sym,
        arg_params          = arg_params,
        aux_params          = aux_params,
        )

    print("-- Extracting feature maps")
    Y_train = model.predict(X = train)
    Y_val = model.predict(X = val)
    print("-- Extracting feature maps - Done")
    print(" shape of feature maps:", Y_train.shape, " ", Y_val.shape)

    np.save("train_fmap.npy", Y_train)
    np.save("val_fmap.npy", Y_val)

    labels = []
    for _, label, _ in train:
        labels.append(label)
    labels = np.concat(labels, 0)
    np.save("train_labels.npy", labels)

    labels = []
    for _, label, _ in train:
        labels.append(label)
    labels = np.concat(labels, 0)
    np.save("val_labels.npy", labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training resnet-v2")
    parser.add_argument('--data-dir', type=str, default='./data/imagenet/', help='the input data directory')
    parser.add_argument('--data-type', type=str, default='imagenet', help='the dataset type')
    parser.add_argument('--batch-size', type=int, default=256, help='the batch size')
    parser.add_argument('--num-examples', type=int, default=1281167, help='the number of training examples')
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument('--model-path', type=str, default='.', help='path to the saved model')
    parser.add_argument('--model-load-epoch', type=int, default=160,
                        help='load the model on an epoch using the model-load-prefix')
    args = parser.parse_args()
    main()
