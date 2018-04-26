
import os
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

def evaluate(ctx, sym, args_params, aux_params, data):
    fmaps = np.zeros([0, 64])
    labels = np.zeros([0, 10])

    executor = sym.simple_bind(ctx[0], "null", data=(args.batch_size, 3, 32, 32))
    executor.copy_params_from(args_params, aux_params)

    for i, batch in enumerate(data):
        print("-- forwarding batch ", i)
        data = batch.data[0]
        label = batch.label[0].asnumpy()

        out = executor.forward(False, data=data)

        print(out[0])
        print(fmaps)
        fmaps = np.concatenate([out[0], fmaps], 0)
        labels = np.concatenate([label, labels], 0)

    return labels, fmaps

def main():
    ctx = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    kv = mx.kvstore.create(args.kv_store)

    model_prefix = "{}/resnet-{}-{}-{}".format(args.model_path, args.data_type, args.depth, kv.rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.model_load_epoch)

    train = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "cifar10_train.rec") if args.data_type == 'cifar10' else
                              os.path.join(args.data_dir, "train_256_q90.rec") if args.aug_level == 1
                              else os.path.join(args.data_dir, "train_480_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        #label_name          = 'softmax_label',
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
        #label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = (3, 32, 32) if args.data_type=="cifar10" else (3, 224, 224),
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)

    out_layer = sym.get_internals()["flatten0_output"]

    arg_params.pop("fc1_bias", None)
    arg_params.pop("fc1_weight", None)

    print("-- Extracting feature maps")
    fmap_train, label_train = evaluate(ctx, out_layer, arg_params, aux_params, train)
    print(" shape of feature maps:", fmap_train.shape, " ", label_train.shape)
    np.save("train_fmap.npy", fmap_train)
    np.save("train_labels.npy", label_train)
    print("-- Extracting feature maps - Done")


    print("-- Extracting feature maps")
    fmap_val  , label_val   = evaluate(ctx, out_layer, arg_params, aux_params, val)
    print(" shape of feature maps:", fmap_val.shape, " ", label_val.shape)
    np.save("val_fmap.npy", fmap_val)
    np.save("val_labels.npy", label_val)
    print("-- Extracting feature maps - Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training resnet-v2")
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str, default='./data/imagenet/', help='the input data directory')
    parser.add_argument('--data-type', type=str, default='imagenet', help='the dataset type')
    parser.add_argument('--batch-size', type=int, default=256, help='the batch size')
    parser.add_argument('--num-examples', type=int, default=1281167, help='the number of training examples')
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument('--depth', type=int, default=50, help='the depth of resnet')
    parser.add_argument('--model-path', type=str, default='.', help='path to the saved model')
    parser.add_argument('--model-load-epoch', type=int, default=160,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--aug-level', type=int, default=2, choices=[1, 2, 3],
                        help='level 1: use only random crop and random mirror\n'
                             'level 2: add scale/aspect/hsv augmentation based on level 1\n'
                             'level 3: add rotation/shear augmentation based on level 2')
    args = parser.parse_args()
    main()
