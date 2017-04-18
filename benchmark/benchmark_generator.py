from subprocess import call
model_configs = [
    {"model": "alexnet_v2.pb", "logit": "alexnet_v2/fc8/squeezed:0", "input_size": "224"},
    {"model": "cifarnet.pb", "logit": "CifarNet/logits/BiasAdd:0", "input_size": "32"},
    {"model": "inception_resnet_v2.pb", "logit": "InceptionResnetV2/Logits/Logits/BiasAdd:0", "input_size": "299"},
    {"model": "inception_v1.pb", "logit": "InceptionV1/Logits/SpatialSqueeze:0", "input_size": "224"},
    {"model": "inception_v2.pb", "logit": "InceptionV2/Logits/SpatialSqueeze:0", "input_size": "224"},
    {"model": "inception_v3.pb", "logit": "InceptionV3/Logits/SpatialSqueeze:0", "input_size": "299"},
    {"model": "inception_v4.pb", "logit": "InceptionV4/Logits/Logits/BiasAdd:0", "input_size": "299"},
    {"model": "lenet.pb", "logit": "LeNet/fc4/BiasAdd:0", "input_size": "28"},
    {"model": "vgg_16.pb", "logit": "vgg_16/fc8/squeezed:0", "input_size": "224"},
    {"model": "vgg_19.pb", "logit": "vgg_19/fc8/squeezed:0", "input_size": "224"},
    {"model": "resnet_v1_50.pb", "logit": "resnet_v1_50/SpatialSqueeze:0", "input_size": "224"},
    {"model": "resnet_v1_101.pb", "logit": "resnet_v1_101/SpatialSqueeze:0", "input_size": "224"},
    {"model": "resnet_v1_152.pb", "logit": "resnet_v1_152/SpatialSqueeze:0", "input_size": "224"},
    {"model": "resnet_v1_200.pb", "logit": "resnet_v1_200/SpatialSqueeze:0", "input_size": "224"},
    {"model": "resnet_v2_50.pb", "logit": "resnet_v2_50/SpatialSqueeze:0", "input_size": "224"},
    {"model": "resnet_v2_101.pb", "logit": "resnet_v2_101/SpatialSqueeze:0", "input_size": "224"},
    {"model": "resnet_v2_152.pb", "logit": "resnet_v2_152/SpatialSqueeze:0", "input_size": "224"},
    {"model": "resnet_v2_200.pb", "logit": "resnet_v2_200/SpatialSqueeze:0", "input_size": "224"},
    {"model": "overfeat.pb", "logit": "overfeat/fc8/squeezed:0", "input_size": "231"}
]

for config in model_configs:
    #  copy things to assets
    call(["./run_with_simpleperf.sh", config['model'], config['logit'], config['input_size']])