from subprocess import call

from model_configs import model_configs

for config in model_configs:
    #  copy things to assets
    call(["./run_with_simpleperf.sh", config['model'], config['logit'], config['input_size']])
