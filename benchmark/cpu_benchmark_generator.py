from subprocess import call

from model_configs import model_configs

for config in model_configs:
    # python benchmark.py --g="models/pruned_output.pb" --i="input:0" --o="pred:0" --n=10000 --ish=100,1
    call(["python", "benchmark.py",
          "--g=models/%s" % config['model'],
          "--i=Placeholder_1:0", "--o=%s" % config['logit'],
          '--ish=2,%s,%s,3' % (config['input_size'], config['input_size']),
          "--n=100"])
