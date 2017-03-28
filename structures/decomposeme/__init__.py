"""
 This package is implementing
 DecomposeMe: Simplifying ConvNets for End-to-End Learning
 Implementing https://arxiv.org/abs/1606.05426
 
 Basically instead of running K*K convolutions, we run 
   relu(conv(1*K, relu(conv(K*1, input) + bias)) + bias)

"""