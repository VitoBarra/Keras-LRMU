
import torch

if __name__ == '__main__':

    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    torch.random.manual_seed(156)

    shape = (5000, 5000)

    # value = math.sqrt(6 / (shape[0] + shape[1]))
    # t = tf.random.uniform(shape=shape, minval=-value, maxval=value)
    # t2 = tf.random.uniform(shape=shape, minval=-value, maxval=value)
    #
    # for i in range(1000):
    #     t = tf.matmul(t, t2)


    device = torch.device("cuda")

    t = torch.rand(shape).to(device)
    t2 = torch.rand(shape).to(device)

    for i in range(1000):
        t = torch.mul(t, t2)

    print(t)