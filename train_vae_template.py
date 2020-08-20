import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import sys
import pickle
import scipy.io


debug = False
img_rows = 28
img_cols = 20
ff = scipy.io.loadmat('data/frey_rawface.mat')
ff = ff["ff"].T.reshape((-1, 1, img_rows, img_cols))
ff = ff.astype('float32') / 255.
print(ff.shape)

n_samples = ff.shape[0]


# Number of parameters
input_size = 560
hidden_size = 128
latent_size = 16
std = 0.02
learning_rate = 0.02
loss_function = 'bce'  # mse or bce
beta1=0.9
beta2=0.999


def get_minibatch(batch_size, idx=0, indices=None):
    start_idx = batch_size * idx
    end_idx = min(start_idx + batch_size, n_samples)

    if indices is None:
        sample_b = ff[start_idx:end_idx]
    else:
        idx = indices[start_idx:end_idx]
        sample_b = ff[idx]

    sample_b = np.resize(sample_b, (batch_size, 560))

    sample_b = np.transpose(sample_b, (1, 0))

    return sample_b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The derivative of the sigmoid function
def dsigmoid(y, x=None):
    return y * (1 - y)

# The tanh function
def tanh(x):
    return np.tanh(x)


# The derivative of the tanh function
def dtanh(y, x=None):
    return 1 - y * y


def softplus(x):
    return np.log(1 + np.exp(x))


def dsoftplus(y, x=None):
    assert x is not None
    return sigmoid(x)


def sample_unit_gaussian(latent_size):
    return np.random.standard_normal(size=(latent_size))


# (Inplace) relu function
def relu(x):
    x[x < 0] = 0

    return x


# Gradient of Relu
def drelu(y, x=None):
    return 1. * (y > 0)


# Initialization was done not exactly according to Kingma et al. 2014 (he used Gaussian).
# input to hidden weight
Wi = np.random.uniform(-std, std, size=(hidden_size, input_size))
Bi = np.random.uniform(-std, std, size=(hidden_size, 1))

# encoding weight (hidden to code mean)
Wm = np.random.uniform(-std, std, size=(latent_size, hidden_size))  # hidden to mean
Bm = np.random.uniform(-std, std, size=(latent_size, 1))  # hidden to mean\

Wv = np.random.uniform(-std, std, size=(latent_size, hidden_size))  # hidden to logvar
Bv = np.random.uniform(-std, std, size=(latent_size, 1))  # hidden to logvar

# weight mapping code to hidden
Wd = np.random.uniform(-std, std, size=(hidden_size, latent_size))
Bd = np.random.uniform(-std, std, size=(hidden_size, 1))

# decoded hidden to output
Wo = np.random.uniform(-std, std, size=(input_size, hidden_size))
Bo = np.random.uniform(-std, std, size=(input_size, 1))


def forward(input):

    # YOUR FORWARD PASS FROM HERE
    batch_size = input.shape[-1]

    if input.ndim == 1:
        input = np.expand_dims(input, axis=1)

    # (1) linear
    # H = W_i \times input + Bi
    h = np.dot(Wi,input)+Bi

    # (2) ReLU
    # H = ReLU(H)
    h = relu(h)

    # (3) h > mu
    # Estimate the means of the latent distributions
    # mean = Wm \times H + Bm
    mean = np.dot(Wm,h)+Bm

    # (4) h > log var
    # Estimate the (diagonal) variances of the latent distributions
    # logvar = Wv \times H + Bv
    logvar = np.dot(Wv,h)+Bv

    # (5) sample the random variable z from means and variances (refer to the "reparameterization trick" to do this)
    # eps is the random number needed to stay the same for backprop
    eps = np.random.standard_normal(size=latent_size)
    sample_z = mean+np.dot(eps.T,np.exp(logvar*0.5))


    # (6) decode z
    # D = Wd \times z + Bd
    d = np.dot(Wd,sample_z)+Bd

    # (7) relu
    # D = ReLU(D)
    d = relu(d)

    # (8) dec to output
    # output = Wo \times D + Bo
    output = np.dot(Wo,d)+Bo

    # # (9) dec to p(x)
    # and (10) reconstruction loss function (same as the
    if loss_function == 'bce':
        # BCE Loss
        p = sigmoid(output)
        rec_loss = -np.sum(np.multiply(input, np.log(p)) + np.multiply(1 - input, np.log(1 - p)))
        # KL Divergence
        kl_div_loss = -0.5*np.sum(1+logvar-np.power(mean,2)-np.exp(logvar))


    elif loss_function == 'mse':
        # MSE Loss
        p = output
        rec_loss = np.sum(0.5 * (p - input) ** 2)
        kl_div_loss = 0

    loss = rec_loss + kl_div_loss
    # variational loss with KL Divergence between P(z|x) and U(0, 1)

    #kl_div_loss = - 0.5 * (1 + logvar - mean^2 - e^logvar)

    # your loss is the combination of
    #loss = rec_loss + kl_div_loss

    # Store the activations for the backward pass
    # activations = ( ... )
    activations = (eps,h,mean,logvar,sample_z,d,output,p,rec_loss,kl_div_loss)

    return loss, activations


def decode(z):

    # basically the decoding part in the forward pass: maaping z to p
    # o = W_d \times z + B_d
    # p = sigmoid(o) if bce or o if mse
    dec = np.dot(Wd, z) + Bd
    dec = relu(dec)
    output = np.dot(Wo, dec) + Bo
    if loss_function == 'bce':
        p = sigmoid(output)

    elif loss_function == 'mse':
        p = output

    return p


def backward(input, activations, scale=True, alpha=1.0):
    # allocating the gradients for the weight matrice
    dWi = np.zeros_like(Wi)
    dWm = np.zeros_like(Wm)
    dWv = np.zeros_like(Wv)
    dWd = np.zeros_like(Wd)
    dWo = np.zeros_like(Wo)
    dBi = np.zeros_like(Bi)
    dBm = np.zeros_like(Bm)
    dBv = np.zeros_like(Bv)
    dBd = np.zeros_like(Bd)
    dBo = np.zeros_like(Bo)

    batch_size = input.shape[-1]
    scaler = batch_size if scale else 1

    eps, h, mean, logvar, z, dec, output, p ,_,_ = activations

    # Perform your BACKWARD PASS (similar to the auto-encoder code)

    # 1st Note:
    # When performing the BW Pass for mean and logvar, note that they should have 2 different terms
    # One coming from the reconstruction loss, and backprop-ed through the hidden layer z
    # One coming from the KL divergence loss
    # So you should sum them up to have the correct gradient

    # 2nd Note:
    # The z is a sample from the distribution P(z|x), this is backprop-able based on the reparameterization trick
    # In order to do that, one random variable must stay the same between forward and backward passes.---->eps

    # The rest of the backward pass should be the same as the AE

    # from loss to output
    if loss_function == 'mse':
        dl_dp = p - input

        # I found that normalizing the loss and gradient by batch size makes learning more stable
        if scale:
            dl_dp = dl_dp / batch_size
        dp_doutput = 1
        dl_doutput = np.multiply(dl_dp,dp_doutput)

    elif loss_function == 'bce':
        dl_dp = -1 * (input / p - (1 - input) / (1 - p))
        dl_dp = dl_dp
        if scale:
            dl_dp = dl_dp / batch_size
        dl_doutput = np.multiply(dl_dp, dsigmoid(p))

    # from output to dec
    # or dl_ddec = Wo.T.dot(dl_doutput)
    dl_ddec = np.dot(Wo.T, dl_doutput)
    dWo += np.dot(dl_doutput, dec.T)
    if batch_size == 1:
        dBo += dl_doutput
    else:
        dBo += np.sum(dl_doutput, axis=-1, keepdims=True)

    # from dec to sample_z
    # through relu 
    dl_ddec = np.multiply(dl_ddec,drelu(dec))

    dl_dz = np.dot(Wd.T, dl_ddec)
    dWd += np.dot(dl_ddec, z.T)
    if batch_size == 1:
        dBd += dl_ddec
    else:
        dBd += np.sum(dl_ddec, axis=-1, keepdims=True)

    ######################
    # from sample_z to mean and logvar
    ######################
    dz_dmean = 1
    dl_dmean = dl_dz
    dl_dvar = np.multiply(dl_dz,np.dot(eps.T,np.exp(logvar*0.5))*0.5)

    ######################
    # from mean to h(before relu)
    ######################
    dl_dh = np.dot(Wm.T,dl_dmean)+np.dot(Wv.T,dl_dvar)

    dWm += np.dot(dl_dmean,h.T)
    dWv += np.dot(dl_dvar,h.T)
    if batch_size == 1:
        dBm += dl_dmean
        dBv += dl_dvar
    else:
        dBm += np.sum(dl_dmean, axis=-1, keepdims=True)
        dBv += np.sum(dl_dvar, axis=-1, keepdims=True)

    ######################
    # from h yo input
    ######################
    # through relu
    dl_dh = np.multiply(dl_dh,drelu(h))

    dWi += np.dot(dl_dh,input.T)
    if batch_size == 1:
        dBi += dl_dh
    else:
        dBi += np.sum(dl_dh,axis=-1,keepdims=True)

    ######################
    # Loss due to KL-Divergence
    ######################
    # KL Loss to logvar
    dKL_dvar = -0.5*(np.ones_like(logvar)-np.exp(logvar))
    if batch_size == 1:
        dKL_dBv = dKL_dvar
    else:
        dKL_dBv = np.sum(dKL_dvar,axis=-1,keepdims=True)
    dKL_dWv = np.dot(dKL_dvar,h.T)

    # kl loss to mean
    dKL_dmean = mean
    if batch_size == 1:
        dKL_dBm = dKL_dmean
    else:
        dKL_dBm = np.sum(dKL_dmean,axis=-1,keepdims=True)
    dKL_dWm = np.dot(dKL_dmean,h.T)

    # kl loss to input
    dKL_dh = np.dot(Wm.T,dKL_dmean)+np.dot(Wv.T,dKL_dvar)
    # through relu
    dKL_dh = np.multiply(drelu(h),dKL_dh)
    if batch_size == 1:
        dKL_dBi = dKL_dh
    else:
        dKL_dBi = np.sum(dKL_dh,axis=-1,keepdims=True)
    dKL_dWi = np.dot(dKL_dh,input.T)
    
    # KL update gradient
    dWv += dKL_dWv
    dWi += dKL_dWi
    dWm += dKL_dWm
    dBv += dKL_dBv
    dBm += dKL_dBm
    dBi += dKL_dBi

    gradients = (dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo)

    return gradients


def train():
    # Momentums for adagrad
    mWi, mWm, mWv, mWd, mWo = np.zeros_like(Wi), np.zeros_like(Wm), np.zeros_like(Wv),\
                              np.zeros_like(Wd), np.zeros_like(Wo)

    mBi, mBm, mBv, mBd, mBo = np.zeros_like(Bi), np.zeros_like(Bm), np.zeros_like(Bv), \
                              np.zeros_like(Bd), np.zeros_like(Bo)

    # Velocities for Adam
    vWi, vWm, vWv, vWd, vWo = np.zeros_like(Wi), np.zeros_like(Wm), np.zeros_like(Wv), \
                              np.zeros_like(Wd), np.zeros_like(Wo)

    vBi, vBm, vBv, vBd, vBo = np.zeros_like(Bi), np.zeros_like(Bm), np.zeros_like(Bv), \
                              np.zeros_like(Bd), np.zeros_like(Bo)

    def save_weights():

        print("Saving weights to %s and moments to %s" % ('weights.vae.pkl', 'momentums.vae.pkl'))

        weights = (Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo)
        with open('models/weights.vae.pkl', 'wb') as output:
            pickle.dump(weights, output, pickle.HIGHEST_PROTOCOL)

        momentums = (mWi, mWm, mWv, mWd, mWo, mBi, mBm, mBv, mBd, mBo)
        with open('models/momentums.vae.pkl', 'wb') as output:
            pickle.dump(momentums, output, pickle.HIGHEST_PROTOCOL)

        return

    batch_size = 128
    n_epoch = 2000 # origin 100000

    save_every = 2000

    # first we have to shuffle the data
    n_samples = ff.shape[0]
    indices = np.arange(n_samples)
    total_loss = 0
    total_kl_loss = 0
    total_pixels = 0
    total_samples = 0
    count = 0
    alpha = 0.0

    n_minibatch = math.ceil(n_samples / batch_size)
    for epoch in range(n_epoch):

        rand_indices = np.random.permutation(indices)

        for i in range(n_minibatch):

            x_i = get_minibatch(batch_size, i, rand_indices)
            bsz = x_i.shape[-1]

            loss, acts = forward(x_i)
            _, _, _, _, z, _, _, _, rec_loss, kl_loss = acts
            # lol I computed kl_div again here

            total_loss += rec_loss
            total_kl_loss += kl_loss
            total_pixels += bsz * 560

            gradients = backward(x_i, acts, alpha=alpha)

            dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo= gradients

            count += 1

            # perform parameter update with Adagrad
            # perform parameter update with Adam
            for param, dparam, mem, velo in zip([Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo],
                                            [dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo],
                                            [mWi, mWm, mWv, mWd, mWo, mBi, mBm, mBv, mBd, mBo],
                                            [vWi, vWm, vWv, vWd, vWo, vBi, vBm, vBv, vBd, vBo]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

                # Adam update
                # bias_correction1 = 1 - beta1 ** count
                # bias_correction2 = 1 - beta2 ** count
                #
                # mem = mem * beta1 + (1 - beta1) * dparam
                # velo = velo * beta2 + (1 - beta2) * dparam * dparam
                # denom = np.sqrt(velo) / math.sqrt(bias_correction2) + 1e-9
                # step_size = learning_rate / bias_correction1
                #
                # param += -step_size * mem / denom

            total_samples += bsz  # lol it can be total_pixels / 560

            if count % 50 == 0:
                avg_loss = total_loss / total_pixels
                avg_kl = total_kl_loss / total_samples
                print("Epoch %d Iteration %d Updates %d Loss per pixel %0.6f avg KLDIV %0.6f " %
                      (epoch, i, count, avg_loss, avg_kl))

            # save weights to file every 500 updates so we can load to visualize later
            if count % 500 == 0:
                save_weights()

    return


def grad_check():
    batch_size = 8
    delta = 0.0001

    x = get_minibatch(batch_size)

    actual_bsz = x.shape[-1]  # because x can be the last batch in the dataset which has bsz < 8

    loss, acts = forward(x)

    gradients = backward(x, acts, scale=False)

    dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo = gradients

    for weight, grad, name in zip([Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo],
                                  [dWi, dWm, dWv, dWd, dWo, dBi, dBm, dBv, dBd, dBo],
                                  ['Wi', 'Wm', 'Wv', 'Wd', 'Wo', 'Bi', 'Bm', 'Bv', 'Bd', 'Bo']):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert (weight.shape == grad.shape), str_

        print("Checking grads for weights %s ..." % name)
        n_warnings = 0
        for i in range(weight.size):

            w = weight.flat[i]

            weight.flat[i] = w + delta
            loss_positive, _ = forward(x)

            weight.flat[i] = w - delta
            loss_negative, _ = forward(x)

            weight.flat[i] = w  # reset old value for this parameter

            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / (2 * delta)

            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)

            if rel_error > 0.001:
                n_warnings += 1
                # print('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
        print("%d gradient mismatch warnings found. " % n_warnings)

    return


def eval():
    while True:

        # read weights from file
        cmd = input("Enter an image number:  ")

        img_idx = int(cmd)

        if img_idx < 0:
            exit()

        fig = plt.figure(figsize=(2, 2))
        n_samples = 1

        sample_ = ff[img_idx]
        org_img = sample_ * 255
        sample_ = np.resize(sample_, (1, 560)).T

        # Here the sample_ is processed by the network to produce the reconstruction

        img = np.sum(p, axis=-1)
        img = img / n_samples

        fig.add_subplot(1, 2, 1)
        plt.imshow(org_img.reshape(28, 20), cmap='gray')

        fig.add_subplot(1, 2, 2)
        plt.imshow(img.reshape(28, 20), cmap='gray')
        plt.show(block=True)

        print("Done")


def sample():
    while True:
        cmd = input("Press anything to continue:  ")

        sample_number = 10
        z_set = []
        for i in range(sample_number):
            z = np.random.randn(latent_size)
            z = np.expand_dims(z, 1)
            z_set.append(z)

        img_set = []
        # The decode function should be implemented before this
        for i in range(sample_number):
            p = decode(z_set[i])
            img_set.append(p)

        fig = plt.figure(figsize=(14,10))
        # gs = gridspec.GridSpec(4, 4)
        # gs.update(wspace=0.05, hspace=0.05)
        for i in range(1,sample_number+1):
            fig.add_subplot(5,2,i)
            plt.imshow(img_set[i-1].reshape(28, 20), cmap='gray')
        # plt.title('reconstructed face %d' % 0)
        plt.show(block=True)


if len(sys.argv) != 2:
    print("Need an argument train or gradcheck or reconstruct")
    exit()

option = sys.argv[1]

if option == 'train':
    train()
elif option in ['grad_check', 'gradcheck']:
    grad_check()
elif option in ['eval', 'sample']:

    # read trained weights from file
    with open('models/weights.vae.pkl', "rb") as f:
        weights = pickle.load(f)

    Wi, Wm, Wv, Wd, Wo, Bi, Bm, Bv, Bd, Bo = weights

    if option == 'eval':
        eval()
    else:
        sample()
else:
    raise NotImplementedError
