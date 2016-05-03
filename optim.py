import numpy
import theano
from theano import tensor



def main():
    # problem settings
    N = 10000 # n voxels
    # TODO: replace C with the correct constants
    C = numpy.random.randn(12).astype('float32')
    # TODO: replace Y with the correct targets
    Y = numpy.random.rand(N).astype('float32')

    # target variables 
    X_ = numpy.random.rand(N, 4).astype('float32')
    X = theano.shared(X_, name='X')

    # groundtruth
    Y_ = theano.shared(Y, name='Y')

    # build a forward computation
    # (note that python uses zero-based index)
    Y_hat = X[:,1] * (C[0] * C[1] * (X[:,2] - C[2])) + \
            C[3] * (C[4] * (X[:,2] - C[5])) + \
            (1 - tensor.exp(X[:,2] * C[6])) * (C[7] - X[:,2]) * tensor.exp(-X[:,2] * C[8]) / \
            (-X[:,2] * (C[9] - X[:,2]) * (C[10] - X[:,2])) \
            + X[:,3] * C[11]

    # cost function
    J = ((Y_ - Y_hat) ** 2).sum()

    # compute the gradient
    dJ = tensor.grad(J, X)

    # build a function for optimization
    step_size = tensor.scalar('step_size', dtype='float32')
    f_up = theano.function([step_size], [J], updates={(X, X - step_size * dJ)})

    # optimization settings
    max_iter = 10000
    step_size0 = .01

    disp_freq = 10

    for n_updates in xrange(max_iter):
        step_size = step_size0 / (n_updates + 1)
        J_curr = f_up(step_size)

        if numpy.mod(n_updates, disp_freq) == 0:
            print n_updates, 'Cost', J_curr

    print 'Done'

    solution = X.get_value()

    numpy.save('solution.npy', solution)

if __name__ == "__main__":
    main()

