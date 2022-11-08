module AutoEncoder
include("NeuralNets.jl")
using Plots
function julia_main(alpha, data_set_size)::Cint
    # parameters
    in_size = 8
    epochs = 1

    # Init the NN
    w = NeuralNet.NN((in_size, 3, in_size))

    # Prepare training data
    inputs = NeuralNet.generate_input(in_size)
    ds = NeuralNet.data_set_generator(data_set_size, inputs)

    # Train the NN
    w, err = NeuralNet.train_weights(ds, w, alpha, epochs)

    plt = plot(1:length(err), err, title="convergance, alpha:$alpha")
    display(plt)
    # Show the trained model
    display(w)
    plt = heatmap(w[1], yflip=true, title="w1, alpha:$alpha, training_size: $(epochs*data_set_size)")
    display(plt)
    plt = heatmap(w[2], yflip=true, title="w2, alpha:$alpha, training_size: $(epochs*data_set_size)")
    display(plt)
    NeuralNet.print_trained_comparison(in_size, inputs, w)

    return 0
end
end
AutoEncoder.julia_main(1, 4000)