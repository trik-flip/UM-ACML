module NeuralNet
function print_weights(weights)
    for weight in weights
        println(weight, "\n")
    end
end
function print_trained_comparison(in_size, inputs, trained_weights)
    for i in 1:in_size
        in = inputs[1:in_size, i]
        _, a = NeuralNet.forward_propagation(in, trained_weights)
        println("prediction:", a[end], "\nTrue:", in, "\n")
    end
end
function generate_input(size)
    inputs = zeros(size, size)
    for i in 1:size
        inputs[i, i] = 1
    end
    return inputs
end
function forward_propagation(l, weights)
    Z = []
    A = []
    for w in weights
        l = [1, l...]
        l = w' * l
        Z = [Z..., l]
        l = σ(l)
        A = [A..., l]
    end
    return Z, A
end
function back_propagation(in, w, out)
    ∂′s = [map(x -> zeros(size(x)[2]), w)...]

    Z, A = forward_propagation(in, w)

    H = length(w)
    for l in H:-1:1
        if l == H
            ∂ = cost(A[l], out) .* σ′(Z[l])
        else
            ∂ = (w[l+1]*∂′s[l+1])[2:end] .* σ′(Z[l])
        end
        ∂′s[l] = ∂
    end
    return ∂′s
end
function gradient_descent(in, w, out, alpha)
    _, A = forward_propagation(in, w)
    ∂ = back_propagation(in, w, out)
    H = length(w)
    for l in H:-1:1
        if l == 1
            w[l] = w[l] .- alpha * [∂[l]'; [in * ∂[l]']...]
        else
            w[l] = w[l] .- alpha * [∂[l]'; A[l-1] * ∂[l]']
        end
    end

    return w
end
using Plots
function train_weights(in_gen, w, alpha, epochs)
    count = 1
    err = []
    for _ in 1:epochs
        for i in in_gen
            count += 1
            w = gradient_descent(i, w, i, alpha)
            _, A = forward_propagation(i, w)
            new_err = log(sum(abs.(A[end] .- i)))
            err = [err..., new_err]
        end
    end

    return w, err

end

function train_weights(in, out, w, alpha, epochs)
    for _ in 1:epochs
        for (i, o) in zip(in, out)
            w = gradient_descent(i, w, o, alpha)
        end
    end
    return w
end

data_set_generator(amount::Int, in) = (in[1:end, rand(1:size(in)[1])] for _ in 1:amount)
cost(in, out) = (in - out)

σ(x::Float64) = 1 / (1 + exp(-x))
σ(x::Vector{Float64}) = map(y -> σ(y), x)

σ′(x::Float64) = σ(x) * (1 - σ(x))
σ′(x::Vector{Float64}) = map(y -> σ′(y), x)


function NN(
    structure
)
    # Generate setup

    weights = []
    for i in 1:length(structure)-1
        w = rand(structure[i] + 1, structure[i+1])
        weights = [weights..., w]
    end
    return weights
end



end